import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import pickle
import random
from glob import glob
from os.path import exists, join
import argparse
import numpy as np
import torch
import sklearn, sklearn.model_selection
from tqdm import tqdm as tqdm_base
from utils.train_utils import IndexDataset
import torch.nn as nn
from torchvision import models
from torchvision.models.feature_extraction import create_feature_extractor
import time
import torch.nn.functional as F
from utils.dataset import CIFARDataset,PlantNet300KLoader,INAT,Food101Dataset
from utils.model import create_model
from peft import get_peft_model, LoraConfig, TaskType

parser = argparse.ArgumentParser()


class RandomLabelNoiseDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, noise_ratio, num_classes, seed):
        self.dataset = dataset
        self.noise_ratio = noise_ratio
        self.num_classes = num_classes
        self.rng = np.random.default_rng(seed)
        self.noise_indices = self.rng.choice(
            len(dataset),
            size=int(noise_ratio * len(dataset)),
            replace=False
        )
        self.noise_index_set = set(self.noise_indices.tolist())

        clean_targets = [self._get_target(i) for i in range(len(dataset))]
        self.targets = list(clean_targets)
        for idx in self.noise_indices:
            clean_label = int(clean_targets[idx])
            random_offset = int(self.rng.integers(1, num_classes))
            self.targets[idx] = (clean_label + random_offset) % num_classes
        self.labels = self.targets

    def _get_target(self, idx):
        if hasattr(self.dataset, "_labels"):
            return self.dataset._labels[idx]
        if hasattr(self.dataset, "targets"):
            return self.dataset.targets[idx]
        _, target = self.dataset[idx]
        return target

    def __getitem__(self, idx):
        image, _ = self.dataset[idx]
        return image, self.targets[idx]

    def __len__(self):
        return len(self.dataset)

    def __getattr__(self, name):
        return getattr(self.dataset, name)

parser.add_argument('--lora', type=int, default=0, help='')
parser.add_argument('--finetune_runs', type=int, default=3, help='')
parser.add_argument('--enhancement', type=int, default=1, help='')
parser.add_argument('--dataset_dir', type=str, default='../data',
                    help='Root directory containing the datasets; override this path for your environment.')
parser.add_argument('--td-path', type=str, default='../data/',
                    help='The dir path of the data.')
parser.add_argument('--output_dir', type=str, default="")
parser.add_argument('--name', type=str,default="")
parser.add_argument('--dataset', type=str, default="nih")
parser.add_argument('--seed', type=int, default=0, help='')
parser.add_argument('--batch_size', type=int, default=16, help='')
parser.add_argument('--cuda', type=int, default=1, help='')
parser.add_argument('--load_target', type=bool, default=False, help='')
parser.add_argument('--model', type=str, default="resnet18")
parser.add_argument('--gpuid', type=str, default='1',help='The ID of GPU.')
parser.add_argument('--imbfactor', type=float, default=0.1, help='')
parser.add_argument('--label_noise_ratio', type=float, default=0, help='Random label noise ratio for Food101 scoring labels.')
cfg = parser.parse_args()

np.random.seed(cfg.seed)
random.seed(cfg.seed)
torch.manual_seed(cfg.seed)
if cfg.cuda:
    torch.cuda.manual_seed_all(cfg.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

task_dir = os.path.join(cfg.output_dir, cfg.name)

td_path = os.path.join(task_dir, f'td-{cfg.name}.pickle')
data_score_path = os.path.join(task_dir, f'data-score-{cfg.name}.pickle')

###
GPUID = cfg.gpuid
os.environ["CUDA_VISIBLE_DEVICES"] = GPUID
###
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
with open(td_path, 'rb') as f:
     pickled_data = pickle.load(f)
     
training_dynamics = pickled_data['training_dynamics']

dataset = cfg.dataset
if cfg.dataset in ['cifar10', 'svhn', 'cinic10','imbcifar10']:
    num_classes=10
elif cfg.dataset in ['cifar100','imbcifar100']:
    num_classes=100
elif cfg.dataset in ['food']:   
    num_classes=101
    
print(num_classes)
data_dir =  os.path.join(cfg.dataset_dir, dataset)

print(f'dataset: {dataset}')
if dataset == 'cifar10':
    train_dataset = CIFARDataset.get_cifar10_train(data_dir)
elif dataset == 'cifar100':
    train_dataset = CIFARDataset.get_cifar100_train(data_dir)
elif cfg.dataset == 'imbcifar10':
    train_dataset = CIFARDataset.get_imbalanced_cifar10_train(
    path=data_dir, 
    imb_factor=cfg.imbfactor,  
    transform=None)
elif cfg.dataset == 'imbcifar100':
    train_dataset = CIFARDataset.get_imbalanced_cifar100_train(
    path=data_dir, 
    imb_factor=cfg.imbfactor,  
    transform=None)
elif cfg.dataset == 'food':
    train_dataset = Food101Dataset.get_food101_train(path=cfg.dataset_dir)

if not 0 <= cfg.label_noise_ratio <= 1:
    raise ValueError("--label_noise_ratio must be between 0 and 1.")
if cfg.label_noise_ratio > 0:
    if cfg.dataset != 'food':
        raise ValueError("--label_noise_ratio is currently intended for Food101 experiments; set --dataset food.")
    train_dataset = RandomLabelNoiseDataset(
        dataset=train_dataset,
        noise_ratio=cfg.label_noise_ratio,
        num_classes=num_classes,
        seed=cfg.seed
    )
    print(f"Applied random label noise: {len(train_dataset.noise_indices)}/{len(train_dataset)} "
          f"({cfg.label_noise_ratio:.2%}) Food101 scoring labels.")
    
train_dataset = IndexDataset(train_dataset)  ####
train_loader = torch.utils.data.DataLoader(train_dataset,
                                            batch_size=cfg.batch_size,
                                            shuffle=False,
                                            num_workers=12, 
                                            pin_memory=True)
data_importance = {}

def training_dynamics_metrics(td_log, dataset, data_importance):
    targets = []
    data_size = len(dataset)

    for i in range(data_size):
        _, (_, y) = dataset[i]
        targets.append(y)
    targets = torch.tensor(targets)
    data_importance['targets'] = targets.type(torch.int32)

    data_importance['correctness'] = torch.zeros(data_size).type(torch.int32)
    data_importance['forgetting'] = torch.zeros(data_size).type(torch.int32)
    data_importance['last_correctness'] = torch.zeros(data_size).type(torch.int32)
    data_importance['accumulated_margin'] = torch.zeros(data_size).type(torch.float32)

    def record_training_dynamics(td_log):
        output = torch.exp(td_log['output'].type(torch.float))
        predicted = output.argmax(dim=1)
        index = td_log['idx'].type(torch.long)

        label = targets[index]

        correctness = (predicted == label).type(torch.int)
        data_importance['forgetting'][index] += torch.logical_and(data_importance['last_correctness'][index] == 1, correctness == 0)
        data_importance['last_correctness'][index] = correctness
        data_importance['correctness'][index] += data_importance['last_correctness'][index]

        batch_idx = range(output.shape[0])
        target_prob = output[batch_idx, label]
        output[batch_idx, label] = 0
        other_highest_prob = torch.max(output, dim=1)[0]
        margin = target_prob - other_highest_prob
        data_importance['accumulated_margin'][index] += margin

    for i, item in enumerate(td_log):
        if i % 200 == 0:
            print(i)
        record_training_dynamics(item)


def EL2N(td_log, dataset, data_importance, max_epoch, cfg):
    data_size = len(dataset)
    targets = []
    for i in range(data_size):
        _, (_, y) = dataset[i]
        y_one_hot = F.one_hot(torch.tensor([y]), num_classes=num_classes)
        targets.append(y_one_hot.squeeze(0))  
    
    targets = torch.stack(targets).to(device="cpu")
    data_importance['targets'] = targets 
    
    data_importance['el2n'] = torch.zeros(data_size, device="cpu").type(torch.float32)  
    l2_loss = torch.nn.MSELoss(reduction='none')

    def record_training_dynamics(td_log,cfg):
        output = torch.exp(td_log['output'].type(torch.float)).to(device="cpu")  
        index = td_log['idx'].type(torch.long).to(device="cpu")  
        label = targets[index] 
        label_num = label.shape[1]
       # label_onehot = torch.nn.functional.one_hot(label, num_classes=num_classes)
        el2n_score = torch.sqrt(l2_loss(label,output).sum(dim=1))

        data_importance['el2n'][index] += el2n_score

    print("recording")
    for i, item in enumerate(td_log):
        if i % 200 == 0:
            print(i)
        if item['epoch'] == max_epoch:
            return
        record_training_dynamics(item,cfg)


def calculate_data_importance(cfg, model, device, train_loader, modelname):
    data_size = len(train_loader.dataset)
    data_importance['effort'] = torch.zeros(data_size, device="cpu").type(torch.float32)
    
    if 'resnet' in modelname:
        feature_dim = model.fc.in_features
    elif 'vit' in modelname:
        feature_dim = model.config.hidden_size
    else:
        raise ValueError(f"Unsupported model type: {modelname}")
    
    data_importance['feature'] = torch.zeros((data_size, feature_dim), device="cpu").type(torch.float32)
    model.eval()
    criterion = nn.CrossEntropyLoss()

    features = []
    def hook(module, input, output):
        if 'resnet' in modelname:
            features.append(output.clone().detach().cpu().view(output.size(0), -1))
        elif 'vit' in modelname:
            cls_token_features = output.last_hidden_state[:, 0, :]
            features.append(cls_token_features.clone().detach().cpu())
    
    if 'resnet' in modelname:
        handle = model.avgpool.register_forward_hook(hook)
    elif 'vit' in modelname:
        handle = model.vit.encoder.register_forward_hook(hook)
    else:
        raise ValueError(f"Unsupported model type: {modelname}")

    for batch_idx, (idx, (inputs, targets)) in enumerate(train_loader):
        if batch_idx % 50 == 0:
            print(f"Processing batch {batch_idx}/{len(train_loader)}")
        
        inputs = inputs.to(device)
        targets = targets.to(device)

        with torch.no_grad():
            _ = model(inputs)
            if features:
                batch_features = features[-1].cpu()  
                if batch_features.size(1) != feature_dim:
                    raise RuntimeError(
                        f"Shape mismatch: batch_features has shape {batch_features.shape}, "
                        f"but expected shape [..., {feature_dim}]"
                    )
                data_importance['feature'][idx] = batch_features
                features.clear()

        for i in range(inputs.size(0)):
            input_single = inputs[i:i+1].clone().detach().requires_grad_(True)
            target_single = targets[i:i+1].clone().detach()
            output_single = model(input_single)
            if 'vit' in modelname:
                output_single = output_single.logits
            single_loss = criterion(output_single, target_single)
            model.zero_grad()
            single_loss.backward()
            grad_norm = torch.norm(
                torch.stack(
                    [torch.norm(param.grad) for param in model.parameters() if param.grad is not None]
                )
            ).cpu()  
            data_importance['effort'][idx[i]] = grad_norm
    
    handle.remove()

    return data_importance


def calculate_feature_nofinetune(cfg, model, device, train_loader, modelname):
    data_size = len(train_loader.dataset)
    
    if 'resnet' in modelname:
        feature_dim = model.fc.in_features
    elif 'vit' in modelname:
        feature_dim = model.config.hidden_size  
    else:
        raise ValueError(f"Unsupported model type: {modelname}")
    
    data_importance['feature_nofinetune'] = torch.zeros((data_size, feature_dim), device="cpu").type(torch.float32)
    model.eval()  # 

    features = []
    def hook(module, input, output):
        if 'resnet' in modelname:
            features.append(output.clone().detach().cpu().view(output.size(0), -1))
        elif 'vit' in modelname:
            cls_token_features = output.last_hidden_state[:, 0, :]
            features.append(cls_token_features.clone().detach().cpu())
    
    if 'resnet' in modelname:
        handle = model.avgpool.register_forward_hook(hook)
    elif 'vit' in modelname:
        handle = model.vit.encoder.register_forward_hook(hook)  
    else:
        raise ValueError(f"Unsupported model type: {modelname}")
    
    for batch_idx, (idx, (inputs, targets)) in enumerate(train_loader):
        inputs = inputs.to(device)
        features.clear()
        
        with torch.no_grad():  
            _ = model(inputs)
        
        if features:
            batch_features = features[-1]
            data_importance['feature_nofinetune'][idx] = batch_features 
        
        torch.cuda.empty_cache()  
        
    handle.remove()
    
    return data_importance



dataset_name = cfg.dataset + "-" + cfg.model + "-" + cfg.name
finetune_path = join(task_dir, f'{dataset_name}-finetune.pt')
model_finetune_path = join(task_dir, f'{dataset_name}-modelfinetune.pt')
#####
model = torch.load(finetune_path)
model = model.to(device)
#####
model2 = create_model(
    model_name=cfg.model,
    num_classes=num_classes,
    imgpretrain=True,
    device=device
)
lora_config = LoraConfig(
    r=8,  #
    lora_alpha=32,  # 
    lora_dropout=0.1,  # 
    target_modules=["query", "value"],  # 
    modules_to_save=["classifier"]
)
if cfg.lora==1:
    print('use lora')
    model2 = get_peft_model(model2, lora_config)

print('training dynamic metrics')
training_dynamics_metrics(training_dynamics, train_dataset, data_importance)


print("el2n metric")
start_time = time.time()
EL2N(training_dynamics, train_dataset, data_importance, max_epoch=cfg.finetune_runs, cfg=cfg)
el2n_time = time.time() - start_time
print(f"EL2N process took {el2n_time:.2f} seconds.")


start_time = time.time()
calculate_data_importance(cfg=cfg, model=model, device=device, train_loader=train_loader, modelname=cfg.model)
calculate_data_importance_time = time.time() - start_time
print(f"effort and feature process took {calculate_data_importance_time:.2f} seconds.")

start_time = time.time()
calculate_feature_nofinetune(cfg=cfg, model=model2, device=device, train_loader=train_loader, modelname=cfg.model)
calculate_feature_nofinetune_time = time.time() - start_time
print(f"feature process took {calculate_feature_nofinetune_time:.2f} seconds.")
print(f'Saving data score at {data_score_path}')

with open(data_score_path, 'wb') as handle:
    pickle.dump(data_importance, handle)
print("save!")
