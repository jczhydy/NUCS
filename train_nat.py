import os,sys
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
sys.path.insert(0,".")
from glob import glob
from os.path import exists, join
import matplotlib.pyplot as plt
import numpy as np
import argparse
import torch
import time
import torch.nn as nn
import random
from utils.train_utils import train_epoch_nat,test_epoch,TrainingDynamicsLogger, IndexDataset
from utils.util import StdRedirect
import pickle
from coreset import CoresetSelection
from torch.utils.data import Subset
from utils.dataset import CIFARDataset,Food101Dataset
from utils.model import create_model
from peft import get_peft_model, LoraConfig, TaskType
import copy
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

parser.add_argument('--data_score_path_A', type=str, help='Path to the smaller model scores (ScoreA)')
parser.add_argument('--beta', type=float, default=1, help='')
parser.add_argument('--alpha', type=float, default=1, help='')
parser.add_argument('--wf', type=float, default=0.05, help='')
parser.add_argument('--norml', type=float, default=0, help='')
parser.add_argument('--normr', type=float, default=1, help='')
parser.add_argument('--krrmode', type=str, default='all',help='all,class')
parser.add_argument('--criterion', type=str, default='ce')

##########################
parser.add_argument('--num_runs', type=int, default=1, help='')
parser.add_argument('--lora', type=int, default=0, help='')
parser.add_argument('--model_finetune_runs', type=int, default=3, help='')
parser.add_argument('--finetune_runs', type=int, default=3, help='')
parser.add_argument('--patience', type=int, default=3, help='')
parser.add_argument('--centerratio', type=float, default=0, help='')
parser.add_argument('--classratiomode', type=str, default='difficulty', help='difficulty,number')
parser.add_argument('--classselectmode', type=str, default='ccs', help='ccs,window')
parser.add_argument('--enhancement', type=int, default=0, help='')
parser.add_argument('--noratio', type=float, default=1, help='')
parser.add_argument('--endratio', type=float, default=1, help='')
parser.add_argument('--finetune_only_head', type=int, default=0, help='')
parser.add_argument('--pretrainpath', type=str, default='', help='')
parser.add_argument('--imgpretrain', type=int, default=1, help='')
parser.add_argument('--sigma', type=float, default=2, help='')
parser.add_argument('--labelsigma', type=float, default=1, help='')
parser.add_argument('--label_mode', type=str,default='random')
parser.add_argument('--label_noise_ratio', type=float, default=0, help='Random label noise ratio for Food101 training labels.')
parser.add_argument('--imbfactor', type=float, default=0.1, help='')
####d2
parser.add_argument('--precomputed-dists', type=str, default='')
parser.add_argument('--precomputed-neighbors', type=str, default='')
parser.add_argument('--gamma', type=float, default=0, help='')
parser.add_argument('--n_neighbor', type=int, default=5, help='')
parser.add_argument('--graph_mode', type=str,default='sum')
parser.add_argument('--graph-sampling-mode', type=str,default='weighted')
####coreset
parser.add_argument('--supplement_mode', type=str,default='random')
parser.add_argument('--mis_ratio', type=float, default=0, help='')
parser.add_argument('--data_score_path', type=str)
parser.add_argument('--data_score_descending', type=int, default=0, help='')
parser.add_argument('--coreset_mode', type=str)
parser.add_argument('--coreset_key', type=str)
parser.add_argument('--coreset_ratio', type=float, default=1, help='')
parser.add_argument('--coreset', type=int, default=0, help='')

parser.add_argument('--td', type=int, default=0, help='')
parser.add_argument('--gpuid', type=str, default='1',help='The ID of GPU.')
parser.add_argument('-f', type=str, default="", help='')
parser.add_argument('--name', type=str,default="")
parser.add_argument('--output_dir', type=str, default="")
parser.add_argument('--dataset', type=str, default="nih")
parser.add_argument('--dataset_dir', type=str, default='../data',
                    help='Root directory containing the datasets; override this path for your environment.')
parser.add_argument('--model', type=str, default="resnet18",choices=['resnet18','resnet34','vit_b','vit_l'])
parser.add_argument('--loss', type=str, default="ce")
parser.add_argument('--seed', type=int, default=0, help='')
parser.add_argument('--cuda', type=bool, default=True, help='')
parser.add_argument('--num_epochs', type=int, default=30, help='')
parser.add_argument('--batch_size', type=int, default=64, help='')
parser.add_argument('--shuffle', type=bool, default=True, help='')
parser.add_argument('--lr', type=float, default=0.0001, help='')
parser.add_argument('--threads', type=int, default=4, help='')

cfg = parser.parse_args()
print("model:",cfg.model,"coreset_mode:",cfg.coreset_mode,"coreset_ratio:",cfg.coreset_ratio,'learning_rate:',cfg.lr)
if cfg.coreset_mode == 'graph':
    print('gamma:',cfg.gamma)
    
np.random.seed(cfg.seed)
random.seed(cfg.seed)
torch.manual_seed(cfg.seed)
if cfg.cuda:
    torch.cuda.manual_seed_all(cfg.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if not exists(cfg.output_dir):
    os.makedirs(cfg.output_dir)    

task_dir = os.path.join(cfg.output_dir, cfg.name)
os.makedirs(task_dir, exist_ok=True)
print(task_dir)
log_path = os.path.join(task_dir, f'log-train-{cfg.name}.log')
sys.stdout = StdRedirect(log_path)
td_path = os.path.join(task_dir, f'td-{cfg.name}.pickle')
data_score_path = os.path.join(task_dir, f'data-score-{cfg.name}.pickle')

GPUID = cfg.gpuid
os.environ["CUDA_VISIBLE_DEVICES"] = GPUID
device_id = list(map(int, GPUID.split(',')))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_dir = os.path.join(cfg.dataset_dir, cfg.dataset)
print('device:',device)

if cfg.dataset == 'cifar10':
    train_dataset = CIFARDataset.get_cifar10_train(data_dir)
    valid_dataset = CIFARDataset.get_cifar10_test(data_dir)
elif cfg.dataset == 'imbcifar10':
    train_dataset = CIFARDataset.get_imbalanced_cifar10_train(
    path=data_dir, 
    imb_factor=cfg.imbfactor,  
    transform=None)
    valid_dataset = CIFARDataset.get_imbalanced_cifar10_test(data_dir)
elif cfg.dataset == 'cifar100':
    train_dataset = CIFARDataset.get_cifar100_train(data_dir)
    valid_dataset = CIFARDataset.get_cifar100_test(data_dir)
elif cfg.dataset == 'imbcifar100':
    train_dataset = CIFARDataset.get_imbalanced_cifar100_train(
    path=data_dir, 
    imb_factor=cfg.imbfactor,  
    transform=None)
    valid_dataset = CIFARDataset.get_imbalanced_cifar100_test(data_dir)
elif cfg.dataset == 'food':
    train_dataset = Food101Dataset.get_food101_train(path=cfg.dataset_dir)
    valid_dataset = Food101Dataset.get_food101_test(path=cfg.dataset_dir)
    
if cfg.dataset in ['cifar10', 'svhn', 'cinic10','imbcifar10']:
    num_classes=10
elif cfg.dataset in ['cifar100','imbcifar100']:
    num_classes=100
elif cfg.dataset in ['food']:   
    num_classes=101

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
          f"({cfg.label_noise_ratio:.2%}) Food101 training labels.")
    
if cfg.lora == 1:
    lora_config = LoraConfig(
        r=8,  
        lora_alpha=32,  
        lora_dropout=0.1,  
        target_modules=["query", "value"],  
        modules_to_save=["classifier"]
    )

model = create_model(
    model_name=cfg.model,
    num_classes=num_classes,
    imgpretrain=cfg.imgpretrain,
    device=device
)

if cfg.lora==1:
    print('use lora')
    model = get_peft_model(model, lora_config)
#######    training
dataset_name = cfg.dataset + "-" + cfg.model + "-" + cfg.name
# Dataset
# split    
total_num = len(train_dataset)
coreset_start_time = time.time()
### coreset
if cfg.coreset:
    if cfg.coreset_mode != 'random':
        with open(cfg.data_score_path, 'rb') as f:
            data_score = pickle.load(f)    
    if cfg.coreset_mode == 'random':
        coreset_index = CoresetSelection.random_selection(total_num=len(train_dataset), num=cfg.coreset_ratio * len(train_dataset))
    elif cfg.coreset_mode == 'all_window':
        coreset_index = CoresetSelection.window_selection(data_score=data_score, ratio=cfg.coreset_ratio, key=cfg.coreset_key,endratio=cfg.endratio)
    elif cfg.coreset_mode == 'window':
        coreset_index = CoresetSelection.class_sample_nat(cfg=cfg,data_score=data_score, ratio=cfg.coreset_ratio, key=cfg.coreset_key, classratiomode='number', classselectmode='window', featuremode='finetune')
        #coreset_index = CoresetSelection.bws_nat(data_score=data_score, ratio=cfg.coreset_ratio, key=cfg.coreset_key, featuremode='feature_nofinetune')
    elif cfg.coreset_mode == 'hard':
        coreset_index = CoresetSelection.hard_selection(data_score=data_score, key=cfg.coreset_key, ratio=cfg.coreset_ratio, descending=False)
    elif cfg.coreset_mode == 'moderate':
        coreset_index = CoresetSelection.moderate(data_score=data_score, ratio=cfg.coreset_ratio)
    elif cfg.coreset_mode == 'labelsample':
        coreset_index = CoresetSelection.label_sample(data_score=data_score, ratio=cfg.coreset_ratio, key=cfg.coreset_key, mode=cfg.label_mode, cfg=cfg, mis_ratio=cfg.mis_ratio)
    elif cfg.coreset_mode == 'ccscp':
        coreset_index = CoresetSelection.label_sample(data_score=data_score, ratio=cfg.coreset_ratio, key=cfg.coreset_key, mode="ccs", cfg=cfg, mis_ratio=cfg.mis_ratio)
    elif cfg.coreset_mode == 'stratified':
        mis_num = int(cfg.mis_ratio * total_num)
        data_score_mask, score_index = CoresetSelection.mislabel_mask(data_score, mis_key=cfg.coreset_key, mis_num=mis_num, mis_descending=True, coreset_key=cfg.coreset_key)
        coreset_num = int(cfg.coreset_ratio * total_num)
        coreset_index, _ = CoresetSelection.stratified_sampling(data_score=data_score_mask, coreset_key=cfg.coreset_key, coreset_num=coreset_num)
        coreset_index = score_index[coreset_index]
    elif cfg.coreset_mode == 'stratified_staff':
        if cfg.data_score_path_A is None:
            raise ValueError("stratified_staff mode requires --data_score_path_A")
        with open(cfg.data_score_path_A, 'rb') as f:
            data_score_A = pickle.load(f)
        mis_num = int(cfg.mis_ratio * total_num)
        data_score_B_masked, easy_index = CoresetSelection.mislabel_mask(
            data_score, 
            mis_key=cfg.coreset_key, 
            mis_num=mis_num, 
            mis_descending=True, 
            coreset_key=cfg.coreset_key
        )
        data_score_A[cfg.coreset_key] = data_score_A[cfg.coreset_key][easy_index]
        current_available_num = len(easy_index)
        coreset_num = int(cfg.coreset_ratio * total_num)
        sub_index = CoresetSelection.stratified_staff(
            data_score_A=data_score_A, 
            data_score_B=data_score_B_masked, 
            key=cfg.coreset_key, 
            coreset_num=coreset_num
        )
        coreset_index = easy_index[sub_index]
    elif cfg.coreset_mode == 'graph':
        coreset_index = CoresetSelection.d2_selection(data_score=data_score, key=cfg.coreset_key, coreset_num=cfg.coreset_ratio * len(train_dataset), cfg=cfg)
    elif cfg.coreset_mode == 'labelweight':
        coreset_index = CoresetSelection.class_sample_nat(cfg=cfg, data_score=data_score, ratio=cfg.coreset_ratio, key=cfg.coreset_key,classratiomode='difficulty', classselectmode='ccs')
    elif cfg.coreset_mode == 'labelwindow':
        coreset_index = CoresetSelection.class_sample_nat(cfg=cfg, data_score=data_score, ratio=cfg.coreset_ratio, key=cfg.coreset_key,classratiomode='difficulty', classselectmode='window')
    elif cfg.coreset_mode == 'labelwindow_traverse':
        coreset_index = CoresetSelection.class_sample_nat_traverse(cfg=cfg, data_score=data_score, ratio=cfg.coreset_ratio, key=cfg.coreset_key,classratiomode='difficulty', classselectmode='window',centerratio=cfg.centerratio)
    elif cfg.coreset_mode == 'labelwindow_number_traverse':
        coreset_index = CoresetSelection.class_sample_nat_traverse(cfg=cfg, data_score=data_score, ratio=cfg.coreset_ratio, key=cfg.coreset_key,classratiomode='number', classselectmode='window',centerratio=cfg.centerratio)
    elif cfg.coreset_mode == 'labelwindow_number_traverse_norm':
        coreset_index = CoresetSelection.class_sample_nat_traverse_norm(cfg=cfg, data_score=data_score, ratio=cfg.coreset_ratio, key=cfg.coreset_key, classratiomode='difficulty', classselectmode='window', featuremode='nofinetune', centerratio=cfg.centerratio, norml=cfg.norml, normr=cfg.normr)
    elif cfg.coreset_mode == 'labelwindow_win':
        coreset_index = CoresetSelection.class_sample_nat(cfg=cfg, data_score=data_score, ratio=cfg.coreset_ratio, key=cfg.coreset_key,classratiomode='difficulty', classselectmode='window',difficultymode='winsorized', alpha=cfg.alpha)
    elif cfg.coreset_mode == 'labelwindow_win_traverse':
        coreset_index = CoresetSelection.class_sample_nat_traverse(cfg=cfg, data_score=data_score, ratio=cfg.coreset_ratio, key=cfg.coreset_key,classratiomode='difficulty', classselectmode='window',difficultymode='winsorized', centerratio=cfg.centerratio, wf=cfg.wf, alpha=cfg.alpha, beta=cfg.beta)
    elif cfg.coreset_mode == 'labelccs':
        coreset_index = CoresetSelection.class_sample_nat(cfg=cfg, data_score=data_score, ratio=cfg.coreset_ratio, key=cfg.coreset_key,classratiomode='difficulty', classselectmode='ccs')

    else:
        print('no coreset method!')
        
    train_dataset = Subset(train_dataset, coreset_index)

coreset_end_time = time.time()
coreset_time = coreset_end_time - coreset_start_time

print("traindata size:",len(train_dataset))
train_dataset = IndexDataset(train_dataset)
print("testdata size:",len(valid_dataset))
# Dataloader
train_loader = torch.utils.data.DataLoader(train_dataset,
                                          batch_size=cfg.batch_size,
                                          shuffle=cfg.shuffle,
                                          num_workers=cfg.threads, 
                                          pin_memory=True)
valid_loader = torch.utils.data.DataLoader(valid_dataset,
                                          batch_size=cfg.batch_size*2,
                                          shuffle=cfg.shuffle,
                                          num_workers=cfg.threads, 
                                          pin_memory=True)

optim = torch.optim.SGD(model.parameters(), lr=cfg.lr, momentum=0.9, weight_decay=5e-4, nesterov=True)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=cfg.num_epochs, eta_min=1e-5)
print(optim)

start_epoch = 0
metrics = []


if cfg.td == True:
    TD_logger = TrainingDynamicsLogger()
else:
    TD_logger = None


patience = cfg.patience  
min_delta = 0.00001  
best_metric = -np.inf  
patience_counter = 0  


criterion = torch.nn.CrossEntropyLoss()
print(criterion)
training_time = 0.0  
    
for epoch in range(start_epoch, cfg.num_epochs):
    print(f"Epoch {epoch + 1}:")
    print('Training:')
    
    epoch_train_start = time.time()
    if cfg.criterion == 'ce':
        avg_loss = train_epoch_nat(cfg=cfg,
                           epoch=epoch,
                           model=model,
                           device=device,
                           optimizer=optim,
                           train_loader=train_loader,
                           criterion=criterion,
                           TD_logger=TD_logger)
    epoch_train_end = time.time()
    print('Training time:',epoch_train_end - epoch_train_start)
    training_time += epoch_train_end - epoch_train_start
    
    print('Testing:')
    if scheduler:
        scheduler.step()
    
    mean_loss, accuracy = test_epoch(
        cfg=cfg,
        epoch=epoch,
        model=model,
        device=device,
        test_loader=valid_loader,
        criterion=criterion
    )
    if (epoch + 1 == cfg.finetune_runs and cfg.coreset == 0):
        if cfg.lora == 1:
            model_copy = copy.deepcopy(model)
            model_copy = model_copy.merge_and_unload()
            torch.save(model_copy, join(task_dir, f'{dataset_name}-finetune.pt'))
            del model_copy  # 
            print("save")
        else:
            torch.save(model, join(task_dir, f'{dataset_name}-finetune.pt'))
            print("save")
             
    if accuracy > best_metric + min_delta:
        best_metric = accuracy
        patience_counter = 0  # 
        torch.save(model, join(task_dir, f'{dataset_name}-best.pt'))
    else:
        patience_counter += 1  # 
    # if (patience_counter >= patience and 'vit' in cfg.model):
    #     print(f"Early stopping at epoch {epoch + 1} due to no improvement.")
    #     break     
    print("\n\n")

print('best acc:',best_metric)
print(f"Training time (excluding testing): {training_time:.2f} seconds")

print(f"Coreset selection time: {coreset_time:.2f} seconds")
if cfg.td == True:
    TD_logger.save_training_dynamics(td_path, data_name=cfg.dataset)
    print('save td!')

print("Done")
