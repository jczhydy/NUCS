import os
import pickle
import pprint
import random
from glob import glob
from os.path import exists, join
import sys
import argparse
import numpy as np
import torch
import sklearn.metrics
from sklearn.metrics import roc_auc_score, accuracy_score
import sklearn, sklearn.model_selection
import torchxrayvision as xrv
import torchvision
from tqdm import tqdm as tqdm_base
from .utils.train_utils import IndexDataset,DataAugmentation
from .utils.dataset import SubsetDataset,NIH_Dataset, NIH_Dataset_2, CheX_Dataset, PC_Dataset
import torch.nn as nn
from torchvision import models
from efficientnet_pytorch import EfficientNet
from torchvision.models.feature_extraction import create_feature_extractor
    
parser = argparse.ArgumentParser()

parser.add_argument('--finetune_runs', type=int, default=3, help='')
parser.add_argument('--enhancement', type=int, default=1, help='')
parser.add_argument('--dataset_dir', type=str, default="")
parser.add_argument('--td-path', type=str, default='../data/',
                    help='The dir path of the data.')
parser.add_argument('--output_dir', type=str, default="")
parser.add_argument('--name', type=str,default="")
parser.add_argument('--taskweights', type=int, default=0, help='')
parser.add_argument('--dataset', type=str, default="nih")
parser.add_argument('--seed', type=int, default=0, help='')
parser.add_argument('--batch_size', type=int, default=32, help='')
parser.add_argument('--cuda', type=int, default=1, help='')
parser.add_argument('--load_target', type=bool, default=False, help='')
parser.add_argument('--unique_patients', type=int, default=0, help='')
parser.add_argument('--data_aug_rot', type=int, default=45, help='')
parser.add_argument('--data_aug_trans', type=float, default=0.15, help='')
parser.add_argument('--data_aug_scale', type=float, default=0.15, help='')
parser.add_argument('--model', type=str, default="resnet34")

cfg = parser.parse_args()

np.random.seed(cfg.seed)
random.seed(cfg.seed)
torch.manual_seed(cfg.seed)
if cfg.cuda:
    torch.cuda.manual_seed_all(cfg.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

task_dir = os.path.join(cfg.output_dir, cfg.name)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# td_path = os.path.join(task_dir, f'td-{cfg.name}.pickle')
data_score_path = os.path.join(task_dir, f'data-score-{cfg.name}.pickle')

# with open(td_path, 'rb') as f:
#      pickled_data = pickle.load(f)
     
# training_dynamics = pickled_data['training_dynamics']
transforms = torchvision.transforms.Compose([
    DataAugmentation(),
    # xrv.datasets.XRayCenterCrop(),
    torchvision.transforms.ToPILImage(),  
    torchvision.transforms.Resize((224, 224))  
])

if cfg.unique_patients==0:
    data_aug = torchvision.transforms.Compose([
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),  # 
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 
    ])
else:
    data_aug = None
datas = []
datas_names = []
if "nih" in cfg.dataset:
    # dataset = xrv.datasets.NIH_Dataset(
    #     imgpath=cfg.dataset_dir + "/NIH/images-224", 
    #     transform=transforms, data_aug=data_aug, unique_patients=cfg.unique_patients,views=["PA","AP"])
    dataset = NIH_Dataset_2(
    imgpath=cfg.dataset_dir + "/NIH/images-224", csvpath = cfg.dataset_dir + "/NIH/Data_Entry_2017_new2.csv",
    transform=transforms, data_aug=data_aug, unique_patients=cfg.unique_patients, views=["PA","AP"])
    #datas.append(dataset)
    datas_names.append("nih")
elif "pc" in cfg.dataset:
    dataset = PC_Dataset(
        imgpath=cfg.dataset_dir + "/PC/images-224", csvpath = cfg.dataset_dir + "/PC/PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv",
        transform=transforms, data_aug=data_aug, unique_patients=cfg.unique_patients, views=["PA","AP"])
   # datas.append(dataset)
    datas_names.append("pc")
else:
    print("dataset error")

# xrv.datasets.relabel_dataset(xrv.datasets.default_pathologies, dataset)
gss = sklearn.model_selection.GroupShuffleSplit(train_size=0.8,test_size=0.2, random_state=cfg.seed)
train_inds, test_inds = next(gss.split(X=range(len(dataset)), groups=dataset.csv.patientid))

if "nih" in cfg.dataset:
    data_aug_train = torchvision.transforms.Compose([
    #  torchvision.transforms.ToPILImage(),
        torchvision.transforms.RandomResizedCrop(224),
        torchvision.transforms.ToTensor(),  # 
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 
    ])
else:
    data_aug_train = torchvision.transforms.Compose([
    #  torchvision.transforms.ToPILImage(),
        torchvision.transforms.RandomResizedCrop(224,scale=(0.8,1.2)),
        torchvision.transforms.ToTensor(),  # 
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 
    ])

data_aug_test = torchvision.transforms.Compose([
   # torchvision.transforms.ToPILImage(),
    torchvision.transforms.ToTensor(),  # 
    torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 
])

if (cfg.unique_patients == 1):
    train_dataset = SubsetDataset(dataset, train_inds,data_aug_train)
    valid_dataset = SubsetDataset(dataset, test_inds,data_aug_test)
    print(data_aug_train)
else:
    train_dataset = xrv.datasets.SubsetDataset(dataset, train_inds)
    valid_dataset = xrv.datasets.SubsetDataset(dataset, test_inds)

train_dataset = IndexDataset(train_dataset)  ####
train_loader = torch.utils.data.DataLoader(train_dataset,
                                            batch_size=cfg.batch_size,
                                            shuffle=False,
                                            num_workers=4, 
                                            pin_memory=True)
data_importance = {}

if cfg.taskweights:
    weights = np.nansum(train_loader.dataset.labels, axis=0)
    weights = weights.max() - weights + weights.mean()
    weights = weights / weights.max()
    weights = torch.from_numpy(weights).to(device).float()
    print("task weights:", weights)
else:
    weights={}
def weighted_l2_loss(label, output, weights):

    loss = (label - output) ** 2  
    weighted_loss = loss * weights**2  
    return weighted_loss.sum(dim=1)  # 

def EL2N(td_log, dataset, data_importance, weights, max_epoch, cfg):
    data_size = len(dataset)
    targets = torch.tensor([dataset[i][1]["lab"] for i in range(data_size)], device=device)  
    data_importance['targets'] = targets.type(torch.int32).to(device)  
    data_importance['el2n'] = torch.zeros(data_size, device=device).type(torch.float32)  
    l2_loss = torch.nn.MSELoss(reduction='none')

    def record_training_dynamics(td_log,weights,cfg):
        output = torch.sigmoid(td_log['output'].type(torch.float)).to(device)  
        index = td_log['idx'].type(torch.long).to(device)  
        label = targets[index]
        # mask = ~torch.isnan(label)
        # output = output[mask].reshape(-1, mask.sum(dim=1).max())
        # label = label[mask].reshape(-1, mask.sum(dim=1).max())    

        label_num = label.shape[1]
        # weights = weights[:label_num]
        if cfg.taskweights:
            weights = weights.to(device)  # 

       # el2n_score = torch.sqrt(weighted_l2_loss(label, output, weights))       
        el2n_score = torch.sqrt(l2_loss(label, output).sum(dim=1))
        data_importance['el2n'][index] += el2n_score

    print("recording")
    for i, item in enumerate(td_log):
        if i % 200 == 0:
            print(i)
        if item['epoch'] == max_epoch:
            return
        record_training_dynamics(item,weights,cfg)


def calculate_data_importance(cfg, model, device, train_loader,modelname):
    data_size = len(train_loader.dataset) 
    data_importance['effort'] = torch.zeros(data_size, device=device).type(torch.float32)
    feature_dim = model.fc.in_features
    data_importance['feature'] = torch.zeros((data_size, feature_dim), device=device).type(torch.float32)
    model.eval()  
    criterion = torch.nn.BCEWithLogitsLoss()

    for batch_idx, (idx, samples) in enumerate(train_loader):
        if batch_idx % 50 == 0:
            print(f"Processing batch {batch_idx}/{len(train_loader)}")
        
        images = samples["img"].to(device)
        targets = samples["lab"].to(device)
        features = []    
        def hook(module, input, output):
            features.append(output.clone().detach())
        
        handle = model.avgpool.register_forward_hook(hook)  # 
        outputs = model(images)  
        if features:
            batch_features = features[-1].view(features[-1].size(0), -1)  # 
            data_importance['feature'][idx] = batch_features  # 

        loss = torch.zeros(1, device=device).float()
        for i in range(images.size(0)):
            image = images[i:i+1].clone().detach().requires_grad_(True)
            target = targets[i:i+1].clone().detach()
            outputs = model(image)  # 
            single_loss = torch.zeros(1, device=device).float()
            
            for task in range(target.shape[1]):  # 
                task_output = outputs[:, task]
                task_target = target[:, task]
                mask = ~torch.isnan(task_target)  # 
                task_output = task_output[mask]
                task_target = task_target[mask]
                if len(task_target) > 0:
                    task_loss = criterion(task_output.float(), task_target.float())
                    if cfg.taskweights:
                        single_loss += weights[task] * task_loss
                    else:
                        single_loss += task_loss
            
            single_loss = single_loss.sum()
            single_loss.backward()  

            grad_norm = torch.norm(
                torch.stack(
                    [torch.norm(param.grad) for param in model.parameters() if param.grad is not None]
                )
            )
            data_importance['effort'][idx[i]] = grad_norm
            model.zero_grad()     
        handle.remove()  # 
    
    return data_importance


def get_feature(model, device, train_loader, modelname):
    data_size = len(train_loader.dataset)
    feature_dim = model.fc.in_features
    data_importance['feature_nofinetune'] = torch.zeros((data_size, feature_dim), device=device).type(torch.float32)
    model.eval()
    features_collected = []
    def hook(module, input, output):
        features_collected.append(output)  
    handle = model.avgpool.register_forward_hook(hook)  
    try:
        for batch_idx, (idx, samples) in enumerate(train_loader):
            if batch_idx % 50 == 0:
                print(f"Processing batch {batch_idx}/{len(train_loader)}")        
            images = samples["img"].to(device, non_blocking=True)
            targets = samples["lab"].to(device, non_blocking=True)       
            with torch.no_grad():
                outputs = model(images)      
            batch_features = features_collected[-1].view(features_collected[-1].size(0), -1)
            data_importance['feature_nofinetune'][idx] = batch_features         
            features_collected.clear()
    finally:
        # 
        handle.remove()  
    return data_importance

def extract_and_save_features(cfg, device, train_loader, model_type='resnet34'):
    data_size = len(train_loader.dataset)
    
    if model_type == 'resnet34':
        model = models.resnet34(pretrained=True)
        feature_dim = 512  # 
    elif model_type == 'densenet121':
        model = models.densenet121(pretrained=True)
        feature_dim = 1024  # 

    model.eval()
    model.to(device)
    if model_type == 'resnet34':
        return_nodes = {'avgpool': 'features'}  # 
    elif model_type == 'densenet121':
        return_nodes = {'features': 'features'}  # 
    
    feature_extractor = create_feature_extractor(model, return_nodes=return_nodes)
    
    # 
    data_importance = {'feature_pretrain': torch.zeros((data_size, feature_dim), device=device, dtype=torch.float32)}

    # 
    idx_list = []
    feature_list = []
    
    for batch_idx, (idx, samples) in enumerate(train_loader):
        if batch_idx % 50 == 0:
            print(f"Processing batch {batch_idx}/{len(train_loader)}")
        
        images = samples["img"].to(device)

        with torch.no_grad():
            features = feature_extractor(images)['features']  # 
  
        features = features.view(features.size(0), -1)  # 
        data_importance['feature_pretrain'][idx] = features
    
    return data_importance



dataset_name = cfg.dataset + "-" + cfg.model + "-" + cfg.name
finetune_path = join(task_dir, f'{dataset_name}-finetune.pt')

temp=dataset.pathologies
pathologies=temp
# if cfg.dataset=='nih':
#     pathologies=temp+["No Finding"]
print(pathologies)

if "resnet34" in cfg.model:
    print("load weight")
    model = torch.load(finetune_path)
    model = model.to(device)
    
    model2 = torchvision.models.resnet34(pretrained=True)  
    num_ftrs = model2.fc.in_features
    model2.fc = nn.Linear(num_ftrs, dataset.labels.shape[1])
    model2 = model2.to(device)
    
elif "resnet18" in cfg.model:
    print("load weight")
    model = torch.load(finetune_path)
    model = model.to(device)

print("load data")
#EL2N(training_dynamics, train_dataset, data_importance, weights=weights, max_epoch=cfg.finetune_runs, cfg=cfg)
calculate_data_importance(cfg=cfg,model=model,device=device,train_loader=train_loader,modelname='resnet')
get_feature(model=model2,device=device,train_loader=train_loader,modelname='resnet')
print(f'Saving data score at {data_score_path}')

with open(data_score_path, 'wb') as handle:
    pickle.dump(data_importance, handle)
print("save!")