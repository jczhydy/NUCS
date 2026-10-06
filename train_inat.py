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
from utils.train_utils import train_epoch_nat,test_epoch, TrainingDynamicsLogger, IndexDataset
from utils.util import StdRedirect
import pickle
from coreset import CoresetSelection
from torch.utils.data import Subset
from utils.dataset import INAT,INaturalistDataset2021
from utils.model import create_model
from peft import get_peft_model, LoraConfig, TaskType
import copy

parser = argparse.ArgumentParser()

parser.add_argument('--krrmode', type=str, default='all',help='all,class')
parser.add_argument('--criterion', type=str, default='ce')
parser.add_argument('--num_runs', type=int, default=1, help='')
parser.add_argument('--lora', type=int, default=0, help='')
parser.add_argument('--model_finetune_runs', type=int, default=6, help='')
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
parser.add_argument('--coreset_mode', type=str, choices=['random', 'stratified', 'window', 'labelsample','moderate','hard','graph','labelweight','labelwindow','labelwindow_traverse','labelwindow_test','labelwindow_number_traverse'])
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
td_dir = os.path.join(task_dir, 'training-dynamics')
os.makedirs(td_dir, exist_ok=True)
data_score_path = os.path.join(task_dir, f'data-score-{cfg.name}.pickle')

GPUID = cfg.gpuid
os.environ["CUDA_VISIBLE_DEVICES"] = GPUID
device_id = list(map(int, GPUID.split(',')))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_dir = os.path.join(cfg.dataset_dir, cfg.dataset)
print('device:',device)


if cfg.dataset == 'inat2021':
    train_dataset = INaturalistDataset2021.get_inaturalist_train(path=cfg.dataset_dir)
    valid_dataset = INaturalistDataset2021.get_inaturalist_valid(path=cfg.dataset_dir)

if cfg.dataset in ['inat2021']:   
    num_classes=10000

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
    if cfg.coreset_mode == 'window':
        coreset_index = CoresetSelection.class_sample_nat(cfg=cfg,data_score=data_score, ratio=cfg.coreset_ratio, key=cfg.coreset_key, classratiomode='number', classselectmode='window', featuremode='finetune')
        #coreset_index = CoresetSelection.bws_nat(data_score=data_score, ratio=cfg.coreset_ratio, key=cfg.coreset_key, featuremode='feature_nofinetune')
    if cfg.coreset_mode == 'hard':
        coreset_index = CoresetSelection.hard_selection(data_score=data_score, key=cfg.coreset_key, ratio=cfg.coreset_ratio, descending=False)
    if cfg.coreset_mode == 'moderate':
        coreset_index = CoresetSelection.moderate(data_score=data_score, ratio=cfg.coreset_ratio)
    if cfg.coreset_mode == 'labelsample':
        coreset_index = CoresetSelection.label_sample(data_score=data_score, ratio=cfg.coreset_ratio, key=cfg.coreset_key, mode=cfg.label_mode, cfg=cfg, mis_ratio=cfg.mis_ratio)
    if cfg.coreset_mode == 'stratified':
        mis_num = int(cfg.mis_ratio * total_num)
        data_score_mask, score_index = CoresetSelection.mislabel_mask(data_score, mis_key=cfg.coreset_key, mis_num=mis_num, mis_descending=True, coreset_key=cfg.coreset_key)
        coreset_num = int(cfg.coreset_ratio * total_num)
        coreset_index, _ = CoresetSelection.stratified_sampling(data_score=data_score_mask, coreset_key=cfg.coreset_key, coreset_num=coreset_num)
        coreset_index = score_index[coreset_index]
    if cfg.coreset_mode == 'graph':
        coreset_index = CoresetSelection.d2_selection(data_score=data_score, key=cfg.coreset_key, coreset_num=cfg.coreset_ratio * len(train_dataset), cfg=cfg)
    if cfg.coreset_mode == 'labelweight':
        coreset_index = CoresetSelection.class_sample_inat(cfg=cfg, data_score=data_score, ratio=cfg.coreset_ratio, key=cfg.coreset_key,classratiomode='difficulty', classselectmode='ccs')
    if cfg.coreset_mode == 'labelwindow':
        coreset_index = CoresetSelection.class_sample_inat(cfg=cfg, data_score=data_score, ratio=cfg.coreset_ratio, key=cfg.coreset_key,classratiomode='difficulty', classselectmode='window')
    # if cfg.coreset_mode == 'labelwindow_traverse':
    #     coreset_index = CoresetSelection.class_sample_nat_traverse(cfg=cfg, data_score=data_score, ratio=cfg.coreset_ratio, key=cfg.coreset_key,classratiomode='difficulty', classselectmode='window',centerratio=cfg.centerratio)
    # if cfg.coreset_mode == 'labelwindow_number_traverse':
    #     coreset_index = CoresetSelection.class_sample_nat_traverse(cfg=cfg, data_score=data_score, ratio=cfg.coreset_ratio, key=cfg.coreset_key,classratiomode='number', classselectmode='window',centerratio=cfg.centerratio)
    # if cfg.coreset_mode == 'labelwindow_test':
    #     coreset_index = CoresetSelection.class_sample_nat(cfg=cfg, data_score=data_score, ratio=cfg.coreset_ratio, key=cfg.coreset_key,classratiomode='number', classselectmode='window')

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
                                          batch_size=cfg.batch_size,
                                          shuffle=cfg.shuffle,
                                          num_workers=cfg.threads, 
                                          pin_memory=True)
optim = torch.optim.SGD(model.parameters(), lr=cfg.lr, momentum=0.9, weight_decay=5e-4, nesterov=True)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=cfg.num_epochs, eta_min=1e-5)

print(optim)

start_epoch = 0
metrics = []


patience = cfg.patience  
min_delta = 0.00001  
best_metric = -np.inf  
patience_counter = 0  


criterion = torch.nn.CrossEntropyLoss()
print(criterion)
training_time = 0.0  

if cfg.criterion == 'CB':
    num = np.zeros(num_classes, dtype=int)
    for _, (_,(_, targets)) in enumerate(train_loader):
        if isinstance(targets, list):
            targets = torch.tensor(targets)  
        targets = targets.cpu().numpy()  
        for cls in range(num_classes):
            num[cls] += np.sum(targets == cls)
    print(num)
    
for epoch in range(start_epoch, cfg.num_epochs):
    print(f"Epoch {epoch + 1}:")
    print('Training:')
    
    epoch_train_start = time.time()
    if cfg.td:
        TD_logger = TrainingDynamicsLogger()
    else:
        TD_logger = None
    current_TD_logger = TD_logger if epoch < cfg.finetune_runs else None
    
    avg_loss = train_epoch_nat(
        cfg=cfg,
        epoch=epoch,
        model=model,
        device=device,
        optimizer=optim,
        train_loader=train_loader,
        criterion=criterion,
        TD_logger=current_TD_logger  
    )
    
    
    epoch_train_end = time.time()
    training_time += epoch_train_end - epoch_train_start
    print('Training time:',epoch_train_end - epoch_train_start)
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
    if current_TD_logger!=None:
        td_path = os.path.join(td_dir, f'td-{cfg.name}-epoch-{epoch}.pickle')
        print(f'Saving training dynamics at {td_path}')
        TD_logger.save_training_dynamics(td_path, data_name=cfg.dataset)
        
    if (epoch + 1 == cfg.finetune_runs and cfg.coreset == 0):
        if cfg.lora == 1:
            model_copy = copy.deepcopy(model)
            model_copy = model_copy.merge_and_unload()
            torch.save(model_copy, join(task_dir, f'{dataset_name}-finetune.pt'))
            del model_copy  
            print("save")
        else:
            torch.save(model, join(task_dir, f'{dataset_name}-finetune.pt'))
            print("save")
             
    if accuracy > best_metric + min_delta:
        best_metric = accuracy
        patience_counter = 0 
        torch.save(model, join(task_dir, f'{dataset_name}-best.pt'))
    else:
        patience_counter += 1 
    # if (patience_counter >= patience and 'vit' in cfg.model):
    #     print(f"Early stopping at epoch {epoch + 1} due to no improvement.")
    #     break     
    print("\n\n")

print('best acc:',best_metric)
# if "vit" in cfg.model:
#     model = model.merge_and_unload()  

# print("Testing merged model:")
# mean_loss, accuracy = test_epoch(
#     cfg=cfg,
#     epoch=epoch,
#     model=model,
#     device=device,
#     test_loader=valid_loader,
#     criterion=criterion
# )
print(f"Training time (excluding testing): {training_time:.2f} seconds")

print(f"Coreset selection time: {coreset_time:.2f} seconds")

# if cfg.td == True:
#     TD_logger.save_training_dynamics(td_path, data_name=cfg.dataset)
#     print('save td!')

print("Done")
