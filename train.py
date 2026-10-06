import os,sys
sys.path.insert(0,".")
import inspect
from glob import glob
from os.path import exists, join
import matplotlib.pyplot as plt
import numpy as np
import argparse
import torch
import torchvision, torchvision.transforms
import sklearn, sklearn.model_selection
import torch.nn as nn
import random
from utils.train_utils import train_epoch, valid_test_epoch, TrainingDynamicsLogger, DataAugmentation, IndexDataset
from utils.util import StdRedirect
import torchxrayvision as xrv
import pickle
from coreset import CoresetSelection
from utils.dataset import SubsetDataset,NIH_Dataset,NIH_Dataset_2
from efficientnet_pytorch import EfficientNet

parser = argparse.ArgumentParser()

parser.add_argument('--num_runs', type=int, default=1, help='')
parser.add_argument('--earlystop', type=int, default=1, help='')
parser.add_argument('--centerratio', type=float, default=1, help='')
parser.add_argument('--enhancement', type=int, default=0, help='')
parser.add_argument('--noratio', type=float, default=1, help='')
parser.add_argument('--endratio', type=float, default=1, help='')
parser.add_argument('--finetune_only_head', type=int, default=0, help='')
parser.add_argument('--pretrainpath', type=str, default='', help='')
parser.add_argument('--imgpretrain', type=int, default=1, help='')
parser.add_argument('--sigma', type=float, default=2, help='')
parser.add_argument('--labelsigma', type=float, default=1, help='')
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
parser.add_argument('--class_balance', type=int, default=0, help='')
parser.add_argument('--coreset_mode', type=str, choices=['random', 'stratified', 'window', 'cluster_balance','gauss','hard','graph','labelweight','labelwindow'])
parser.add_argument('--coreset_key', type=str,help='forgetting,accumulated_margin,el2n')
parser.add_argument('--coreset_ratio', type=float, default=1, help='')
parser.add_argument('--coreset', type=int, default=0, help='')

parser.add_argument('--unique_patients', type=int, default=1, help='')
parser.add_argument('--td', type=int, default=0, help='')
parser.add_argument('--gpuid', type=str, default='1',help='The ID of GPU.')
parser.add_argument('-f', type=str, default="", help='')
parser.add_argument('--name', type=str,default="")
parser.add_argument('--output_dir', type=str, default="")
parser.add_argument('--dataset', type=str, default="nih")
parser.add_argument('--dataset_dir', type=str, default="")
parser.add_argument('--model', type=str, default="resnet34")
parser.add_argument('--seed', type=int, default=0, help='')
parser.add_argument('--cuda', type=bool, default=True, help='')
parser.add_argument('--num_epochs', type=int, default=30, help='')
parser.add_argument('--batch_size', type=int, default=64, help='')
parser.add_argument('--shuffle', type=bool, default=True, help='')
parser.add_argument('--lr', type=float, default=0.0001, help='')
parser.add_argument('--threads', type=int, default=6, help='')
parser.add_argument('--taskweights', type=int, default=0, help='')

parser.add_argument('--data_aug_rot', type=int, default=45, help='')
parser.add_argument('--data_aug_trans', type=float, default=0.15, help='')
parser.add_argument('--data_aug_scale', type=float, default=0.15, help='')

cfg = parser.parse_args()
print("taskweights:",cfg.taskweights,"model:",cfg.model,"coreset_mode:",cfg.coreset_mode,"coreset_ratio:",cfg.coreset_ratio,"coreset_key:",cfg.coreset_key)

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
print('device:',device)


if cfg.unique_patients==0:
    data_aug = torchvision.transforms.Compose([
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),  # 
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 
    ])
else:
    data_aug = None


print(data_aug)

transforms = torchvision.transforms.Compose([
    DataAugmentation(),
    torchvision.transforms.ToPILImage(),  
    torchvision.transforms.Resize((224, 224))  
])

datas = []
datas_names = []
if "nih" in cfg.dataset:
    dataset = NIH_Dataset_2(
    imgpath=cfg.dataset_dir + "/NIH/images-224", csvpath = cfg.dataset_dir +"/NIH/Data_Entry_2017_new2.csv",
    transform=transforms, data_aug=data_aug, unique_patients=cfg.unique_patients, views=["PA","AP"])
   # datas.append(dataset)
    datas_names.append("nih")

print("datas_names", datas_names)
temp=dataset.pathologies
pathologies=temp
# if cfg.dataset=='nih':
#     pathologies=temp+["No Finding"]
print(pathologies)
# xrv.datasets.relabel_dataset(xrv.datasets.default_pathologies, dataset)

train_datas = []
test_datas = []

if "resnet34" in cfg.model:
    if cfg.imgpretrain==1:
        model = torchvision.models.resnet34(pretrained=True)  
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, dataset.labels.shape[1])
        print("predict class number:",dataset.labels.shape[1])
        print("imagenet weight")
        model = model.to(device)
    else:
        model = torch.load(cfg.pretrainpath)  
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, dataset.labels.shape[1])
        print("predict class number:",dataset.labels.shape[1])
        print("chex pretrained weight")
        model = model.to(device)
elif "resnet18" in cfg.model:
    model = torchvision.models.resnet18(pretrained=True)  
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, dataset.labels.shape[1])
    print("predict class number:",dataset.labels.shape[1])
    model = model.to(device)
elif "efficientnet-b0" in cfg.model:
    model = EfficientNet.from_pretrained('efficientnet-b0')
    num_ftrs = model._fc.in_features  # 
    model._fc = nn.Linear(num_ftrs, dataset.labels.shape[1])  #
    print("predict class number:",dataset.labels.shape[1]) 
    model = model.to(device)

#######    training
dataset_name = cfg.dataset + "-" + cfg.model + "-" + cfg.name
# Dataset
# split    
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
total_num = len(train_dataset)

### coreset
if cfg.coreset:
    if cfg.coreset_mode != 'random':
        with open(cfg.data_score_path, 'rb') as f:
            data_score = pickle.load(f)    
    if cfg.coreset_mode == 'random':
        coreset_index = CoresetSelection.random_selection(total_num=len(train_dataset), num=cfg.coreset_ratio * len(train_dataset))
    if cfg.coreset_mode == 'window':
        coreset_index = CoresetSelection.class_sample(cfg=cfg,data_score=data_score, ratio=cfg.coreset_ratio, balance_key=cfg.coreset_key, key=cfg.coreset_key,classratiomode='number', classselectmode='window')
    if cfg.coreset_mode == 'hard':
        coreset_index = CoresetSelection.hard_selection(data_score=data_score, key=cfg.coreset_key, ratio=cfg.coreset_ratio, descending=False)
    if cfg.coreset_mode == 'stratified':
        mis_num = int(cfg.mis_ratio * total_num)
        data_score_mask, score_index = CoresetSelection.mislabel_mask(data_score, mis_key=cfg.coreset_key, mis_num=mis_num, mis_descending=True, coreset_key=cfg.coreset_key)
        coreset_num = int(cfg.coreset_ratio * total_num)
        coreset_index, _ = CoresetSelection.stratified_sampling(data_score=data_score_mask, coreset_key=cfg.coreset_key, coreset_num=coreset_num)
        coreset_index = score_index[coreset_index]
    if cfg.coreset_mode == 'graph':
        coreset_index = CoresetSelection.d2_selection(data_score=data_score, key=cfg.coreset_key, coreset_num=cfg.coreset_ratio * len(train_dataset), cfg=cfg)
    if cfg.coreset_mode == 'labelweight':
        coreset_index = CoresetSelection.class_sample(cfg=cfg,data_score=data_score, ratio=cfg.coreset_ratio, balance_key=cfg.coreset_key, key=cfg.coreset_key)
       # coreset_index = CoresetSelection.class_sample_traverse(data_score=data_score, ratio=cfg.coreset_ratio, balance_key=cfg.coreset_key, key=cfg.coreset_key, centerratio=cfg.centerratio)
    if cfg.coreset_mode == 'labelwindow':
        print("balance_mode",'window')
        coreset_index = CoresetSelection.class_sample(cfg=cfg,data_score=data_score, ratio=cfg.coreset_ratio, balance_key=cfg.coreset_key, key=cfg.coreset_key,classratiomode='difficulty', classselectmode='window') 
        
    train_dataset = xrv.datasets.SubsetDataset(train_dataset, coreset_index)

print("traindata size:",len(train_dataset))
train_dataset = IndexDataset(train_dataset)
print("testdata size:",len(valid_dataset))
# load
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

#optim = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=1e-5, amsgrad=True)
optim = torch.optim.SGD(model.parameters(), lr=cfg.lr, momentum=0.9, weight_decay=5e-4, nesterov=True)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=cfg.num_epochs)
print(optim)
criterion = torch.nn.BCEWithLogitsLoss()
start_epoch = 0
metrics = []
##td
td_log = []
if cfg.td==True:
   TD_logger = TrainingDynamicsLogger()
else:
   TD_logger = None

#########
best_label_aucs = None
patience = 5  # 
min_delta = 0.0001  # 
best_metric = -np.inf  # 
patience_counter = 0  # 

if cfg.coreset==1:
    num_runs = cfg.num_runs
else:
    num_runs = 1
all_avg_auc_valid_list = []
all_avg_task_auc_list = []

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
base_seed = cfg.seed
for run in range(num_runs):
    print(f"Run {run + 1}/{num_runs}:")
    set_seed(base_seed)  
    if "resnet34" in cfg.model:
        if cfg.imgpretrain==1:
            model = torchvision.models.resnet34(pretrained=True)
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, dataset.labels.shape[1])
            print("predict class number:", dataset.labels.shape[1])
            print("imagenet weight")
            model = model.to(device)
        else:
            model = torch.load(cfg.pretrainpath)  
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, dataset.labels.shape[1])
            print("predict class number:", dataset.labels.shape[1])
            print("chex pretrained weight")
            model = model.to(device)

    seed = base_seed + run  
    set_seed(seed)
    
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

    #optim = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=1e-5, amsgrad=True)
    optim = torch.optim.SGD(model.parameters(), lr=cfg.lr, momentum=0.9, weight_decay=5e-4, nesterov=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=cfg.num_epochs)
    
    run_best_metric = -np.inf
    run_best_label_aucs = None
    run_patience_counter = 0  # 
    
    for epoch in range(start_epoch, cfg.num_epochs):
        print(f"Epoch {epoch + 1}:")
        print('Training:')
        avg_loss = train_epoch(cfg=cfg,
                               epoch=epoch,
                               model=model,
                               device=device,
                               optimizer=optim,
                               train_loader=train_loader,
                               criterion=criterion,
                               TD_logger=TD_logger)
        print('Testing:')
        scheduler.step()
        auc_valid, task_auc, _, _ = valid_test_epoch(
            name='Valid',
            pathologies=pathologies,
            epoch=epoch,
            model=model,
            device=device,
            data_loader=valid_loader,
            criterion=criterion
        )
        if (epoch + 1 == 3 and cfg.coreset == 0):
            torch.save(model, join(task_dir, f'{dataset_name}-finetune.pt'))
            print("save finetune model!")

            
        avg_auc_valid = np.mean(auc_valid)
        avg_task_auc = task_auc
        
        if avg_auc_valid > run_best_metric + min_delta:
            run_best_metric = avg_auc_valid
            run_best_label_aucs = avg_task_auc
            if cfg.coreset==1:
                torch.save(model, os.path.join(task_dir, f'{dataset_name}-best-run{run+1}.pt'))
            else:
                torch.save(model, join(task_dir, f'{dataset_name}-best.pt'))
            run_patience_counter = 0  # 
        else:
            run_patience_counter += 1  # 
        
        if (run_patience_counter >= patience):
            print(f"Early stopping at epoch {epoch + 1} in run {run + 1} due to no improvement in AUC.")
            break
        
        print("\n")
        print(f"Best average AUC for run {run + 1}: {run_best_metric:.4f}")
        for idx, auc_value in enumerate(run_best_label_aucs):
            pathology = pathologies[idx] if idx < len(pathologies) else f'Label {idx + 1}'
            if not np.isnan(auc_value):
                print(f'{pathology}: AUC: {auc_value:.4f}')
            else:
                print(f'{pathology}: AUC: N/A (Not enough positive/negative samples)')
        
        print("\n\n")

    all_avg_auc_valid_list.append(run_best_metric)
    all_avg_task_auc_list.append(run_best_label_aucs)

final_avg_auc_valid = np.mean(all_avg_auc_valid_list)
final_avg_task_auc = np.mean(all_avg_task_auc_list, axis=0)  
print("\nFinal average metrics:")
print(f"Average AUC (Validation): {final_avg_auc_valid:.4f}")
for idx, auc_value in enumerate(final_avg_task_auc):
    pathology = pathologies[idx] if idx < len(pathologies) else f'Label {idx + 1}'
    if not np.isnan(auc_value):
        print(f'{pathology}: AUC: {auc_value:.4f}')
    else:
        print(f'{pathology}: AUC: N/A (Not enough positive/negative samples)')
###########       
if cfg.td==True:
   TD_logger.save_training_dynamics(td_path, data_name=cfg.dataset)
   print('save td!')

print("Done")
