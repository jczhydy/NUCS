import os
import pickle
from glob import glob
from os.path import exists, join
import sys
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, accuracy_score
import sklearn, sklearn.model_selection
import cv2
from tqdm import tqdm as tqdm_base
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.metrics import accuracy_score
import torch.nn.functional as F

class DataAugmentation:
    def __call__(self, img):

        if len(img.shape) == 2:  # 
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)  
        elif img.shape[0] == 1:  
            img = img.squeeze(0)  # 
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)  # 
        elif img.shape[2] == 3:  # 
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 
        else:
            raise ValueError("Unexpected image shape: {}".format(img.shape))
        if img.shape[0] == 3:  
            img = img.transpose(1, 2, 0)  
     
        if img.dtype == np.float32 or img.dtype == np.float64:
            img = (img + 1024) / 2048.0
            img = (img * 255).clip(0, 255).astype(np.uint8)
        elif img.dtype == np.uint8:
            print("Image is already uint8, shape:", img.shape)
        else:
            raise ValueError("Unsupported image dtype: {}".format(img.dtype))

        # img = torchvision.transforms.ToPILImage()(img)  # 
        # print("Converted to PIL image, size:", img.size)  # 
        # plt.imshow(img)
        # plt.show()
        return img
    
def tqdm(*args, **kwargs):
    if hasattr(tqdm_base, '_instances'):
        for instance in list(tqdm_base._instances):
            tqdm_base._decr_instances(instance)
    return tqdm_base(*args, **kwargs)

class IndexDataset(torch.utils.data.Dataset):
    """
    The dataset also return index.
    """
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, idx):
        return idx, self.dataset[idx]

    def __len__(self):
        return len(self.dataset)
        
    @property
    def labels(self):
        return self.dataset.labels
    
class StdRedirect:
    def __init__(self, filename):
        self.stream = sys.stdout
        self.file = open(filename,'w')

    def write(self, data):
        self.stream.write(data)
        self.stream.flush()
        self.file.write(data)
        self.file.flush()

    def flush(self):
        pass

    def __del__(self):
        self.file.close()

class TrainingDynamicsLogger(object):
    """
    Helper class for saving training dynamics for each iteration.
    Maintain a list containing output probability for each sample.
    """
    def __init__(self, filename=None):
        self.training_dynamics = []

    def log_tuple(self, tuple):
        self.training_dynamics.append(tuple)

    def save_training_dynamics(self, filepath, data_name=None):
        pickled_data = {
            'data-name': data_name,
            'training_dynamics': self.training_dynamics
        }

        with open(filepath, 'wb') as handle:
            pickle.dump(pickled_data, handle)


def train_epoch(cfg, epoch, model, device, train_loader, optimizer, criterion, TD_logger=None, limit=None):
    model.train()
    avg_loss = []  
    if cfg.taskweights:
        weights = np.nansum(train_loader.dataset.labels, axis=0)
        weights = weights.max() - weights + weights.mean()
        weights = weights / weights.max()
        weights = torch.from_numpy(weights).to(device).float()
        print("task weights:", weights)
    else:
        weights = torch.ones(train_loader.dataset.labels.shape[1]).to(device).float()
        
    for batch_idx, (idx,samples) in enumerate(train_loader):
        
        if limit and (batch_idx > limit):
            print("Breaking out of training loop.")
            break
        optimizer.zero_grad()  
        images = samples["img"].float().to(device)

        targets = samples["lab"].to(device)
        outputs = model(images)

        loss = torch.zeros(1).to(device).float()
        
        for task in range(targets.shape[1]):
            task_output = outputs[:, task]
            task_target = targets[:, task]
            # mask = ~torch.isnan(task_target)
            # task_output = task_output[mask]
            # task_target = task_target[mask]
            if len(task_target) > 0:
                task_loss = criterion(task_output.float(), task_target.float())
                if cfg.taskweights:
                   loss += weights[task] * task_loss
                else:
                   loss += task_loss
        # loss = loss.sum()
        loss.backward()
        avg_loss.append(loss.detach().cpu().numpy())
        optimizer.step()
        if TD_logger:
            log_tuple = {
                'epoch': epoch,
                'idx': idx.type(torch.long).clone(),
                'output': outputs.detach()
            }
            TD_logger.log_tuple(log_tuple)
    mean_loss = np.mean(avg_loss)
    print(f'Epoch {epoch + 1} - Train - Avg Loss = {mean_loss:4.4f}')

    return np.mean(avg_loss)


def train_epoch_infobatch(cfg, epoch, model, device, train_data, train_loader, optimizer, criterion, TD_logger=None, limit=None):
    model.train()  # 
    avg_loss = []  # 
    
    for batch_idx, (idx, (inputs, targets)) in enumerate(train_loader):
        
        if limit and batch_idx > limit:  # 
            print("Breaking out of training loop.")
            break
 # 
        inputs = inputs.float().to(device)  # 
        targets = targets.to(device)  # 
        
        outputs = model(inputs)  # 
        if 'vit' in cfg.model:
            outputs = outputs.logits

        loss = criterion(outputs, targets)
        loss = train_data.update(loss)

        optimizer.zero_grad() 
        loss.backward()  # 
        avg_loss.append(loss.detach().cpu().numpy())  # 
        optimizer.step()  # 

    mean_loss = np.mean(avg_loss)  # 
    print(f'Epoch {epoch + 1} - Train - Avg Loss = {mean_loss:4.4f}')
    
    return mean_loss

def train_epoch_nat(cfg, epoch, model, device, train_loader, optimizer, criterion, TD_logger=None, limit=None):
    model.train()  # 
    avg_loss = []  # 
    
    for batch_idx, (idx, (inputs, targets)) in enumerate(train_loader):
        
        if limit and batch_idx > limit:  # 
            print("Breaking out of training loop.")
            break

        optimizer.zero_grad()  # 
        inputs = inputs.float().to(device)  # 
        targets = targets.to(device)  # 
        
        outputs = model(inputs)  # 
        if 'vit' in cfg.model:
            outputs = outputs.logits

        loss = criterion(outputs, targets)
        
        loss.backward()  # 
        avg_loss.append(loss.detach().cpu().numpy())  # 
        optimizer.step()  # 
        
        if TD_logger:
            log_tuple = {
                'epoch': epoch,
                'idx': idx.type(torch.long).clone(),
                'output': F.log_softmax(outputs, dim=1).detach().cpu().type(torch.half)
            }
            TD_logger.log_tuple(log_tuple)

    mean_loss = np.mean(avg_loss)  # 
    print(f'Epoch {epoch + 1} - Train - Avg Loss = {mean_loss:4.4f}')
    
    return mean_loss




def valid_test_epoch(name, pathologies, epoch, model, device, data_loader, criterion, limit=None):
    model.eval()
    task_outputs = {task: [] for task in range(len(pathologies))}
    task_targets = {task: [] for task in range(len(pathologies))}

    with torch.no_grad():
        for batch_idx, samples in enumerate(data_loader):
            if limit and batch_idx > limit:
                print("breaking out")
                break

            images = samples["img"].to(device)
            targets = samples["lab"].to(device)
            outputs = model(images)
            #####
            outputs = torch.sigmoid(outputs)
            #####
            # Process each task output
            for task in range(targets.shape[1]):
                task_output = outputs[:, task]
                task_target = targets[:, task]
                
                # Append the task outputs and targets directly as torch tensors
                task_outputs[task].append(task_output.detach().cpu())
                task_targets[task].append(task_target.detach().cpu())

        # Convert list of tensors to single tensors
        for task in range(len(task_targets)):
            task_outputs[task] = torch.cat(task_outputs[task], dim=0).numpy()
            task_targets[task] = torch.cat(task_targets[task], dim=0).numpy()

        # Calculate AUC and F1 for each pathology
        task_aucs = []
        task_f1s = []
        for task, pathology in enumerate(pathologies):
            if len(np.unique(task_targets[task])) > 1:
                # AUC calculation
                task_auc = roc_auc_score(task_targets[task], task_outputs[task])
                task_aucs.append(task_auc)
                print(f"Task {task} ({pathology}) - AUC = {task_auc:4.4f}")
            else:
                print(f"Task {task} ({pathology}) - AUC cannot be calculated (only one class present)")
                task_aucs.append(np.nan)
                task_f1s.append(np.nan)

    # Calculate the mean AUC
    task_aucs = np.asarray(task_aucs)
    auc = np.mean(task_aucs[~np.isnan(task_aucs)])
    print(f'Epoch {epoch + 1} - {name} - Avg AUC = {auc:4.4f}')

    # Optionally calculate F1 score here
    task_f1s = np.asarray(task_f1s)
    f1 = np.mean(task_f1s[~np.isnan(task_f1s)])

    return auc, task_aucs, f1, task_f1s

def clear_label(data_score):
    def clean_vector(vector):
        return torch.where((vector == 0) | (vector == 1), vector, torch.tensor(0, dtype=vector.dtype))
    
    cleaned_targets = torch.stack([clean_vector(row) for row in data_score])
    return cleaned_targets


def test_epoch(cfg, epoch, model, device, test_loader, criterion):
    model.eval()  #
    avg_loss = []  # 
    all_preds = []  # 
    all_labels = []  # 

    with torch.no_grad():  
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs = inputs.float().to(device) 
            targets = targets.to(device)  
            
            outputs = model(inputs)  # 
            if 'vit' in cfg.model:
                outputs = outputs.logits
           
            loss = criterion(outputs, targets)
            avg_loss.append(loss.detach().cpu().numpy())  # 
            
            _, predicted = torch.max(outputs, 1)  # 
            all_preds.append(predicted.cpu().numpy())  # 
            all_labels.append(targets.cpu().numpy())  # 

    mean_loss = np.mean(avg_loss)

    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    accuracy = np.mean(all_preds == all_labels)  # 

    print(f'Epoch {epoch + 1} - Test - Avg Loss = {mean_loss:4.4f}, Accuracy = {accuracy * 100:.2f}%')

    return mean_loss, accuracy



def focal_loss(logits,labels, alpha, gamma):
    """Compute the focal loss between `logits` and the ground truth `labels`.

    Focal loss = -alpha_t * (1-pt)^gamma * log(pt)
    where pt is the probability of being classified to the true class.
    pt = p (if true class), otherwise pt = 1 - p. p = sigmoid(logit).

    Args:
      labels: A float tensor of size [batch, num_classes].
      logits: A float tensor of size [batch, num_classes].
      alpha: A float tensor of size [batch_size]
        specifying per-example weight for balanced cross entropy.
      gamma: A float scalar modulating loss from hard and easy examples.

    Returns:
      focal_loss: A float32 scalar representing normalized total loss.
    """    
    BCLoss = F.binary_cross_entropy_with_logits(input = logits, target = labels,reduction = "none")

    if gamma == 0.0:
        modulator = 1.0
    else:
        modulator = torch.exp(-gamma * labels * logits - gamma * torch.log(1 + 
            torch.exp(-1.0 * logits)))

    loss = modulator * BCLoss

    weighted_loss = alpha * loss
    focal_loss = torch.sum(weighted_loss)

    focal_loss /= torch.sum(labels)
    return focal_loss

def test_epoch_2(cfg, epoch, model, device, test_loader, criterion):
    model.eval()  # Set the model to evaluation mode.
    avg_loss = []  # Store the loss for each batch.
    all_preds = []  # Store all predictions.
    all_labels = []  # Store all ground-truth labels.
    num_classes = None

    with torch.no_grad():  # Disable gradient computation.
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs = inputs.float().to(device)  # Move the input data to the device.
            targets = targets.to(device)  # Move the labels to the device.

            outputs = model(inputs)
            if 'vit' in cfg.model:
                outputs = outputs.logits
            num_classes = outputs.shape[1]

            loss = criterion(outputs, targets)
            avg_loss.append(loss.detach().cpu().numpy())

            _, predicted = torch.max(outputs, 1)
            all_preds.append(predicted.cpu().numpy())
            all_labels.append(targets.cpu().numpy())

    mean_loss = np.mean(avg_loss)

    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    accuracy = np.mean(all_preds == all_labels)
    class_ids = np.arange(num_classes)
    per_class_recalls = recall_score(
        all_labels, all_preds, labels=class_ids, average=None, zero_division=0
    )
    per_class_test_counts = np.bincount(all_labels, minlength=num_classes)

    worst_class_accuracy = np.min(per_class_recalls)

    recall_diff = np.max(per_class_recalls) - np.min(per_class_recalls)
    recall_std = np.std(per_class_recalls)

    print(f'Epoch {epoch + 1} - Test Results:')
    print(f'  - Avg Loss             = {mean_loss:4.4f}')
    print(f'  - Overall Accuracy     = {accuracy * 100:.2f}%')
    print(f'  - Worst-Class Accuracy = {worst_class_accuracy * 100:.2f}%')
    print(f'  - Recall Diff (Max-Min)= {recall_diff :.2f}')
    print(f'  - Recall Std Dev       = {recall_std :.2f}')

    metrics = {
        'mean_loss': mean_loss,
        'accuracy': accuracy,
        'worst_class_accuracy': worst_class_accuracy,
        'recall_difference': recall_diff,
        'recall_std': recall_std,
        'per_class_recalls': per_class_recalls,
        'per_class_test_counts': per_class_test_counts
    }

    return metrics
