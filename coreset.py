import torch
import os,sys
sys.path.insert(0,".")
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18
from sklearn.cluster import KMeans
import numpy as np
import random
from utils.d2_sampling import GraphDensitySampler
from sklearn.preprocessing import MinMaxScaler
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import Ridge
import gc
from scipy.stats.mstats import winsorize

def normalize_with_sklearn(scores, norml, normr):
    
    scaler = MinMaxScaler(feature_range=(norml, normr))
    normalized = scaler.fit_transform(scores.reshape(-1, 1)).flatten()
    return normalized
    
def accuracy(logit, target):
    output = F.sigmoid(logit)
    output_np = output.cpu().numpy()
    target_np = target.cpu().numpy()
    auc_scores = []

    for i in range(target_np.shape[1]):
        auc = roc_auc_score(target_np[:, i], output_np[:, i])
        auc_scores.append(auc)
    mean_auc = sum(auc_scores) / target_np.shape[1]
    
    return mean_auc

def check_acc(w, featureset, labelset):
    y = torch.tensor(labelset).to(device='cuda')
    featureset = torch.tensor(featureset).to('cuda')
    output = w@featureset.mT
    
    # softmax_fun = torch.nn.Softmax(dim=1)
    # s = softmax_fun(output.mT)
    s = output.mT
    prec = accuracy(s, y)
    # acc = prec/len(featureset)
    return prec

def check_acc2(featureset, labelset, model):
    y = torch.tensor(labelset).to(device='cuda')
    featureset = torch.tensor(featureset).to('cuda')
    output = model.predict(featureset.cpu())
    output = torch.tensor(output).to('cuda')
    # softmax_fun = torch.nn.Softmax(dim=1)
    # s = softmax_fun(output.mT)
    prec = accuracy(output, y)
    # acc = prec/len(featureset)
    return prec

def accuracy_nat(logit, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
   # output = F.softmax(logit, dim=1)  # 
    output = logit
    maxk = max(topk)

    _, pred = output.topk(maxk, 1, True, True)  
    target_labels = target.argmax(dim=1)  #
    correct = pred.eq(target_labels.view(-1, 1).expand_as(pred))  # 

    res = []
    for k in topk:
        correct_k = correct[:, :k].view(-1).float().sum(0, keepdim=True)  # 
        res.append(correct_k * 100)  # 

    return res
    
def check_acc_nat(w, featureset, labelset):
    y = torch.tensor(labelset).to(device='cuda')
    featureset = torch.tensor(featureset).to('cuda')
    output = w@featureset.mT
    
    softmax_fun = torch.nn.Softmax(dim=1)
    s = softmax_fun(output.mT)
    prec, = accuracy_nat(s, y, topk=(1,))
    acc = prec/len(featureset)
    
    return acc.item()

def accuracy_nat_per_class(logit, target):
    num_classes = logit.size(1)  # 
    class_correct = torch.zeros(num_classes)  # 
    class_total = torch.zeros(num_classes)  # 

    _, pred = logit.max(dim=1)  # 
    target_labels = target.argmax(dim=1)  # 

    for cls in range(num_classes):
        class_mask = (target_labels == cls)  
        class_total[cls] = class_mask.sum().item()  
        class_correct[cls] = (pred[class_mask] == cls).sum().item()  

    class_acc = class_correct / class_total  
    class_acc[torch.isnan(class_acc)] = 0  

    return class_acc.mean()

def check_acc3_nat(featureset, labelset, model):
    y = torch.tensor(labelset).to(device='cpu')  # 
    featureset = torch.tensor(featureset)  # 
    output = model.predict(featureset)  # 
    output = torch.tensor(output)  # 

    softmax_fun = torch.nn.Softmax(dim=1)  # 
    output = softmax_fun(output)  # 

    acc = accuracy_nat_per_class(output, y)
    return acc.item()

def check_acc2_nat(featureset, labelset, model):
    y = torch.tensor(labelset).to(device='cpu')
    featureset = torch.tensor(featureset)
    output = model.predict(featureset)
    output = torch.tensor(output)
    
    softmax_fun = torch.nn.Softmax(dim=1)
    # s = softmax_fun(output.mT)
    output =  softmax_fun(output)
    prec, = accuracy_nat(output, y)
    acc = prec/len(featureset)
    
    return acc.item()

def check_acc2_nat_inat(featureset, labelset, model, batch_size=50000):
    total_samples = len(labelset)
    total_correct = 0
    
    labels_numeric = np.asarray(labelset) if not isinstance(labelset, np.ndarray) else labelset
    
    for i in range(0, total_samples, batch_size):
        batch_features = featureset[i:i+batch_size]
        batch_labels = labels_numeric[i:i+batch_size]
        batch_pred = model.predict(batch_features)
    
        batch_correct = np.sum(np.argmax(batch_pred, axis=1) == batch_labels)
        total_correct += batch_correct

        del batch_features, batch_pred
        gc.collect()
    
    return total_correct / total_samples 
    
def cal_w_regression(featureset, labelset, lambda_=1):
    ######## x shape : (n, d), n is number of data and d is dimension of data ########
    x = torch.tensor(featureset).to('cuda')
    labelset = labelset
    
    label_num = labelset.shape[1]
    w = []
    for idx in range(label_num):
        y = torch.tensor(labelset[:,idx]).type(torch.float32).to('cuda')
        
        if x.shape[0] > x.shape[1]:
            I = torch.eye(x.shape[1], device='cuda')
            H = (x.mT@x + lambda_*I)
            invH = torch.inverse(H)
            w.append(invH@x.mT@y)
        else:
            I = torch.eye(x.shape[0], device='cuda')
            H = (x@x.mT + lambda_*I)
            invH = torch.inverse(H)
            w.append(x.mT@invH@y)

    w = torch.stack(w, 0)
    return w
    
class CoresetSelection(object):
    
    
    @staticmethod
    def hard_selection(data_score, key, ratio, descending):
        start_ratio = 1-ratio
        score = data_score[key].cpu()
        score_sorted_index = score.argsort(descending=descending)
        all_num = score.shape[0]      
        # print(f'Low priority {key}: {score[score_sorted_index[:15]]}')
        # print(f'High priority {key}: {score[score_sorted_index[-15:]]}')
        return score_sorted_index[int(start_ratio*all_num):]
    
    @staticmethod
    def easy_selection(data_score, key, ratio, descending):
        score = data_score[key].cpu()
        score_sorted_index = score.argsort(descending=descending)
        all_num = score.shape[0]      
        # print(f'Low priority {key}: {score[score_sorted_index[:15]]}')
        # print(f'High priority {key}: {score[score_sorted_index[-15:]]}')
        return score_sorted_index[:int(ratio*all_num)]
    
    @staticmethod
    def window_selection(data_score, key, ratio, endratio, descending=0):
        if descending == 0:
            des = False
        else:
            des = True
        start_ratio = endratio-ratio
        if start_ratio>=0:
            end_ratio = endratio
        else:
            start_ratio = 0
            end_ratio = ratio
            
        score = data_score[key].cpu()
        score_sorted_index = score.argsort(descending=des)
        all_num = score.shape[0]      
        return score_sorted_index[int(start_ratio*all_num):int(end_ratio*all_num)]

        

    @staticmethod
    def mislabel_mask(data_score, mis_key, mis_num, mis_descending, coreset_key):
        mis_score = data_score[mis_key].cpu()
        mis_score_sorted_index = mis_score.argsort(descending=mis_descending)
        hard_index = mis_score_sorted_index[:mis_num]
        # print(f'Bad data -> High priority {mis_key}: {data_score[mis_key][hard_index][:15]}')
        #print(f'Prune {hard_index.shape[0]} samples.')
        easy_index = mis_score_sorted_index[mis_num:]
        data_score[coreset_key] = data_score[coreset_key][easy_index]

        return data_score, easy_index

        
    @staticmethod
    def stratified_sampling(data_score, coreset_key, coreset_num):
        stratas = 20
        #print('Using stratified sampling...')
        score = data_score[coreset_key].cpu()
        total_num = coreset_num
        min_score = torch.min(score)
        max_score = torch.max(score) * 1.0001
        step = (max_score - min_score) / stratas
        def bin_range(k):
            return min_score + k * step, min_score + (k + 1) * step

        strata_num = []
        ##### calculate number for each strata #####
        for i in range(stratas):
            start, end = bin_range(i)
            num = torch.logical_and(score >= start, score < end).sum()
            strata_num.append(num)

        strata_num = torch.tensor(strata_num)

        def bin_allocate(num, bins):
            sorted_index = torch.argsort(bins)
            sort_bins = bins[sorted_index]
            num_bin = bins.shape[0]
            rest_exp_num = num
            budgets = []
            for i in range(num_bin):
                rest_bins = num_bin - i
                avg = rest_exp_num // rest_bins
                cur_num = min(sort_bins[i].item(), avg)
                budgets.append(cur_num)
                rest_exp_num -= cur_num

            rst = torch.zeros((num_bin,)).type(torch.int)
            rst[sorted_index] = torch.tensor(budgets).type(torch.int)

            return rst

        budgets = bin_allocate(total_num, strata_num)
        ##### sampling in each strata #####
        selected_index = []
        sample_index = torch.arange(data_score[coreset_key].shape[0])

        for i in range(stratas):
            start, end = bin_range(i)
            mask = torch.logical_and(score >= start, score < end)
            pool = sample_index[mask]
            rand_index = torch.randperm(pool.shape[0])
            selected_index += [idx.item() for idx in pool[rand_index][:budgets[i]]]

        return selected_index, None
    
    def stratified_staff(data_score_A, data_score_B, key, coreset_num, stratas=20):
        """
        data_score_A: Dictionary of scores from the smaller model.
        data_score_B: Dictionary of scores from the original model.
        key: Key shared by both dictionaries (for example, 'loss').
        """
        # Extract tensors.
        sA = data_score_A[key].cpu()
        sB = data_score_B[key].cpu()
        print(sA.shape[0])
        print()
        if sA.shape[0] != sB.shape[0]:
            raise ValueError("ScoreA and ScoreB files must have the same number of samples.")
        total_num = int(coreset_num)
        # 1. Determine bin boundaries based on ScoreA.
        min_sA = torch.min(sA)
        max_sA = torch.max(sA) * 1.0001
        step = (max_sA - min_sA) / stratas
        strata_indices = []
        strata_n = []
        weights = []
        # 2. Compute information for each bin.
        for i in range(stratas):
            lower = min_sA + i * step
            upper = min_sA + (i + 1) * step
            mask = torch.logical_and(sA >= lower, sA < upper)
            indices = torch.where(mask)[0]
            strata_indices.append(indices)
            n_k = indices.shape[0]
            strata_n.append(n_k)
            if n_k > 0:
                # Core formula: W_k = sum(ScoreB) / sum(ScoreA).
                sum_A = torch.sum(sA[mask])
                sum_B = torch.sum(sB[mask])
                # Avoid division by a very small value.
                w_k = (sum_B / (sum_A + 1e-8)).item()
                weights.append(w_k)
            else:
                weights.append(0.0)
        strata_n = torch.tensor(strata_n)
        weights = torch.tensor(weights)
        # 3. Compute the base budget using the existing fair allocation logic.
        # Note: bin_allocate must be defined in the class or as a static method.
        base_budgets = CoresetSelection.bin_allocate(total_num, strata_n)
        # 4. Adjust the budget using the ScoreB/ScoreA weights.
        modified_budgets = base_budgets.float() * weights
        # 5. Normalize and round to ensure the total equals coreset_num.
        if modified_budgets.sum() > 0:
            scale = total_num / modified_budgets.sum()
            final_budgets = torch.round(modified_budgets * scale).type(torch.int)
        else:
            final_budgets = base_budgets
        # 6. Correct rounding differences and enforce bin capacity limits.
        diff = total_num - final_budgets.sum()
        # Apply a simple correction for any remaining difference.
        if diff != 0:
            adjust_idx = torch.argmax(strata_n)
            final_budgets[adjust_idx] += diff
        # Ensure final_budgets[i] <= strata_n[i].
        for i in range(stratas):
            if final_budgets[i] > strata_n[i]:
                excess = final_budgets[i] - strata_n[i]
                final_budgets[i] = strata_n[i]
                # Try to distribute the excess to bins that are not full.
                for j in range(stratas):
                    if final_budgets[j] < strata_n[j]:
                        can_take = min(excess, strata_n[j] - final_budgets[j])
                        final_budgets[j] += can_take
                        excess -= can_take
                        if excess <= 0: break
        # 7. Perform the final sampling.
        selected_index = []
        for i in range(stratas):
            pool = strata_indices[i]
            count = int(final_budgets[i])
            if count > 0:
                perm = torch.randperm(pool.shape[0])
                selected_index += pool[perm[:count]].tolist()

        return selected_index

    @staticmethod
    def bin_allocate(num, bins):
        """Keep the allocation logic consistent with the original implementation."""
        sorted_index = torch.argsort(bins)
        sort_bins = bins[sorted_index]
        num_bin = bins.shape[0]
        rest_exp_num = num
        budgets = []
        for i in range(num_bin):
            rest_bins = num_bin - i
            avg = rest_exp_num // rest_bins
            cur_num = min(sort_bins[i].item(), avg)
            budgets.append(cur_num)
            rest_exp_num -= cur_num

        rst = torch.zeros((num_bin,)).type(torch.int)
        rst[sorted_index] = torch.tensor(budgets).type(torch.int)
        return rst
    

    @staticmethod
    def random_selection(total_num, num):
        score_random_index = torch.randperm(total_num)

        return score_random_index[:int(num)]
    
    @staticmethod
    def d2_selection(data_score, key, coreset_num, cfg):
        score = data_score[key].cpu()
        # score_sorted_index = score.argsort(descending=True)
        features = data_score['feature_nofinetune'].cpu().numpy()
        # load data scores from training 100% data
        sampling_method = GraphDensitySampler(X=features, y=None,
                                              gamma=cfg.gamma,
                                              seed=0, importance_scores=data_score[cfg.coreset_key], args=cfg)
                                              # n_neighbor=args.n_neighbor, graph_mode=args.graph_mode,
                                              # graph_sampling_mode=args.graph_sampling_mode,
                                              # precomputed_dists=args.precomputed_dists,
                                              # precomputed_neighbors=args.precomputed_neighbors
                                              # )
        coreset_index = sampling_method.select_batch_(coreset_num)
        coreset_index_no_mis = np.array(coreset_index.copy())
       # coreset_index = score_sorted_index[coreset_index]
        graph_scores = sampling_method.starting_density

        return np.array(coreset_index)
    
    def moderate(data_score, ratio):

        label = data_score['targets'].cpu()
        score_clear = label.numpy()
        features = data_score['feature_nofinetune'].cpu().numpy()
        total_samples = score_clear.shape[0]
        
        categories = np.argmax(score_clear, axis=1)
        unique_classes = np.unique(categories)
        centers = np.zeros_like(features)  
        
        for cls in unique_classes:
            indices = np.where(categories == cls)[0]
            cls_center = features[indices].mean(axis=0)
            centers[indices] = cls_center
        
        difficulties = np.linalg.norm(features - centers, axis=1)
        
        sorted_indices = np.argsort(difficulties)
        k = int(total_samples * ratio + 0.5)  
        start = (total_samples - k) // 2
        selected_indices = sorted_indices[start:start + k]
        
        return selected_indices
    
    def label_sample(data_score, ratio, key, mode, cfg, centerratio=0, mis_ratio=0):
        print(mode)
        label = data_score['targets'].cpu()
        score_clear = label.numpy()
        total_samples = score_clear.shape[0]

        all_class_samples = [np.where(score_clear[:, idx] == 1)[0] for idx in range(score_clear.shape[1])]
        unselected_samples_by_class = {idx: [] for idx in range(score_clear.shape[1])}
        data_score_np = data_score[key].cpu().numpy()
        disease_scores = [
            np.mean(data_score_np[samples]) if len(samples) > 0 else 0 
            for samples in all_class_samples
        ]
        disease_scores = torch.tensor(disease_scores, dtype=torch.float32)
        # disease_scores = disease_scores / disease_scores.sum()
        # itx = 0
        # for i in range(score_clear.shape[1]):
        #     class_num = len(all_disease_samples[i])
        #     itx += disease_scores[i]*class_num
        # itx = itx/total_samples
        # score_weight_normalized = (disease_scores/itx)**labelsigma
        # sorted_indices = np.argsort(-score_weight_normalized)
        
        all_selected_coreset_index = []

        for idx in range(score_clear.shape[1]):
            idx = int(idx)
            disease_samples = all_class_samples[idx]
            unselected_samples = disease_samples
            unselected_samples_by_class[idx] = unselected_samples
            num_selected = int(len(unselected_samples) * ratio )  
            coreset_num = num_selected
            data_score_temp = {key: data_score[key][unselected_samples]}
            data_score_temp['feature_nofinetune'] = data_score['feature_nofinetune'][unselected_samples]           
            class_ratio = ratio
            if mode == "ccs":
                mis_num = int(mis_ratio * len(unselected_samples))
                if key=='accumulated_margin':
                    data_score_mask, score_index = CoresetSelection.mislabel_mask(
                        data_score=data_score_temp, mis_key=key, mis_num=mis_num,
                        mis_descending=False, coreset_key=key
                    )
                else:
                    data_score_mask, score_index = CoresetSelection.mislabel_mask(
                        data_score=data_score_temp, mis_key=key, mis_num=mis_num,
                        mis_descending=True, coreset_key=key
                    )
                coreset_index, _ = CoresetSelection.stratified_sampling(
                    data_score=data_score_mask, coreset_key=key, coreset_num=coreset_num
                )
                global_coreset_index = [unselected_samples[score_index][i] for i in coreset_index]
            elif mode == "hard":
                coreset_index = CoresetSelection.hard_selection(data_score=data_score_temp, key=key, ratio=class_ratio, descending=False)
                global_coreset_index = [unselected_samples[i] for i in coreset_index]
            elif mode == "moderate":
                coreset_index = CoresetSelection.moderate(data_score=data_score_temp,ratio=class_ratio, descending=False)
                global_coreset_index = [unselected_samples[i] for i in coreset_index]
            elif mode == 'random':
                coreset_index = CoresetSelection.random_selection(total_num=len(unselected_samples), num=coreset_num)
                global_coreset_index = [unselected_samples[i] for i in coreset_index]
            elif mode == 'graph':
                coreset_index = CoresetSelection.d2_selection(data_score=data_score_temp, key=key, coreset_num=coreset_num, cfg=cfg)
                global_coreset_index = [unselected_samples[i] for i in coreset_index]
                    
            all_selected_coreset_index.extend(global_coreset_index)
            if len(all_selected_coreset_index) == total_samples:
                break
            
        expected_num_samples = int(total_samples * ratio)
        if len(all_selected_coreset_index) > expected_num_samples:
            excess = len(all_selected_coreset_index) - expected_num_samples
            all_selected_coreset_index = random.sample(all_selected_coreset_index, expected_num_samples)
        # If the number of selected samples is less than expected, randomly add some
        elif len(all_selected_coreset_index) < expected_num_samples:
            deficit = expected_num_samples - len(all_selected_coreset_index)
            remaining_samples = list(set(range(total_samples)) - set(all_selected_coreset_index))
            additional_samples = random.sample(remaining_samples, deficit)
            all_selected_coreset_index.extend(additional_samples)
        
        return all_selected_coreset_index
    


    def class_sample_nat(cfg, data_score, ratio, key, classratiomode='difficulty', classselectmode='ccs', featuremode='nofinetune', difficultymode='winsorized', labelsigma=1, alpha=1):
        print('classratiomode:',classratiomode)
        print('classselectmode:',classselectmode)
        print('featuremode:',featuremode)
        
        data = data_score
        label = data['targets'].cpu()
        feature_all = data['feature_nofinetune'].cpu()
        feature_all_finetune = data['feature'].cpu()
        score_clear = label.numpy()
        total_samples = score_clear.shape[0]
        all_samples = [np.where(score_clear[:, idx] == 1)[0] for idx in range(score_clear.shape[1])]   
        class_num_total = 0
        #####
        data_score_np = data[key].cpu().numpy()
        if difficultymode == 'winsorized':
            print('difficultymode:winsorized')
            disease_scores = [
                # limits=[0.1, 0.1] significa que os 10% inferiores e 10% superiores dos dados são limitados
                np.mean(winsorize(data_score_np[samples], limits=[0.05, 0.05])) if len(samples) > 0 else 0 
                for samples in all_samples
            ]
        else:
            # Cálculo da média original
            disease_scores = [
                np.mean(data_score_np[samples]) if len(samples) > 0 else 0 
                for samples in all_samples
            ]
        disease_scores = torch.tensor(disease_scores, dtype=torch.float32)
        disease_scores = disease_scores / disease_scores.sum()
        itx = 0
        for i in range(score_clear.shape[1]):
            class_num = len(all_samples[i])
            itx += disease_scores[i]*class_num
            class_num_total += class_num 
        itx = itx/class_num_total  
        score_weight_normalized = (disease_scores/itx)
        ####
        if classratiomode =='difficulty':
            print('classratiomode:difficulty')
            score_weight = score_weight_normalized
        elif classratiomode =='number':
            print('classratiomode:number')
            score_weight = np.ones_like(score_weight_normalized)
        sorted_indices = np.argsort(-score_weight_normalized)
    
        def get_index(centerratio): 
            unselected_samples_by_class = {idx: [] for idx in range(score_clear.shape[1])}
            num_sample_by_class = {idx: [] for idx in range(score_clear.shape[1])}
            all_selected_coreset_index = []
            for idx in sorted_indices:
                idx = int(idx)
                class_samples = all_samples[idx]
                unselected_samples_by_class[idx] = class_samples
                class_weight = score_weight[idx]
                num_selected = int(len(class_samples) * ratio * class_weight)  
                coreset_num = min(int(num_selected), len(class_samples))
                num_sample_by_class[idx] = coreset_num
                
            diff = int(total_samples * ratio) - sum(num_sample_by_class.values())
            unselected_counts = {idx: len(unselected_samples_by_class[idx]) - num_sample_by_class[idx]
                                for idx in range(score_clear.shape[1])}
            total_unselected = sum(unselected_counts.values())
            unselected_ratios = {idx: (count / total_unselected if total_unselected > 0 else 0)
                                for idx, count in unselected_counts.items()}
            if diff > 0:
                selected_classes = random.choices(list(unselected_ratios.keys()), weights=list(unselected_ratios.values()), k=diff)
                for idx in selected_classes:
                    num_sample_by_class[idx] += 1
            else:
                diff = -diff  
                selected_classes = random.choices(list(num_sample_by_class.keys()), weights=list(num_sample_by_class.values()), k=diff)
                for idx in selected_classes:
                    num_sample_by_class[idx] -= 1      
            all_selected_coreset_index = []
            for idx in range(score_clear.shape[1]):
                idx = int(idx)
                class_samples = unselected_samples_by_class[idx]
                coreset_num = num_sample_by_class[idx]
                data_score_temp = {key: data[key][class_samples]}
                class_ratio = coreset_num/len(class_samples)           
                ########
                if classselectmode == 'window':
                    coreset_index = CoresetSelection.window_selection(data_score=data_score_temp, key=key, ratio=class_ratio, endratio=centerratio)
                    global_coreset_index = [class_samples[i] for i in coreset_index]  
                ################
                elif classselectmode == 'ccs':
                    mis_num = min(int(centerratio * len(class_samples)),len(class_samples)-coreset_num)
                    data_score_mask, score_index = CoresetSelection.mislabel_mask(
                        data_score=data_score_temp, mis_key=key, mis_num=mis_num,
                        mis_descending=True, coreset_key=key
                    )
                    coreset_index, _ = CoresetSelection.stratified_sampling(
                        data_score=data_score_mask, coreset_key=key, coreset_num=coreset_num
                    )
                    global_coreset_index = [class_samples[score_index][i] for i in coreset_index]
                #############
                all_selected_coreset_index.extend(global_coreset_index)
                if len(all_selected_coreset_index) == total_samples:
                    break       
            expected_num_samples = int(total_samples * ratio)
            if len(all_selected_coreset_index) > expected_num_samples:
                excess = len(all_selected_coreset_index) - expected_num_samples
                all_selected_coreset_index = random.sample(all_selected_coreset_index, expected_num_samples)
            elif len(all_selected_coreset_index) < expected_num_samples:
                deficit = expected_num_samples - len(all_selected_coreset_index)
                remaining_samples = list(set(range(total_samples)) - set(all_selected_coreset_index))
                additional_samples = random.sample(remaining_samples, deficit)
                all_selected_coreset_index.extend(additional_samples)
        
            return all_selected_coreset_index
            
        label_all = score_clear
        
        def get_best_index():
            #centerratio_all = np.linspace(0, 1, 20)  
            centerratio_all = np.linspace(0, 1, 11) 
            labelsigma_all =  [labelsigma]    #[1]
            best_index = []
            best_auc = 0
            best_center = -1
            best_labelsigma = -1
            
            for j in labelsigma_all:
                for i in centerratio_all:
                    selected_index = get_index(centerratio=i)
                    selected_label = label_all[selected_index]
                    if featuremode=='nofinetune':
                        selected_feature = feature_all[selected_index]
                        feature_set = feature_all
                    elif featuremode=='finetune':
                        selected_feature = feature_all_finetune[selected_index]
                        feature_set = feature_all_finetune
                    ############
                    if cfg.coreset_mode=='window' or cfg.coreset_mode=='labelwindow_test':
                        w = cal_w_regression(selected_feature, selected_label)
                        outcome = check_acc_nat(w=w, featureset=feature_set, labelset=label_all)
                    # ###########
                    else:
                        model = Ridge(alpha=alpha)
                        model.fit(selected_feature, selected_label)
                        if cfg.krrmode=='all':
                            outcome = check_acc2_nat(featureset=feature_set,labelset=label_all,model=model)   
                        elif cfg.krrmode=='class':
                            outcome = check_acc3_nat(featureset=feature_set,labelset=label_all,model=model) 
                    ##########          
                   # print(f"labelsigma={j}, centerratio={i}, outcome={outcome}")
                    if outcome > best_auc:
                        best_index = selected_index
                        best_auc = outcome
                        best_center = i
                        best_labelsigma = j 
            print(f"Best labelsigma: {best_labelsigma}, Best centerratio: {best_center}")
            return best_index
            
        all_index = get_best_index()
            
        return all_index

 

    def class_sample_inat(cfg, data_score, ratio, key, classratiomode='difficulty', classselectmode='ccs', featuremode='nofinetune', labelsigma=1):
        print(f'classratiomode:{classratiomode}, classselectmode:{classselectmode}, featuremode:{featuremode}')
        
        data = data_score
        label = data['targets'].cpu().numpy()  
        feature_all = data['feature_nofinetune'].cpu()
        
        all_samples = [np.where(label[:, idx] == 1)[0] for idx in range(label.shape[1])]
        class_num_total = sum(len(samples) for samples in all_samples)
        
        data_score_np = data[key].cpu().numpy()
        disease_scores = np.array([
            np.mean(data_score_np[samples]) if len(samples) > 0 else 0 
            for samples in all_samples
        ])
        disease_scores = disease_scores / disease_scores.sum()
        
        itx = sum(disease_scores[i] * len(all_samples[i]) for i in range(label.shape[1])) / class_num_total
        score_weight_normalized = disease_scores / itx
        
        score_weight = score_weight_normalized if classratiomode == 'difficulty' else np.ones_like(score_weight_normalized)
        sorted_indices = np.argsort(-score_weight_normalized)
        
        def get_index(centerratio):

            unselected_samples_by_class = {idx: all_samples[idx].copy() for idx in range(label.shape[1])}
            num_sample_by_class = np.zeros(label.shape[1], dtype=np.int32)
            for idx in sorted_indices:
                idx = int(idx)
                class_samples = all_samples[idx]
                class_weight = score_weight[idx]
                num_selected = int(len(class_samples) * ratio * class_weight)
                num_sample_by_class[idx] = min(num_selected, len(class_samples))
            
            total_selected = num_sample_by_class.sum()
            target_total = int(len(label) * ratio)
            diff = target_total - total_selected
            
            if diff != 0:
                if diff > 0:
                    remaining_counts = np.array([
                        len(unselected_samples_by_class[idx]) - num_sample_by_class[idx] 
                        for idx in range(label.shape[1])
                    ])
                    probs = remaining_counts / remaining_counts.sum()
                    additional = np.random.choice(
                        label.shape[1], size=abs(diff), p=probs
                    )
                    np.add.at(num_sample_by_class, additional, 1)
                else:
                    probs = num_sample_by_class / num_sample_by_class.sum()
                    to_remove = np.random.choice(
                        label.shape[1], size=abs(diff), p=probs
                    )
                    np.subtract.at(num_sample_by_class, to_remove, 1)
            selected_indices = []            
            for idx in range(label.shape[1]):
                class_samples = unselected_samples_by_class[idx]
                coreset_num = num_sample_by_class[idx]
                
                if coreset_num <= 0:
                    continue
                if cfg.dataset=='inat2021':
                    data_score_temp = {'el2n': data['el2n'][class_samples]}  
                else:
                    data_score_temp = {key: data[key][class_samples]}
                class_ratio = coreset_num / len(class_samples)
                
                if classselectmode == 'window':
                    coreset_index = CoresetSelection.window_selection(
                        data_score=data_score_temp, key=key, 
                        ratio=class_ratio, endratio=centerratio
                    )
                    global_indices = class_samples[coreset_index]
                elif classselectmode == 'ccs':
                    mis_num = min(int(centerratio * len(class_samples)), len(class_samples)-coreset_num)
                    data_score_mask, score_index = CoresetSelection.mislabel_mask(
                        data_score=data_score_temp, mis_key=key, mis_num=mis_num,
                        mis_descending=True, coreset_key=key
                    )
                    coreset_index, _ = CoresetSelection.stratified_sampling(
                        data_score=data_score_mask, coreset_key=key, coreset_num=coreset_num
                    )
                    global_indices = class_samples[score_index][coreset_index]
                
                selected_indices.extend(global_indices.tolist())
                
                if len(selected_indices) >= target_total:
                    break
            if len(selected_indices) > target_total:
                selected_indices = selected_indices[:target_total]
            elif len(selected_indices) < target_total:
                remaining = list(set(range(len(label))) - set(selected_indices))
                selected_indices.extend(np.random.choice(remaining, target_total - len(selected_indices), replace=False))
            
            return np.array(selected_indices, dtype=np.int32)
        
        def get_best_index():
            centerratio_all = np.linspace(0, 1, 20)  # 
            best_params = {
                'index': np.array([], dtype=np.int32),
                'auc': 0,
                'center': -1,
                'labelsigma': -1
            }
            
            feature_set = feature_all if featuremode == 'nofinetune' else data['feature'].cpu()
            for i in centerratio_all:
                selected_index = get_index(i)
                selected_label = label[selected_index]
                selected_feature = feature_set[selected_index]
                if cfg.coreset_mode in ['window', 'labelwindow_test']:
                    w = cal_w_regression(selected_feature, selected_label)
                    outcome = check_acc_nat(w=w, featureset=feature_set, labelset=label)
                else:
                    model = Ridge(alpha=1.0)
                    model.fit(selected_feature, selected_label)
                    del selected_feature, selected_label
                    gc.collect()
                    if cfg.dataset=='inat2021':
                        label_numeric = np.argmax(label, axis=1) 
                        outcome = check_acc2_nat_inat(featureset=feature_set, labelset=label_numeric,model=model)
                    else:
                        outcome = check_acc2_nat(featureset=feature_set, labelset=label, model=model)
                if outcome > best_params['auc']:
                    best_params.update({
                        'index': selected_index.copy(),
                        'auc': outcome,
                        'center': i,
                        'labelsigma': labelsigma
                    })

            
            print(f"Best params: labelsigma={best_params['labelsigma']}, centerratio={best_params['center']}")
            return best_params['index']
        
        return get_best_index()
    
    
    def class_sample_nat_traverse(cfg, data_score, ratio, key, classratiomode='difficulty', classselectmode='ccs', difficultymode='average', featuremode='nofinetune', centerratio=1, wf=0.05, alpha=1, beta=1.0):

        print('featuremode:',featuremode)
        mis_descending = True
        descending = 0
        
        data = data_score
        label = data['targets'].cpu()
        feature_all = data['feature_nofinetune'].cpu()
        #feature_all_finetune = data['feature'].cpu()
        score_clear = label.numpy()
        total_samples = score_clear.shape[0]
        all_samples = [np.where(score_clear[:, idx] == 1)[0] for idx in range(score_clear.shape[1])]   
        class_num_total = 0
        #####
        data_score_np = data[key].cpu().numpy()
        if key == 'accumulated_margin':
            min_val = np.min(data_score_np)
            max_val = np.max(data_score_np)
            normalized_scores = 1 - (data_score_np - min_val) / (max_val - min_val)
            final_scores = normalized_scores
            data_score_np = final_scores
            descending = 1
            mis_descending = False
            
        if difficultymode == 'winsorized':
            print('difficultymode:winsorized')
            print('winsorization factor:',wf)
            disease_scores = [
                np.mean(winsorize(data_score_np[samples], limits=[wf, wf])) if len(samples) > 0 else 0 
                for samples in all_samples
            ]
        else:
            # Cálculo da média original
            disease_scores = [
                np.mean(data_score_np[samples]) if len(samples) > 0 else 0 
                for samples in all_samples
            ]
        disease_scores = torch.tensor(disease_scores, dtype=torch.float32)
                    
        disease_scores = disease_scores / disease_scores.sum()
        itx = 0
        for i in range(score_clear.shape[1]):
            class_num = len(all_samples[i])
            itx += disease_scores[i]*class_num
            class_num_total += class_num 
        itx = itx/class_num_total  
        score_weight_normalized = (disease_scores/itx)
        ####
        if classratiomode == 'difficulty':
            print('classratiomode:difficulty')
            
            # --- [Change 2] Apply the nonlinear beta parameter. ---
            if beta != 1.0:
                print(f'Applying non-linearity with beta={beta}')
                # Convert to NumPy for computation.
                if isinstance(score_weight_normalized, torch.Tensor):
                    w_temp = score_weight_normalized.numpy()
                else:
                    w_temp = score_weight_normalized
                
                # 1. Apply the power transformation.
                # beta > 1: Increase the differences (make difficult classes harder).
                # beta < 1: Reduce the differences (make weights more uniform).
                w_pow = np.power(w_temp, beta)
                
                # 2. Renormalize the weights.
                # Ensure sum(Class_Size_i * Weight_i) = Total_Samples;
                # otherwise, the total sample count may deviate substantially from the target ratio.
                current_weighted_sum = 0
                for i in range(score_clear.shape[1]):
                    current_weighted_sum += w_pow[i] * len(all_samples[i])
                
                # Compute the scaling factor.
                scaling_factor = total_samples / current_weighted_sum if current_weighted_sum > 0 else 1
                
                score_weight = w_pow * scaling_factor
                
                # Convert back to a tensor for consistency with subsequent operations.
                score_weight = torch.tensor(score_weight, dtype=torch.float32)
            else:
                score_weight = score_weight_normalized
        elif classratiomode =='number':
            print('classratiomode:number')
            score_weight = np.ones_like(score_weight_normalized)
        sorted_indices = np.argsort(-score_weight_normalized)
    
        def get_index(centerratio): 
            unselected_samples_by_class = {idx: [] for idx in range(score_clear.shape[1])}
            num_sample_by_class = {idx: [] for idx in range(score_clear.shape[1])}
            all_selected_coreset_index = []
            for idx in sorted_indices:
                idx = int(idx)
                class_samples = all_samples[idx]
                unselected_samples_by_class[idx] = class_samples
                class_weight = score_weight[idx]
                num_selected = int(len(class_samples) * ratio * class_weight)  
                coreset_num = min(int(num_selected), len(class_samples))
                num_sample_by_class[idx] = coreset_num
                
            diff = int(total_samples * ratio) - sum(num_sample_by_class.values())
            unselected_counts = {idx: len(unselected_samples_by_class[idx]) - num_sample_by_class[idx]
                                for idx in range(score_clear.shape[1])}
            total_unselected = sum(unselected_counts.values())
            unselected_ratios = {idx: (count / total_unselected if total_unselected > 0 else 0)
                                for idx, count in unselected_counts.items()}
            if diff > 0:
                selected_classes = random.choices(list(unselected_ratios.keys()), weights=list(unselected_ratios.values()), k=diff)
                for idx in selected_classes:
                    num_sample_by_class[idx] += 1
            else:
                diff = -diff  
                selected_classes = random.choices(list(num_sample_by_class.keys()), weights=list(num_sample_by_class.values()), k=diff)
                for idx in selected_classes:
                    num_sample_by_class[idx] -= 1      
            all_selected_coreset_index = []
            for idx in range(score_clear.shape[1]):
                idx = int(idx)
                class_samples = unselected_samples_by_class[idx]
                coreset_num = num_sample_by_class[idx]
                data_score_temp = {key: data[key][class_samples]}
                class_ratio = coreset_num/len(class_samples)           
                ########
                if classselectmode == 'window':
                    coreset_index = CoresetSelection.window_selection(data_score=data_score_temp, key=key, ratio=class_ratio, endratio=centerratio,descending=descending)
                    global_coreset_index = [class_samples[i] for i in coreset_index]  
                ################
                elif classselectmode == 'ccs':
                    mis_num = min(int(centerratio * len(class_samples)),len(class_samples)-coreset_num)
                    data_score_mask, score_index = CoresetSelection.mislabel_mask(
                        data_score=data_score_temp, mis_key=key, mis_num=mis_num,
                        mis_descending=mis_descending, coreset_key=key
                    )
                    coreset_index, _ = CoresetSelection.stratified_sampling(
                        data_score=data_score_mask, coreset_key=key, coreset_num=coreset_num
                    )
                    global_coreset_index = [class_samples[score_index][i] for i in coreset_index]
                #############
                all_selected_coreset_index.extend(global_coreset_index)
                if len(all_selected_coreset_index) == total_samples:
                    break       
            expected_num_samples = int(total_samples * ratio)
            if len(all_selected_coreset_index) > expected_num_samples:
                excess = len(all_selected_coreset_index) - expected_num_samples
                all_selected_coreset_index = random.sample(all_selected_coreset_index, expected_num_samples)
            elif len(all_selected_coreset_index) < expected_num_samples:
                deficit = expected_num_samples - len(all_selected_coreset_index)
                remaining_samples = list(set(range(total_samples)) - set(all_selected_coreset_index))
                additional_samples = random.sample(remaining_samples, deficit)
                all_selected_coreset_index.extend(additional_samples)
        
            return all_selected_coreset_index
            
        label_all = score_clear
        
        def get_best_index():
            centerratio_all = [centerratio]  
            labelsigma_all =  [1]
            best_index = []
            best_auc = 0
            best_center = -1
            best_labelsigma = -1
            
            for j in labelsigma_all:
                for i in centerratio_all:
                    selected_index = get_index(centerratio=i)
                    selected_label = label_all[selected_index]
                    if featuremode=='nofinetune':
                        selected_feature = feature_all[selected_index]
                        feature_set = feature_all
                    # elif featuremode=='finetune':
                    #     selected_feature = feature_all_finetune[selected_index]
                    #     feature_set = feature_all_finetune
                    ############
                    if cfg.coreset_mode=='window' or cfg.coreset_mode=='labelwindow_test':
                        w = cal_w_regression(selected_feature, selected_label)
                        outcome = check_acc_nat(w=w, featureset=feature_set, labelset=label_all)
                    # ###########
                    else:
                        model = Ridge(alpha=alpha)
                        model.fit(selected_feature, selected_label)
                        outcome = check_acc2_nat(featureset=feature_set,labelset=label_all,model=model)    
                    ##########          
                   # print(f"labelsigma={j}, centerratio={i}, outcome={outcome}")
                    if outcome > best_auc:
                        best_index = selected_index
                        best_auc = outcome
                        best_center = i
                        best_labelsigma = j 
            print(f"Best labelsigma: {best_labelsigma}, Best centerratio: {best_center}")
            return best_index
            
        all_index = get_best_index()
            
        return all_index

    def class_sample_nat_traverse_norm(cfg, data_score, ratio, key, classratiomode='difficulty', classselectmode='ccs', featuremode='nofinetune', centerratio=1, norml=0, normr=1):
        print('classratiomode:',classratiomode)
        print('classselectmode:',classselectmode)
        print('featuremode:',featuremode)
        
        data = data_score
        label = data['targets'].cpu()
        feature_all = data['feature_nofinetune'].cpu()
        feature_all_finetune = data['feature'].cpu()
        score_clear = label.numpy()
        total_samples = score_clear.shape[0]
        all_samples = [np.where(score_clear[:, idx] == 1)[0] for idx in range(score_clear.shape[1])]   
        class_num_total = 0
        #####
        data_score_np = data[key].cpu().numpy()
        if normr != norml:
            print('normalize')
            data_score_np = normalize_with_sklearn(data_score_np, norml, normr)
        
        disease_scores = [
            np.mean(data_score_np[samples]) if len(samples) > 0 else 0 
            for samples in all_samples
        ]
        disease_scores = torch.tensor(disease_scores, dtype=torch.float32)
        disease_scores = disease_scores / disease_scores.sum()
        itx = 0
        for i in range(score_clear.shape[1]):
            class_num = len(all_samples[i])
            itx += disease_scores[i]*class_num
            class_num_total += class_num 
        itx = itx/class_num_total  
        score_weight_normalized = (disease_scores/itx)
        ####
        if classratiomode =='difficulty':
            print('classratiomode:difficulty')
            score_weight = score_weight_normalized
        elif classratiomode =='number':
            print('classratiomode:number')
            score_weight = np.ones_like(score_weight_normalized)
        sorted_indices = np.argsort(-score_weight_normalized)
    
        def get_index(centerratio): 
            unselected_samples_by_class = {idx: [] for idx in range(score_clear.shape[1])}
            num_sample_by_class = {idx: [] for idx in range(score_clear.shape[1])}
            all_selected_coreset_index = []
            for idx in sorted_indices:
                idx = int(idx)
                class_samples = all_samples[idx]
                unselected_samples_by_class[idx] = class_samples
                class_weight = score_weight[idx]
                num_selected = int(len(class_samples) * ratio * class_weight)  
                coreset_num = min(int(num_selected), len(class_samples))
                num_sample_by_class[idx] = coreset_num
                
            diff = int(total_samples * ratio) - sum(num_sample_by_class.values())
            unselected_counts = {idx: len(unselected_samples_by_class[idx]) - num_sample_by_class[idx]
                                for idx in range(score_clear.shape[1])}
            total_unselected = sum(unselected_counts.values())
            unselected_ratios = {idx: (count / total_unselected if total_unselected > 0 else 0)
                                for idx, count in unselected_counts.items()}
            if diff > 0:
                selected_classes = random.choices(list(unselected_ratios.keys()), weights=list(unselected_ratios.values()), k=diff)
                for idx in selected_classes:
                    num_sample_by_class[idx] += 1
            else:
                diff = -diff  
                selected_classes = random.choices(list(num_sample_by_class.keys()), weights=list(num_sample_by_class.values()), k=diff)
                for idx in selected_classes:
                    num_sample_by_class[idx] -= 1      
            all_selected_coreset_index = []
            for idx in range(score_clear.shape[1]):
                idx = int(idx)
                class_samples = unselected_samples_by_class[idx]
                coreset_num = num_sample_by_class[idx]
                data_score_temp = {key: data[key][class_samples]}
                class_ratio = coreset_num/len(class_samples)           
                ########
                if classselectmode == 'window':
                    coreset_index = CoresetSelection.window_selection(data_score=data_score_temp, key=key, ratio=class_ratio, endratio=centerratio)
                    global_coreset_index = [class_samples[i] for i in coreset_index]  
                ################
                elif classselectmode == 'ccs':
                    mis_num = min(int(centerratio * len(class_samples)),len(class_samples)-coreset_num)
                    data_score_mask, score_index = CoresetSelection.mislabel_mask(
                        data_score=data_score_temp, mis_key=key, mis_num=mis_num,
                        mis_descending=True, coreset_key=key
                    )
                    coreset_index, _ = CoresetSelection.stratified_sampling(
                        data_score=data_score_mask, coreset_key=key, coreset_num=coreset_num
                    )
                    global_coreset_index = [class_samples[score_index][i] for i in coreset_index]
                #############
                all_selected_coreset_index.extend(global_coreset_index)
                if len(all_selected_coreset_index) == total_samples:
                    break       
            expected_num_samples = int(total_samples * ratio)
            if len(all_selected_coreset_index) > expected_num_samples:
                excess = len(all_selected_coreset_index) - expected_num_samples
                all_selected_coreset_index = random.sample(all_selected_coreset_index, expected_num_samples)
            elif len(all_selected_coreset_index) < expected_num_samples:
                deficit = expected_num_samples - len(all_selected_coreset_index)
                remaining_samples = list(set(range(total_samples)) - set(all_selected_coreset_index))
                additional_samples = random.sample(remaining_samples, deficit)
                all_selected_coreset_index.extend(additional_samples)
        
            return all_selected_coreset_index
            
        label_all = score_clear
        
        def get_best_index():
            centerratio_all = [centerratio]  
            labelsigma_all =  [1]
            best_index = []
            best_auc = 0
            best_center = -1
            best_labelsigma = -1
            
            for j in labelsigma_all:
                for i in centerratio_all:
                    selected_index = get_index(centerratio=i)
                    selected_label = label_all[selected_index]
                    if featuremode=='nofinetune':
                        selected_feature = feature_all[selected_index]
                        feature_set = feature_all
                    elif featuremode=='finetune':
                        selected_feature = feature_all_finetune[selected_index]
                        feature_set = feature_all_finetune
                    ############
                    if cfg.coreset_mode=='window' or cfg.coreset_mode=='labelwindow_test':
                        w = cal_w_regression(selected_feature, selected_label)
                        outcome = check_acc_nat(w=w, featureset=feature_set, labelset=label_all)
                    # ###########
                    else:
                        model = Ridge(alpha=1.0)
                        model.fit(selected_feature, selected_label)
                        outcome = check_acc2_nat(featureset=feature_set,labelset=label_all,model=model)    
                    ##########          
                   # print(f"labelsigma={j}, centerratio={i}, outcome={outcome}")
                    if outcome > best_auc:
                        best_index = selected_index
                        best_auc = outcome
                        best_center = i
                        best_labelsigma = j 
            print(f"Best labelsigma: {best_labelsigma}, Best centerratio: {best_center}")
            return best_index
            
        all_index = get_best_index()
            
        return all_index
