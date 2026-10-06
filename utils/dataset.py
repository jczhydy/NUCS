import collections
import os
import os.path
import pprint
import random
import numpy as np
import pandas as pd
from typing import Dict, List
import skimage.transform
from skimage.io import imread
import torch
import torchxrayvision as xrv
from PIL import Image
import os
from torch.utils.data import Dataset
from torchvision import datasets, transforms
import torchvision
from torchvision.datasets import CIFAR10,CIFAR100
from datasets import load_dataset
import json
import torch.utils.data as data

def default_loader(path):
    return Image.open(path).convert('RGB')

class Food101Dataset(object):
    @staticmethod
    def get_food101_train(path, transform=None, identity_transform=False):
        if transform is None:
            mean = [0.485, 0.456, 0.406]  #
            std = [0.229, 0.224, 0.225]
            transform = transforms.Compose([
                transforms.RandomResizedCrop(224),  # 
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ])
        if identity_transform:
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
            transform = transforms.Compose([
                transforms.Resize(256),  
                transforms.CenterCrop(224),  
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ])
        
        trainset = torchvision.datasets.Food101(
            root=path,
            split='train',
            transform=transform,
            download=True
        )
        return trainset

    @staticmethod
    def get_food101_test(path):

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
        
        testset = torchvision.datasets.Food101(
            root=path,
            split='test',
            transform=transform_test,
            download=True
        )
        return testset
    
class INAT(data.Dataset):
    def __init__(self, root, ann_file, is_train=True):

        print('Loading annotations from: ' + os.path.basename(ann_file))
        with open(ann_file) as data_file:
            ann_data = json.load(data_file)

        self.imgs = [aa['file_name'] for aa in ann_data['images']]
        if 'annotations' in ann_data.keys():
            self.classes = [aa['category_id'] for aa in ann_data['annotations']]
        else:
            self.classes = [0] * len(self.imgs)

        print('\t' + str(len(self.imgs)) + ' images')
        print('\t' + str(len(set(self.classes))) + ' classes')

        self.root = root
        self.is_train = is_train
        self.loader = default_loader

        self.im_size = [299, 299]  # 
        self.mu_data = [0.485, 0.456, 0.406]  # 
        self.std_data = [0.229, 0.224, 0.225]  # 
        self.brightness = 0.4  # 
        self.contrast = 0.4  # 
        self.saturation = 0.4  # 
        self.hue = 0.25  

        self.center_crop = transforms.CenterCrop((224,224))
        self.scale_aug = transforms.RandomResizedCrop(224)
        self.flip_aug = transforms.RandomHorizontalFlip()
        self.resize_aug = transforms.Resize(224)
        self.color_aug = transforms.ColorJitter(self.brightness, self.contrast, self.saturation, self.hue)
        self.tensor_aug = transforms.ToTensor()
        self.norm_aug = transforms.Normalize(mean=self.mu_data, std=self.std_data)

    def __getitem__(self, index):

        path = os.path.join(self.root, self.imgs[index])  # 
        img = self.loader(path)  # 
        species_id = self.classes[index]  # 

        if self.is_train:
            img = self.scale_aug(img)
            img = self.flip_aug(img)
           # img = self.center_crop(img)
           # img = self.color_aug(img)
        else:
            img = self.resize_aug(img)
            img = self.center_crop(img)

        img = self.tensor_aug(img)
        img = self.norm_aug(img)

        return img, species_id  # 

    def __len__(self):

        return len(self.imgs)


class PlantNetDataLoader:
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
    
        return len(self.dataset)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):  # 
            idx = idx.item()  # 
        if isinstance(idx, np.integer):  # 
            idx = int(idx)  # 
        sample = self.dataset[idx]
        image = sample['image']
        label = sample['label']

        if self.transform is not None:
            image = self.transform(image)

        return image, label
    
class PlantNet300KLoader:
    @staticmethod
    def get_plantnet_train(path=None, transform=None, identity_transform=False):

        if transform is None:
            mean = [0.485, 0.456, 0.406]  # 
            std = [0.229, 0.224, 0.225]   # 
            transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),  
                transforms.Resize(224),
                transforms.RandomCrop(224),
                transforms.ToTensor(),            
                transforms.Normalize(mean=mean, std=std)  
            ])

        trainset = load_dataset("mikehemberger/plantnet300K", split="train", cache_dir=path)
        
        train_loader = PlantNetDataLoader(trainset, transform=transform)
        return train_loader

    @staticmethod
    def get_plantnet_test(path=None):

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        transform_test = transforms.Compose([
            transforms.Resize(224), 
            transforms.RandomCrop(224),
            transforms.ToTensor(),             
            transforms.Normalize(mean=mean, std=std)  
        ])

        testset = load_dataset("mikehemberger/plantnet300K", split="test", cache_dir=path)
        
        test_loader = PlantNetDataLoader(testset, transform=transform_test)
        return test_loader


def get_img_num_per_cls(cifar_version, imb_factor=None):

    cls_num = int(cifar_version)
    img_max = 50000/cls_num 
    if imb_factor is None:
        return [img_max] * cls_num
    img_num_per_cls = []
    for cls_idx in range(cls_num):
        num = img_max * (imb_factor**(cls_idx / (cls_num - 1.0)))
        img_num_per_cls.append(int(num))
    return img_num_per_cls

class ImbalancedCIFAR10(Dataset):
    def __init__(self, root, train=True, imb_factor=None, transform=None):
        self.dataset = torchvision.datasets.CIFAR10(root=root, train=train, download=True, transform=transform)
        # self.transform = transform
        self.classes = self.dataset.classes
        self.img_num_per_cls = get_img_num_per_cls('10', imb_factor)
        
        self.filtered_data = []
        self.filtered_labels = []
        class_counts = {i: 0 for i in range(10)}  # 
        
        for idx, (image, label) in enumerate(self.dataset):
            class_idx = label
            if class_counts[class_idx] < self.img_num_per_cls[class_idx]:
                self.filtered_data.append(image)
                self.filtered_labels.append(label)
                class_counts[class_idx] += 1
                
                if all(count >= self.img_num_per_cls[i] for i, count in class_counts.items()):
                    break
    
    def __len__(self):
        return len(self.filtered_data)

    def __getitem__(self, idx):
        image, label = self.filtered_data[idx], self.filtered_labels[idx]
        # if self.transform:
        #     image = self.transform(image)
        return image, label

class ImbalancedCIFAR100(Dataset):
    def __init__(self, root, train=True, imb_factor=None, transform=None):
        self.dataset = torchvision.datasets.CIFAR100(root=root, train=train, download=True, transform=transform)
        #self.transform = transform
        self.classes = self.dataset.classes
        self.img_num_per_cls = get_img_num_per_cls('100', imb_factor)
        
        self.filtered_data = []
        self.filtered_labels = []
        class_counts = {i: 0 for i in range(100)}  # 
        
        for idx, (image, label) in enumerate(self.dataset):
            class_idx = label
            if class_counts[class_idx] < self.img_num_per_cls[class_idx]:
                self.filtered_data.append(image)
                self.filtered_labels.append(label)
                class_counts[class_idx] += 1

                if all(count >= self.img_num_per_cls[i] for i, count in class_counts.items()):
                    break
    
    def __len__(self):
        return len(self.filtered_data)

    def __getitem__(self, idx):
        image, label = self.filtered_data[idx], self.filtered_labels[idx]
        # if self.transform:
        #     image = self.transform(image)
        return image, label
    
class CIFARDataset(object):
    @staticmethod
    def get_cifar10_transform(name):
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2470, 0.2435, 0.2616]
        if name == 'AutoAugment':
            policy = transforms.AutoAugmentPolicy.CIFAR10
            augmenter = transforms.AutoAugment(policy)
        elif name == 'RandAugment':
            augmenter = transforms.RandAugment()
        elif name == 'AugMix':
            augmenter = transforms.AugMix()
        else: raise f"Unknown augmentation method: {name}!"

        transform = transforms.Compose([
            augmenter,
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

        return transform

    @staticmethod
    def get_cifar10_train(path, transform=None, identity_transform=False):
        if transform is None:
            mean = [0.4914, 0.4822, 0.4465]
            std = [0.2470, 0.2435, 0.2616]
            transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
                transforms.RandomHorizontalFlip(),
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ])
        if identity_transform:
            mean = [0.4914, 0.4822, 0.4465]
            std = [0.2470, 0.2435, 0.2616]
            transform = transforms.Compose([
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ])
        trainset = torchvision.datasets.CIFAR10(root=path, train=True, download=True, transform=transform)
        return trainset

    @staticmethod
    def get_cifar10_test(path):
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2470, 0.2435, 0.2616]
        transform_test = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
        testset = torchvision.datasets.CIFAR10(root=path, train=False, download=True, transform=transform_test)
        return testset
    
    @staticmethod
    def get_imbalanced_cifar10_train(path, imb_factor=None, transform=None, identity_transform=False):
        if transform is None:
            mean = [0.4914, 0.4822, 0.4465]
            std = [0.2470, 0.2435, 0.2616]
            transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
                transforms.RandomHorizontalFlip(),
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ])
        if identity_transform:
            mean = [0.4914, 0.4822, 0.4465]
            std = [0.2470, 0.2435, 0.2616]
            transform = transforms.Compose([
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ])

        trainset = ImbalancedCIFAR10(root=path, train=True, imb_factor=imb_factor, transform=transform)
        return trainset

    @staticmethod
    def get_imbalanced_cifar10_test(path, transform=None):
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2470, 0.2435, 0.2616]
        if transform is None:
            transform = transforms.Compose([
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ])

        testset = CIFAR10(root=path, train=False, download=True, transform=transform)
        return testset
    
    @staticmethod
    def get_cifar100_train(path, transform=None, identity_transform=False):
        if transform is None:
            mean=[0.507, 0.487, 0.441]
            std=[0.267, 0.256, 0.276]
            transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
                transforms.RandomHorizontalFlip(),
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ])
        if identity_transform:
            mean=[0.507, 0.487, 0.441]
            std=[0.267, 0.256, 0.276]
            transform = transforms.Compose([
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ])
        trainset = torchvision.datasets.CIFAR100(root=path, train=True, download=True, transform=transform)
        return trainset

    @staticmethod
    def get_imbalanced_cifar100_train(path, imb_factor=None, transform=None, identity_transform=False):
        if transform is None:
            mean = [0.507, 0.487, 0.441]
            std = [0.267, 0.256, 0.276]
            transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
                transforms.RandomHorizontalFlip(),
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ])
        if identity_transform:
            mean = [0.507, 0.487, 0.441]
            std = [0.267, 0.256, 0.276]
            transform = transforms.Compose([
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ])
        trainset = ImbalancedCIFAR100(root=path, train=True, imb_factor=imb_factor, transform=transform)
        return trainset

    @staticmethod
    def get_cifar100_test(path):
        mean = [0.507, 0.487, 0.441]
        std = [0.267, 0.256, 0.276]
        transform_test = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
        testset = CIFAR100(root=path, train=False, download=True, transform=transform_test)
        return testset
    
    @staticmethod
    def get_imbalanced_cifar100_test(path):
        mean = [0.507, 0.487, 0.441]
        std = [0.267, 0.256, 0.276]
        transform_test = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
        testset = CIFAR100(root=path, train=False, download=True, transform=transform_test)
        return testset

class INaturalistDataset2021(object):
    @staticmethod
    def get_inaturalist_train(path, transform=None):
        if transform is None:
            transform = transforms.Compose([
                transforms.Resize(224),
                transforms.RandomCrop(224),
                transforms.ToTensor(),
            ])
        trainset = torchvision.datasets.INaturalist(
            root=path,
            version='2021_train_mini',
            target_type=['full'],
            transform=transform,
            download=False
        )
        return trainset

    @staticmethod
    def get_inaturalist_valid(path, transform=None):
        if transform is None:
            transform = transforms.Compose([
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
            ])
        validset = torchvision.datasets.INaturalist(
            root=path,
            version='2021_valid',
            target_type=['full'],
            transform=transform,
            download=False 
        )
        return validset
    
class SVHNDataset(object):
    @staticmethod
    def get_svhn_train(path, transform=None):
        if transform is None:
            transform = transforms.Compose([
                transforms.ToTensor(),
            ])
        trainset = torchvision.datasets.SVHN(root=path, split='train', download=True, transform=transform)
        return trainset

    @staticmethod
    def get_svhn_test(path):
        transform_test = transforms.Compose([
            transforms.ToTensor(),
        ])
        testset = torchvision.datasets.SVHN(root=path, split='test', download=True, transform=transform_test)
        return testset

class CINIC10Dataset(object):
    @staticmethod
    def get_cinic10_train(path, transform=None, identity_transform=False):
        if transform is None:
            mean = [0.47889522, 0.47227842, 0.43047404]
            std = [0.24205776, 0.23828046, 0.25874835]
            transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ])
        if identity_transform:
            mean = [0.47889522, 0.47227842, 0.43047404]
            std = [0.24205776, 0.23828046, 0.25874835]
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ])
        path = os.path.join(path, 'train')
        trainset = torchvision.datasets.ImageFolder(root=path, transform=transform)
        return trainset

    @staticmethod
    def get_cinic10_test(path):
        mean = [0.47889522, 0.47227842, 0.43047404]
        std = [0.24205776, 0.23828046, 0.25874835]
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
        path = os.path.join(path, 'test')
        testset = torchvision.datasets.ImageFolder(root=path, transform=transform_test)
        return testset

class ImageNetDataset(object):
    @staticmethod
    def get_ImageNet_train(path, transform=None):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
        trainset = datasets.ImageFolder(
            path,
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                # transforms.ColorJitter(
                #     brightness=0.4,
                #     contrast=0.4,
                #     saturation=0.4),
                transforms.ToTensor(),
                normalize,
            ]))


        return trainset

    @staticmethod
    def get_ImageNet_test(path):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
        testset = datasets.ImageFolder(
            path,
            transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
        ]))
        return testset
    
default_pathologies = [
    'Atelectasis',
    'Consolidation',
    'Infiltration',
    'Pneumothorax',
    'Edema',
    'Emphysema',
    'Fibrosis',
    'Effusion',
    'Pneumonia',
    'Pleural_Thickening',
    'Cardiomegaly',
    'Nodule',
    'Mass',
    'Hernia',
    'Lung Lesion',
    'Fracture',
    'Lung Opacity',
    'Enlarged Cardiomediastinum'
]
USE_INCLUDED_FILE = "USE_INCLUDED_FILE"

thispath = os.path.dirname(os.path.realpath(__file__))
datapath = os.path.join(thispath, "data")

def normalize(img, maxval, reshape=False):
    """Scales images to be roughly [-1024 1024].

    Call xrv.utils.normalize moving forward.
    """
    return xrv.utils.normalize(img, maxval, reshape)

def apply_transforms(sample, transform, seed=None) -> Dict:
    """Applies transforms to the image and masks.
    The seeds are set so that the transforms that are applied
    to the image are the same that are applied to each mask.
    This way data augmentation will work for segmentation or 
    other tasks which use masks information.
    """

    if seed is None:
        MAX_RAND_VAL = 2147483647
        seed = np.random.randint(MAX_RAND_VAL)

    if transform is not None:
        random.seed(seed)
        torch.random.manual_seed(seed)
        sample["img"] = transform(sample["img"])

        if "pathology_masks" in sample:
            for i in sample["pathology_masks"].keys():
                random.seed(seed)
                torch.random.manual_seed(seed)
                sample["pathology_masks"][i] = transform(sample["pathology_masks"][i])

        if "semantic_masks" in sample:
            for i in sample["semantic_masks"].keys():
                random.seed(seed)
                torch.random.manual_seed(seed)
                sample["semantic_masks"][i] = transform(sample["semantic_masks"][i])

    return sample

class Dataset:
    def __init__(self):
        pass

    pathologies: List[str]
    labels: np.ndarray
    csv: pd.DataFrame
    def totals(self) -> Dict[str, Dict[str, int]]:

        counts = [dict(collections.Counter(items[~np.isnan(items)]).most_common()) for items in self.labels.T]
        return dict(zip(self.pathologies, counts))

    def __repr__(self) -> str:

        if xrv.utils.in_notebook():
            pprint.pprint(self.totals())
        return self.string()

    def check_paths_exist(self):
        if not os.path.isdir(self.imgpath):
            raise Exception("imgpath must be a directory")
        if not os.path.isfile(self.csvpath):
            raise Exception("csvpath must be a file")

    def limit_to_selected_views(self, views):
        """This function is called by subclasses to filter the
        images by view based on the values in .csv['view']
        """
        if type(views) is not list:
            views = [views]
        if '*' in views:
            # if you have the wildcard, the rest are irrelevant
            views = ["*"]
        self.views = views

        # missing data is unknown
        self.csv.view.fillna("UNKNOWN", inplace=True)

        if "*" not in views:
            self.csv = self.csv[self.csv["view"].isin(self.views)]  # Select the view

class SubsetDataset(Dataset):
    
    def __init__(self, dataset, idxs=None, data_aug=None):
        super(SubsetDataset, self).__init__()
        self.dataset = dataset
        self.pathologies = dataset.pathologies

        self.idxs = idxs
        self.labels = self.dataset.labels[self.idxs]
        self.csv = self.dataset.csv.iloc[self.idxs]
        self.csv = self.csv.reset_index(drop=True)

        if hasattr(self.dataset, 'which_dataset'):
            # keep information about the source dataset from a merged dataset
            self.which_dataset = self.dataset.which_dataset[self.idxs]
        
        # Apply data augmentation.
        self.data_aug = data_aug

    def string(self):
        return self.__class__.__name__ + " num_samples={}\n".format(len(self)) + "└ of " + self.dataset.string().replace("\n", "\n  ")

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, idx):
        # Retrieve the original sample.
        sample = self.dataset[self.idxs[idx]]
        if self.data_aug:
            sample['img'] = self.data_aug(sample['img'])
        
        return sample


class RSNA_Pneumonia_Dataset(Dataset):
    """RSNA Pneumonia Detection Challenge

    Citation:

    Augmenting the National Institutes of Health Chest Radiograph Dataset
    with Expert Annotations of Possible Pneumonia. Shih, George, Wu,
    Carol C., Halabi, Safwan S., Kohli, Marc D., Prevedello, Luciano M.,
    Cook, Tessa S., Sharma, Arjun, Amorosa, Judith K., Arteaga, Veronica,
    Galperin-Aizenberg, Maya, Gill, Ritu R., Godoy, Myrna C.B., Hobbs,
    Stephen, Jeudy, Jean, Laroia, Archana, Shah, Palmi N., Vummidi, Dharshan,
    Yaddanapudi, Kavitha, and Stein, Anouk. Radiology: Artificial
    Intelligence, 1 2019. doi: 10.1148/ryai.2019180041.

    More info: https://www.rsna.org/en/education/ai-resources-and-training/ai-image-challenge/RSNA-Pneumonia-Detection-Challenge-2018

    Challenge site:
    https://www.kaggle.com/c/rsna-pneumonia-detection-challenge

    JPG files stored here:
    https://academictorrents.com/details/95588a735c9ae4d123f3ca408e56570409bcf2a9
    """

    def __init__(self,
                 imgpath,
                 csvpath=USE_INCLUDED_FILE,
                 dicomcsvpath=USE_INCLUDED_FILE,
                 views=["PA"],
                 transform=None,
                 data_aug=None,
                 nrows=None,
                 seed=0,
                 pathology_masks=False,
                 extension=".jpg"
                 ):

        super(RSNA_Pneumonia_Dataset, self).__init__()
        np.random.seed(seed)  # Reset the seed so all runs are the same.
        self.imgpath = imgpath
        self.transform = transform
        self.data_aug = data_aug
        self.pathology_masks = pathology_masks

        self.pathologies = ["Pneumonia", "No Finding"]

        #self.pathologies = sorted(self.pathologies)

        self.extension = extension
        self.use_pydicom = (extension == ".dcm")

        # # Load data
        if csvpath == USE_INCLUDED_FILE:
            self.csvpath = os.path.join(datapath, "kaggle_stage_2_train_labels.csv.zip")
        else:
            self.csvpath = csvpath
        # self.raw_csv = pd.read_csv(self.csvpath, nrows=nrows)

        # # The labels have multiple instances for each mask
        # # So we just need one to get the target label
        # self.csv = self.raw_csv.groupby("patientId").first()

        # if dicomcsvpath == USE_INCLUDED_FILE:
        #     self.dicomcsvpath = os.path.join(datapath, "kaggle_stage_2_train_images_dicom_headers.csv.gz")
        # else:
        #     self.dicomcsvpath = dicomcsvpath

        #self.dicomcsv = pd.read_csv(self.dicomcsvpath, nrows=nrows, index_col="patientid")

        #self.csv = self.csv.join(self.dicomcsv, on="patientId")
        self.csv = pd.read_csv(self.csvpath)
        # Remove images with view position other than specified
       # self.csv["view"] = self.csv['ViewPosition']
        #self.limit_to_selected_views(views)

        self.csv = self.csv.reset_index()

        # Get our classes.
        #print(self.csv["Target"].values)
        #labels = [self.csv["Target"].values, self.csv["Target"].values]
        labels = []
        for target in self.csv["Target"].values:
            if target == 1:
                labels.append([1, 0])  # Pneumonia
            else:
                labels.append([0, 1])  # Normal

        self.labels = np.array(labels).astype(np.float32)
        # set if we have masks
        #self.csv["has_masks"] = ~np.isnan(self.csv["x"])

        #self.labels = np.asarray(labels).T
        self.labels = self.labels.astype(np.float32)
        # patientid
        self.csv["patientid"] = self.csv["patientId"].astype(str)

    def string(self):
        return self.__class__.__name__ + " num_samples={} data_aug={}".format(len(self), self.data_aug)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sample = {}
        sample["idx"] = idx
        sample["lab"] = self.labels[idx]

        imgid = self.csv['patientId'].iloc[idx]
        img_path = os.path.join(self.imgpath, imgid + self.extension)

        if self.use_pydicom:
            try:
                import pydicom
            except ImportError as e:
                raise Exception("Please install pydicom to work with this dataset")

            img = pydicom.filereader.dcmread(img_path).pixel_array
        else:
            img = imread(img_path)

        sample["img"] = normalize(img, maxval=255, reshape=True)

        # if self.pathology_masks:
        #     sample["pathology_masks"] = self.get_mask_dict(imgid, sample["img"].shape[2])

        sample = apply_transforms(sample, self.transform)
        sample = apply_transforms(sample, self.data_aug)

        return sample


       
class NIH_Dataset(Dataset):

    def __init__(self,
                imgpath,
                csvpath=USE_INCLUDED_FILE,
                bbox_list_path=USE_INCLUDED_FILE,
                views=["PA"],
                transform=None,
                data_aug=None,
                seed=0,
                unique_patients=True,
                pathology_masks=False
                ):
        super(NIH_Dataset, self).__init__()

        np.random.seed(seed)  # Reset the seed so all runs are the same.
        self.imgpath = imgpath

        if csvpath == USE_INCLUDED_FILE:
            self.csvpath = os.path.join(datapath, "Data_Entry_2017_v2020.csv.gz")
        else:
            self.csvpath = csvpath

        self.transform = transform
        self.data_aug = data_aug
        self.pathology_masks = pathology_masks

        self.pathologies = ["Atelectasis", "Consolidation", "Infiltration",
                            "Pneumothorax", "Edema", "Emphysema", "Fibrosis",
                            "Effusion", "Pneumonia", "Pleural_Thickening",
                            "Cardiomegaly", "Nodule", "Mass", "Hernia"]

        self.pathologies = sorted(self.pathologies)

        # Load data
        self.check_paths_exist()
        self.csv = pd.read_csv(self.csvpath)

        # Remove images with view position other than specified
        self.csv["view"] = self.csv['View Position']
        self.limit_to_selected_views(views)

        if unique_patients:
            self.csv = self.csv.groupby("Patient ID").first()

        self.csv = self.csv.reset_index()
        ####### pathology masks ########
        # Get our classes.
        self.labels = []
        for pathology in self.pathologies:
            self.labels.append(self.csv["Finding Labels"].str.contains(pathology).values)

        self.labels = np.asarray(self.labels).T
        self.labels = self.labels.astype(np.float32)

        # add consistent csv values

        # offset_day_int
        # self.csv["offset_day_int"] =

        # patientid
        self.csv["patientid"] = self.csv["Patient ID"].astype(str)

        # age
        # self.csv['age_years'] = self.csv['Patient Age'] * 1.0

        # sex
        self.csv['sex_male'] = self.csv['Patient Gender'] == 'M'
        self.csv['sex_female'] = self.csv['Patient Gender'] == 'F'

    def string(self):
        return self.__class__.__name__ + " num_samples={} views={} data_aug={}".format(len(self), self.views, self.data_aug)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sample = {}
        sample["idx"] = idx
        sample["lab"] = self.labels[idx]

        imgid = self.csv['Image Index'].iloc[idx]
        img_path = os.path.join(self.imgpath, imgid)
        
        # Read the image with PIL or OpenCV.
        img = Image.open(img_path).convert("RGB")  # Convert the image to RGB.
        img = np.array(img)  # Convert the image to a NumPy array.

        sample["img"] = normalize(img, maxval=255, reshape=True)

        # if self.pathology_masks:
        #     sample["pathology_masks"] = self.get_mask_dict(imgid, sample["img"].shape[2])

        sample = apply_transforms(sample, self.transform)
        sample = apply_transforms(sample, self.data_aug)

        return sample

class NIH_Dataset_2(Dataset):

    def __init__(self,
                imgpath,
                csvpath=USE_INCLUDED_FILE,
                bbox_list_path=USE_INCLUDED_FILE,
                views=["PA"],
                transform=None,
                data_aug=None,
                seed=0,
                unique_patients=True,
                pathology_masks=False
                ):
        super(NIH_Dataset_2, self).__init__()

        np.random.seed(seed)  # Reset the seed so all runs are the same.
        self.imgpath = imgpath

        if csvpath == USE_INCLUDED_FILE:
            self.csvpath = os.path.join(datapath, "Data_Entry_2017_v2020.csv.gz")
        else:
            self.csvpath = csvpath

        self.transform = transform
        self.data_aug = data_aug
        self.pathology_masks = pathology_masks

        self.pathologies = ["Atelectasis", "Consolidation", "Infiltration",
                            "Pneumothorax", "Edema", "Emphysema", "Fibrosis",
                            "Effusion", "Pneumonia", "Pleural_Thickening",
                            "Cardiomegaly", "Nodule", "Mass", "Hernia", "No Finding"]  #,"No Finding"

        self.pathologies = sorted(self.pathologies)

        # Load data
        self.check_paths_exist()
        self.csv = pd.read_csv(self.csvpath)

        # Remove images with view position other than specified
        self.csv["view"] = self.csv['View Position']
        self.limit_to_selected_views(views)

        if unique_patients:
            self.csv = self.csv.groupby("Patient ID").first()

        self.csv = self.csv.reset_index()
        ####### pathology masks ########
        # Get our classes.
        self.labels = []
        for pathology in self.pathologies:
            self.labels.append(self.csv["Finding Labels"].str.contains(pathology).values)

        self.labels = np.asarray(self.labels).T
        self.labels = self.labels.astype(np.float32)

        # Add new label: 1 if all others are 0, else 0
        # new_label = (np.sum(self.labels, axis=1) == 0).astype(np.float32)
        # self.labels = np.hstack([self.labels, new_label[:, np.newaxis]])
        
        self.csv["patientid"] = self.csv["Patient ID"].astype(str)

    def string(self):
        return self.__class__.__name__ + " num_samples={} views={} data_aug={}".format(len(self), self.views, self.data_aug)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sample = {}
        sample["idx"] = idx
        sample["lab"] = self.labels[idx]

        imgid = self.csv['Image Index'].iloc[idx]
        img_path = os.path.join(self.imgpath, imgid)
        
        # 
        img = Image.open(img_path).convert("RGB")  # 
        img = np.array(img)  # 

        sample["img"] = normalize(img, maxval=255, reshape=True)
        sample = apply_transforms(sample, self.transform)
        sample = apply_transforms(sample, self.data_aug)

        return sample

class CheX_Dataset(Dataset):

    def __init__(self,
                 imgpath,
                 csvpath=USE_INCLUDED_FILE,
                 basedir=None,
                 views=["PA"],
                 transform=None,
                 data_aug=None,
                 flat_dir=True,
                 seed=0,
                 unique_patients=True
                 ):

        super(CheX_Dataset, self).__init__()
        np.random.seed(seed)  # Reset the seed so all runs are the same.

        self.pathologies = ["Enlarged Cardiomediastinum",
                            "Cardiomegaly",
                            "Lung Opacity",
                            "Lung Lesion",
                            "Edema",
                            "Consolidation",
                            "Pneumonia",
                            "Atelectasis",
                            "Pneumothorax",
                            "Pleural Effusion",
                            "Pleural Other",
                            "Fracture",
                            # "Support Devices",
                            "No Finding"]

        self.pathologies = sorted(self.pathologies)

        self.imgpath = imgpath
        self.basedir = basedir if basedir is not None else imgpath
        self.transform = transform
        self.data_aug = data_aug
        if csvpath == USE_INCLUDED_FILE:
            self.csvpath = os.path.join(datapath, "chexpert_train.csv.gz")
        else:
            self.csvpath = csvpath
        self.csv = pd.read_csv(self.csvpath)
        self.views = views

        self.csv["view"] = self.csv["Frontal/Lateral"]  # Assign view column
        self.csv.loc[(self.csv["view"] == "Frontal"), "view"] = self.csv["AP/PA"]  # If Frontal change with the corresponding value in the AP/PA column otherwise remains Lateral
        self.csv["view"] = self.csv["view"].replace({'Lateral': "L"})  # Rename Lateral with L

        self.limit_to_selected_views(views)

        if unique_patients:
            self.csv["PatientID"] = self.csv["Path"].str.extract(pat=r'(patient\d+)')
            self.csv = self.csv.groupby("PatientID").first().reset_index()

        # Get our classes.
        # healthy = self.csv["No Finding"] == 1
        labels = []
        for pathology in self.pathologies:
            if pathology in self.csv.columns:
                mask = self.csv[pathology]
                healthy = self.csv["No Finding"] == 1
                if pathology != "No Finding": 
                    self.csv.loc[healthy, pathology] = 0
                
                labels.append(mask.values)
        self.labels = np.asarray(labels).T
        self.labels = self.labels.astype(np.float32)

        # Make all the -1 values into nans to keep things simple
        self.labels[self.labels == -1] = 1

        self.labels[np.isnan(self.labels)] = 0

        # Rename pathologies
        self.pathologies = list(np.char.replace(self.pathologies, "Pleural Effusion", "Effusion"))

        if 'train' in self.csvpath:
            patientid = self.csv.Path.str.split("train/", expand=True)[1]
        elif 'valid' in self.csvpath:
            patientid = self.csv.Path.str.split("valid/", expand=True)[1]
        else:
            raise NotImplementedError

        patientid = patientid.str.split("/study", expand=True)[0]
        patientid = patientid.str.replace("patient", "")

        # patientid
        self.csv["patientid"] = patientid

    def string(self):
        return self.__class__.__name__ + " num_samples={} views={} data_aug={}".format(len(self), self.views, self.data_aug)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sample = {}
        sample["idx"] = idx
        sample["lab"] = self.labels[idx]

        imgid = self.csv['Path'].iloc[idx]
        # print(imgid)
        # print(os.path)
        # clean up path in csv so the user can specify the path
        #imgid = imgid.replace("CheXpert-v1.0-small/", "").replace("CheXpert-v1.0/", "")
        img_path = os.path.join(self.basedir, imgid)
        img = imread(img_path)

        sample["img"] = normalize(img, maxval=255, reshape=True)

        sample = apply_transforms(sample, self.transform)
        sample = apply_transforms(sample, self.data_aug)

        return sample
    

class PC_Dataset(Dataset):
    """PadChest dataset from the Hospital San Juan de Alicante - University of
    Alicante

    Note that images with null labels (as opposed to normal), and images that
    cannot be properly loaded (listed as 'missing' in the code) are excluded,
    which makes the total number of available images slightly less than the
    total number of image files.

    Citation:

    PadChest: A large chest x-ray image dataset with multi-label annotated
    reports. Aurelia Bustos, Antonio Pertusa, Jose-Maria Salinas, and Maria
    de la Iglesia-Vayá. arXiv preprint, 2019. https://arxiv.org/abs/1901.07441

    Dataset website:
    http://bimcv.cipf.es/bimcv-projects/padchest/

    Download full size images here:
    https://academictorrents.com/details/dec12db21d57e158f78621f06dcbe78248d14850

    Download resized (224x224) images here (recropped):
    https://academictorrents.com/details/96ebb4f92b85929eadfb16761f310a6d04105797
    """

    def __init__(self,
                 imgpath,
                 csvpath=USE_INCLUDED_FILE,
                 views=["PA"],
                 transform=None,
                 data_aug=None,
                 flat_dir=True,
                 seed=0,
                 unique_patients=True
                 ):

        super(PC_Dataset, self).__init__()
        np.random.seed(seed)  # Reset the seed so all runs are the same.

        self.pathologies = ["Atelectasis", "Consolidation", "Infiltration",
                            "Pneumothorax", "Edema", "Emphysema", "Fibrosis",
                            "Effusion", "Pneumonia", "Pleural_Thickening",
                            "Cardiomegaly", "Nodule", "Mass", "Hernia", "Fracture",
                            "Granuloma", "Flattened Diaphragm", "Bronchiectasis",
                            "Aortic Elongation", "Scoliosis",
                            "Hilar Enlargement", "Tuberculosis",
                            "Air Trapping", "Costophrenic Angle Blunting", "Aortic Atheromatosis",
                            "Hemidiaphragm Elevation",
                             "Tube'",'normal']  
#         self.pathologies = ['Atelectasis','Consolidation','Infiltration','Pneumothorax','Edema','Emphysema',
#     'Fibrosis','Effusion','Pneumonia','Pleural_Thickening','Cardiomegaly','Nodule','Mass','Hernia',
#     'Fracture', 'normal',"No Finding"
# ]

        self.pathologies = sorted(self.pathologies)

        mapping = dict()

        mapping["Infiltration"] = ["infiltrates",
                                   "interstitial pattern",
                                   "ground glass pattern",
                                   "reticular interstitial pattern",
                                   "reticulonodular interstitial pattern",
                                   "alveolar pattern",
                                   "consolidation",
                                   "air bronchogram"]
        mapping["Pleural_Thickening"] = ["pleural thickening"]
        mapping["Consolidation"] = ["air bronchogram"]
        mapping["Hilar Enlargement"] = ["adenopathy",
                                        "pulmonary artery enlargement"]
        mapping["Support Devices"] = ["device",
                                      "pacemaker"]
        mapping["Tube'"] = ["stent'"]  

        self.imgpath = imgpath
        self.transform = transform
        self.data_aug = data_aug
        self.flat_dir = flat_dir
        if csvpath == USE_INCLUDED_FILE:
            self.csvpath = os.path.join(datapath, "PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv.gz")
        else:
            self.csvpath = csvpath

        self.check_paths_exist()
        self.csv = pd.read_csv(self.csvpath, low_memory=False)

        # Standardize view names
        self.csv.loc[self.csv["Projection"].isin(["AP_horizontal"]), "Projection"] = "AP Supine"

        self.csv["view"] = self.csv['Projection']
        self.limit_to_selected_views(views)

        # Remove null stuff
        self.csv = self.csv[~self.csv["Labels"].isnull()]

        # Remove missing files
        missing = ["216840111366964012819207061112010307142602253_04-014-084.png",
                   "216840111366964012989926673512011074122523403_00-163-058.png",
                   "216840111366964012959786098432011033083840143_00-176-115.png",
                   "216840111366964012558082906712009327122220177_00-102-064.png",
                   "216840111366964012339356563862009072111404053_00-043-192.png",
                   "216840111366964013076187734852011291090445391_00-196-188.png",
                   "216840111366964012373310883942009117084022290_00-064-025.png",
                   "216840111366964012283393834152009033102258826_00-059-087.png",
                   "216840111366964012373310883942009170084120009_00-097-074.png",
                   "216840111366964012819207061112010315104455352_04-024-184.png",
                   "216840111366964012819207061112010306085429121_04-020-102.png",
                   "216840111366964012989926673512011083134050913_00-168-009.png",  # broken PNG file (chunk b'\x00\x00\x00\x00')
                   "216840111366964012373310883942009152114636712_00-102-045.png",  # "OSError: image file is truncated"
                   "216840111366964012819207061112010281134410801_00-129-131.png",  # "OSError: image file is truncated"
                   "216840111366964012487858717522009280135853083_00-075-001.png",  # "OSError: image file is truncated"
                   "216840111366964012989926673512011151082430686_00-157-045.png",  # broken PNG file (chunk b'\x00\x00\x00\x00')
                   "216840111366964013686042548532013208193054515_02-026-007.png",  # "OSError: image file is truncated"
                   "216840111366964013590140476722013058110301622_02-056-111.png",  # "OSError: image file is truncated"
                   "216840111366964013590140476722013043111952381_02-065-198.png",  # "OSError: image file is truncated"
                   "216840111366964013829543166512013353113303615_02-092-190.png",  # "OSError: image file is truncated"
                   "216840111366964013962490064942014134093945580_01-178-104.png",  # "OSError: image file is truncated"
                   ]
        self.csv = self.csv[~self.csv["ImageID"].isin(missing)]

        if unique_patients:
            self.csv = self.csv.groupby("PatientID").first().reset_index()

        # Filter out age < 10 (paper published 2019)
        self.csv = self.csv[(2019 - self.csv.PatientBirth > 10)]

        # Get our classes.
        labels = []
        for pathology in self.pathologies:
            mask = self.csv["Labels"].str.contains(pathology.lower())
            if pathology in mapping:
                for syn in mapping[pathology]:
                    # print("mapping", syn)
                    mask |= self.csv["Labels"].str.contains(syn.lower())
            labels.append(mask.values)
        self.labels = np.asarray(labels).T
        self.labels = self.labels.astype(np.float32)
        ############
        # no_finding_index = self.pathologies.index("No Finding")
        # all_zeros_mask = np.all(self.labels == 0, axis=1)  # 
        # self.labels[all_zeros_mask, no_finding_index] = 1  # 
        #self.pathologies[self.pathologies.index("Tube'")] = "Tube"
        # offset_day_int
        # dt = pd.to_datetime(self.csv["StudyDate_DICOM"], format="%Y%m%d")
        # self.csv["offset_day_int"] = dt.astype(np.int64) // 10**9 // 86400

        # patientid
        self.csv["patientid"] = self.csv["PatientID"].astype(str)

        # age
        # self.csv['age_years'] = (2017 - self.csv['PatientBirth'])

        # # sex
        # self.csv['sex_male'] = self.csv['PatientSex_DICOM'] == 'M'
        # self.csv['sex_female'] = self.csv['PatientSex_DICOM'] == 'F'

    def string(self):
        return self.__class__.__name__ + " num_samples={} views={} data_aug={}".format(len(self), self.views, self.data_aug)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sample = {}
        sample["idx"] = idx
        sample["lab"] = self.labels[idx]
        ###########
        # if np.all(sample["lab"] == 0):
        #     no_finding_index = self.pathologies.index("No Finding")
        #     sample["lab"][no_finding_index] = 1
            
        imgid = self.csv['ImageID'].iloc[idx]
        img_path = os.path.join(self.imgpath, imgid)
        img = imread(img_path)

        sample["img"] = normalize(img, maxval=65535, reshape=True)

        sample = apply_transforms(sample, self.transform)
        sample = apply_transforms(sample, self.data_aug)

        return sample
