import os
import copy
import collections
from glob import glob
from tqdm import tqdm

import torch
from torchvision import transforms
from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader

import utils

def get_pretrain_config(dataset_name):
    config = {
        # 'stl_10/unlabeled': {'img_size': 96, 'in_channels': 3, 'batch_size': 256},
        'cifar_10/train': {
            'img_size': 32,  'in_channels': 3,  'batch_size': 256, 'patch_size': 2,
            'emb_dim': 192, 'mask_ratio': 0.75, 'en_num_heads': 3, 'en_depth': 12, 
            'de_emb_dim': None,  'de_num_heads': 3, 'de_depth': 4, 
            'use_norm_pix_loss': True,
            'base_lr': 1.5e-4, 'weight_decay': 0.05, 'num_epochs': 2000, 'warmup_epochs': 200,
            'save_interval':500,
        },
    }
    
    return config[dataset_name]


def get_train_config(dataset_name):
    config = {
        'cifar_10': {
            'img_size': 32,  'in_channels': 3,  'batch_size': 128, 'patch_size': 2,
            'emb_dim': 192, 'mask_ratio': 0.0, 'en_num_heads': 3, 'en_depth': 12, 
            'de_emb_dim': None,  'de_num_heads': 3, 'de_depth': 4, 
            'use_norm_pix_loss': True,
            'base_lr': 1e-3, 'weight_decay': 0.05, 'num_epochs': 30, 'warmup_epochs': 5,
            'save_interval': 50,
        },
    }
    
    return config[dataset_name]


def get_pretrain_transform(img_size):
    tf = list()
    tf.extend([
            transforms.ToTensor(), 
            transforms.Resize(img_size),
            transforms.Normalize([0.5], [0.5]),
    ])
    return transforms.Compose(tf)
    

def get_transform(dataset_name, train_set=False, img_size=32):
    """Given dataset name, we get the corresponding transform."""

    tf = list()
    if train_set and dataset_name != 'mnist' and dataset_name != 'svhn' and dataset_name != 'chars74k_fnt_num':
        tf.append(transforms.RandomHorizontalFlip())

    tf.extend([
            transforms.ToTensor(), 
            transforms.Resize(img_size),
            transforms.Normalize([0.5], [0.5]),
    ])
    return transforms.Compose(tf)


# This can handle empty samples in a folder.
class ImgDataset(Dataset):
    def __init__(self, parent_dir, label_idx_dict=None, transform=None):
        self.img_list = []
        self.label_list = []
        self.label_idx_dict = label_idx_dict
        
        sub_dirs = [f.name for f in os.scandir(parent_dir) if f.is_dir()]
        sub_dirs.sort()
        if self.label_idx_dict is None:
            self.label_idx_dict = {label:idx for idx, label in enumerate(sub_dirs)}

        self.classes = self.label_idx_dict.keys()  # To show what classes are in the dataset.
            
        for sub_dir in sub_dirs:
            full_path = os.path.join(parent_dir, sub_dir)
            file_extensions = ['*.JPG', '*.JPEG', '*.jpg', '*.png', '*.PNG']
            
            # img_paths = glob(os.path.join(full_path, '*.JPG')) + glob(os.path.join(full_path, '*.jpg'))
            
            img_paths = []
            for extension in file_extensions:
                img_paths.extend(glob(os.path.join(full_path, extension)))
            img_paths.sort()

            labels = [self.label_idx_dict[sub_dir]] * len(img_paths)
            self.img_list += img_paths
            self.label_list += labels
            
        self.transform = transform
        
    def __len__(self):
        return len(self.label_list)
    
    def __getitem__(self, idx):
        img_path = self.img_list[idx]
        image = default_loader(img_path)
        label = self.label_list[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

    
# This can handle empty samples in a folder.
class UnlabeledImgDataset(Dataset):
    def __init__(self, parent_dir, transform=None):
        self.img_list = []
        self.transform = transform
        file_extensions = ['*.png', '*.PNG', '*.JPG', '*.JPEG', '*.jpg']
        
        sub_dirs = [f.name for f in os.scandir(parent_dir) if f.is_dir()]
        sub_dirs.sort()
        
        if len(sub_dirs) > 0:                # For getting unlabeled data from labeled datasets. e.g., CIFAR-10.
            for sub_dir in sub_dirs:
                for extension in file_extensions:
                    self.img_list.extend(glob(os.path.join(parent_dir, sub_dir, extension)))
        else:                                # For getting unlabeled data. e.g., STL-10 unlabeled protion.
            for extension in file_extensions:
                    self.img_list.extend(glob(os.path.join(parent_dir, extension)))
                    
        self.img_list.sort()
        
    def __len__(self):
        return len(self.img_list)
    
    def __getitem__(self, idx):
        img_path = self.img_list[idx]
        image = default_loader(img_path)
        return self.transform(image)


@torch.no_grad()
def evaluate_model(model, data_loader, tqdm_desc=None,):
    device = next(model.parameters()).device

    loss_metric = utils.MeanMetric()
    acc_metric = utils.MeanMetric()

    loss_ce = torch.nn.CrossEntropyLoss()

    with utils.eval_mode(model):
        for (x, y) in tqdm(data_loader, desc=tqdm_desc):
            x = x.to(device)
            y = y.to(device)
            y_pred = model(x)
            loss = loss_ce(y_pred, y)
            
            loss_metric.update_state(loss.item())
            acc_metric.update_state(utils.compute_accuracy(y, y_pred))

    return loss_metric.result(), acc_metric.result()