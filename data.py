# -*- coding: utf-8 -*-
import os
import pathlib
import copy

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, random_split
import torchvision

from detection_transforms import RandomHorizontalFlip, RandomVerticalFlip, RandomRotate, RandomResize, Compose, ToTensor
import utils


class BedsoreDataset(object):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        self.data = torchvision.datasets.VOCDetection(root, year='2007')
        self.label_dict = {'1期':1, '2期':2, '3期':3, '4期':4, '不可分期':5, '深部组织损伤':6}

    def __getitem__(self, idx):
        
        img = self.data[idx][0]
        
        boxes,labels = [],[]
        image_id = torch.tensor([idx])
        
        fname = self.data[idx][1]['annotation']['filename'][:-4]
        mask_path = f'{self.root}/VOCdevkit/VOC2007/SegmentationClass/{fname}.png'
        if os.path.exists(mask_path):
            mask = Image.open(mask_path)
        else:
            mask = None
        
        if isinstance(self.data[idx][1]['annotation']['object'], list):
            for i in self.data[idx][1]['annotation']['object']: 
                bbox = list(i['bndbox'].values())
                bbox = list(map(float,bbox))
                boxes.append(bbox)
                labels.append(self.label_dict[i['name']])
        else:
            bbox = list(self.data[idx][1]['annotation']['object']['bndbox'].values())
            bbox = list(map(float,bbox))
            boxes.append(bbox)
            labels.append(self.label_dict[self.data[idx][1]['annotation']['object']['name']])
            
        target = {}
        target['boxes'] = torch.as_tensor(boxes, dtype=torch.float32)
        target['labels'] = torch.as_tensor(labels, dtype=torch.int64)
        target['image_id'] = torch.as_tensor(image_id, dtype=torch.int64)
        
        if self.transforms is not None:
            img, target = self.transforms(img, target)
        if mask:
            target['masks'] = torchvision.transforms.ToTensor()(mask)
        else:
            w = int(self.data[idx][1]['annotation']['size']['width'])
            h = int(self.data[idx][1]['annotation']['size']['height'])
            target['masks'] = torch.zeros((3,h,w))

        return img, target

    def __len__(self):
        return len(self.data)


class BedsoreDataModule(LightningDataModule):

    def __init__(self, root, batch_size, seed=32):
        super().__init__()
        self.root = root
        self.batch_size = batch_size
        self.seed = seed

    def setup(self):
        tfmc_train = Compose([
            ToTensor()
        ])
        #  tfmc_train = Compose([
            #  (RandomHorizontalFlip(0.5)),
            #  ToTensor()
        #  ])
        tfmc_valid = Compose([
            ToTensor()
        ])
        ds = BedsoreDataset(self.root, transforms=tfmc_train)
        self.train_ds, self.valid_ds = torch.utils.data.random_split(
            ds, [len(ds)-20, 20], generator=torch.Generator().manual_seed(self.seed))
        self.valid_ds = copy.deepcopy(self.valid_ds)
        self.valid_ds.dataset.transforms = tfmc_valid  # 如果验证集要调整transformer



    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True, num_workers=8, collate_fn=utils.collate_fn)

    def val_dataloader(self):
        return DataLoader(self.valid_ds, batch_size=self.batch_size, shuffle=False, num_workers=8, collate_fn=utils.collate_fn)


if __name__ == '__main__':
    root = 'data'
    tfmc = T.transforms.Compose([
        T.ToTensor(),
    ])

    dm = BedsoreDataModule(root, batch_size=8)
    dm.setup()
    print(len(dm.train_ds))
    print(len(dm.valid_ds))
    out = next(iter(dm.train_dataloader()))
    import ipdb;ipdb.set_trace() 
    pass
