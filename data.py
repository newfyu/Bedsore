# -*- coding: utf-8 -*-
import copy
import os
import pathlib

import numpy as np
import torch
import torchvision
import torchvision.transforms as T
from PIL import Image
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, random_split
import albumentations as album
from albumentations.pytorch import ToTensor
import lmdb

import utils


class BedsoreLMDB(object):
    def __init__(self,
                 root='data',
                 transforms=None,
                 image_set='train',
                 val=False,
                 chunk_id=0,
                 chunk_num=5):
        self.root = root
        self.transforms = transforms
        self.data = torchvision.datasets.VOCDetection(root,
                                                      year='2007',
                                                      image_set=image_set)
        self.label_dict = {
            '1期': 1,
            '2期': 2,
            '3期': 3,
            '4期': 4,
            '不可分期': 5,
            '深部组织损伤': 6
        }

        if chunk_num > 0:  # 如果是训练集，划分一个验证集出来
            torch.manual_seed(32)
            sample_id = torch.randperm(len(self.data))
            valid_id = sample_id.chunk(chunk_num)[chunk_id].tolist()
            train_id = list(set(sample_id.tolist()) - set(valid_id))
            if val == True:
                self.sample_id = valid_id
            else:
                self.sample_id = train_id
        else:
            self.sample_id = list(range(len(self.data)))

        env = lmdb.open(f'{self.root}/arr_lmdb')
        self.txn = env.begin(write=False)

    def __getitem__(self, idx):
        if idx >= len(self.sample_id):
            raise StopIteration
        idx = self.sample_id[idx]
        anno = self.txn.get(f'anno_{idx}'.encode())  # sub self.data[idx][1]
        anno = eval(anno)
        w = anno['annotation']['size']['width']
        h = anno['annotation']['size']['height']

        image = self.txn.get(f'data_{idx}'.encode())
        image = np.frombuffer(image, dtype=np.uint8)  #arr
        image = image.reshape(int(h), int(w), 3)

        boxes, labels = [], []
        image_id = torch.tensor([idx])

        fname = anno['annotation']['filename'][:-4]

        mask = self.txn.get(f'mask_{idx}'.encode())
        if mask is not None:
            mask = np.frombuffer(mask, dtype=np.uint8)
            mask = mask.reshape(int(h), int(w))
            obj_ids = np.unique(mask)[1:]
            masks = mask == obj_ids[:, None, None]
            mask_class = self.txn.get(f'mask_class_{idx}'.encode())
            mask_class = np.frombuffer(mask_class, dtype=np.uint8)
            mask_class = mask_class.reshape(int(h), int(w))
            mask_class = masks * mask_class
            mask_label = mask_class.max(1).max(1).tolist()
            ccc = {
                137: 7,
                173: 8,
                98: 9
            }  # 137-7:Necrotic,173-8:slough,98-9:Granulation
            mask_label = [ccc[i] if i in ccc else i for i in mask_label]
            num_objs = len(obj_ids)
            mask_boxes = []
            for i in range(num_objs):
                pos = np.where(masks[i])
                xmin = float(np.min(pos[1]))
                xmax = float(np.max(pos[1]))
                ymin = float(np.min(pos[0]))
                ymax = float(np.max(pos[0]))
                mask_boxes.append([xmin, ymin, xmax, ymax])
        else:
            masks = None
            mask_boxes = []
            mask_label = []

        if isinstance(anno['annotation']['object'], list):
            for i in anno['annotation']['object']:
                bbox = list(i['bndbox'].values())
                bbox = list(map(float, bbox))
                boxes.append(bbox)
                labels.append(self.label_dict[i['name']])
        else:
            bbox = list(anno['annotation']['object']['bndbox'].values())
            bbox = list(map(float, bbox))
            boxes.append(bbox)
            labels.append(
                self.label_dict[anno['annotation']['object']['name']])

        pre_masks = torch.zeros(len(labels), image.shape[0], image.shape[1])
        labels = labels + mask_label
        target = {}
        boxes = boxes + mask_boxes
        target['boxes'] = torch.as_tensor(boxes, dtype=torch.float32)
        target['labels'] = torch.as_tensor(labels, dtype=torch.int64)
        target['image_id'] = torch.as_tensor(image_id, dtype=torch.int64)
        target['fname'] = fname

        if masks is not None:
            masks = torch.from_numpy(masks)
            masks = torch.cat((pre_masks, masks), dim=0)
            target['masks'] = masks
        else:
            target['masks'] = pre_masks

        if self.transforms is not None:
            transformed = self.transforms(
                image=image,
                bboxes=target['boxes'],
                mask=target['masks'].permute(1, 2, 0).numpy(),
                category_ids=target['labels'].tolist())
            image = transformed['image']
            target['boxes'] = torch.as_tensor(transformed['bboxes'],
                                              dtype=torch.float32)
            target['masks'] = transformed['mask'][0].permute(2, 0, 1)

        return image, target

    def __len__(self):
        return len(self.sample_id)


class BedsoreDataset(object):
    def __init__(self, root='data', transforms=None, image_set='train'):
        self.root = root
        self.transforms = transforms
        self.data = torchvision.datasets.VOCDetection(root,
                                                      year='2007',
                                                      image_set=image_set)
        self.label_dict = {
            '1期': 1,
            '2期': 2,
            '3期': 3,
            '4期': 4,
            '不可分期': 5,
            '深部组织损伤': 6
        }

    def __getitem__(self, idx):
        image = self.data[idx][0]

        boxes, labels = [], []
        image_id = torch.tensor([idx])

        fname = self.data[idx][1]['annotation']['filename'][:-4]
        mask_class_path = f'{self.root}/VOCdevkit/VOC2007/SegmentationClass/{fname}.png'
        mask_object_path = f'{self.root}/VOCdevkit/VOC2007/SegmentationObject/{fname}.png'
        if os.path.exists(mask_object_path):
            mask = Image.open(mask_object_path).convert('L')
            mask = np.array(mask)
            obj_ids = np.unique(mask)[1:]
            masks = mask == obj_ids[:, None, None]
            mask_class = Image.open(mask_class_path).convert('L')
            mask_class = np.array(mask_class)
            mask_class = masks * mask_class
            mask_label = mask_class.max(1).max(1).tolist()
            ccc = {
                137: 7,
                173: 8,
                98: 9
            }  # 137-7:Necrotic,173-8:slough,98-9:Granulation
            mask_label = [ccc[i] if i in ccc else i for i in mask_label]
            num_objs = len(obj_ids)
            mask_boxes = []
            for i in range(num_objs):
                pos = np.where(masks[i])
                xmin = float(np.min(pos[1]))
                xmax = float(np.max(pos[1]))
                ymin = float(np.min(pos[0]))
                ymax = float(np.max(pos[0]))
                mask_boxes.append([xmin, ymin, xmax, ymax])
        else:
            masks = None
            mask_boxes = []
            mask_label = []

        if isinstance(self.data[idx][1]['annotation']['object'], list):
            for i in self.data[idx][1]['annotation']['object']:
                bbox = list(i['bndbox'].values())
                bbox = list(map(float, bbox))
                boxes.append(bbox)
                labels.append(self.label_dict[i['name']])
        else:
            bbox = list(
                self.data[idx][1]['annotation']['object']['bndbox'].values())
            bbox = list(map(float, bbox))
            boxes.append(bbox)
            labels.append(self.label_dict[self.data[idx][1]['annotation']
                                          ['object']['name']])

        pre_masks = torch.zeros(len(labels), image.size[1], image.size[0])
        labels = labels + mask_label
        target = {}
        boxes = boxes + mask_boxes
        target['boxes'] = torch.as_tensor(boxes, dtype=torch.float32)
        target['labels'] = torch.as_tensor(labels, dtype=torch.int64)
        target['image_id'] = torch.as_tensor(image_id, dtype=torch.int64)
        target['fname'] = fname

        if masks is not None:
            masks = torch.from_numpy(masks)
            masks = torch.cat((pre_masks, masks), dim=0)
            target['masks'] = masks
        else:
            target['masks'] = pre_masks
        if self.transforms is not None:
            transformed = self.transforms(
                image=np.array(image),
                bboxes=target['boxes'].tolist(),
                mask=target['masks'].permute(1, 2, 0).numpy(),
                category_ids=target['labels'].tolist())

            target['boxes'] = torch.as_tensor(transformed['bboxes'],
                                              dtype=torch.float32)
            target['masks'] = transformed['mask'][0].permute(2, 0, 1)

            return image, target

    def __len__(self):
        return len(self.data)


class BedsoreDataModule(LightningDataModule):
    def __init__(self,
                 root,
                 batch_size,
                 num_valid,
                 trans_prob,
                 num_workers=8,
                 seed=32):
        super().__init__()
        self.root = root
        self.batch_size = batch_size
        self.seed = seed
        self.num_workers = num_workers

        tfmc_train = album.Compose(
            [
                album.RandomSizedBBoxSafeCrop(
                    800, 800, p=trans_prob, erosion_rate=0.2),
                album.HorizontalFlip(p=trans_prob),
                album.VerticalFlip(p=trans_prob),
                album.ShiftScaleRotate(p=trans_prob, rotate_limit=90),
                album.RandomBrightnessContrast(p=trans_prob),
                #  album.RGBShift(r_shift_limit=30, g_shift_limit=30, b_shift_limit=30, p=0.3),
                ToTensor()
            ],
            bbox_params=album.BboxParams(format='pascal_voc',
                                         label_fields=['category_ids']))
        tfmc_valid = album.Compose([ToTensor()],
                                   bbox_params=album.BboxParams(
                                       format='pascal_voc',
                                       label_fields=['category_ids']))

        ds = BedsoreDataset(self.root, transforms=tfmc_train)
        self.train_ds, self.valid_ds = torch.utils.data.random_split(
            ds, [len(ds) - num_valid, num_valid],
            generator=torch.Generator().manual_seed(self.seed))
        self.valid_ds = copy.deepcopy(self.valid_ds)
        self.valid_ds.dataset.transforms = tfmc_valid  # 如果验证集要调整transformer
        self.test_ds = BedsoreDataset(self.root,
                                      transforms=tfmc_valid,
                                      image_set='val')

    def train_dataloader(self):
        return DataLoader(self.train_ds,
                          batch_size=self.batch_size,
                          shuffle=True,
                          num_workers=self.num_workers,
                          collate_fn=utils.collate_fn)

    def val_dataloader(self):
        return DataLoader(self.valid_ds,
                          batch_size=1,
                          shuffle=False,
                          num_workers=self.num_workers,
                          collate_fn=utils.collate_fn)

    def test_dataloader(self):
        return DataLoader(self.test_ds,
                          batch_size=1,
                          shuffle=False,
                          num_workers=self.num_workers,
                          collate_fn=utils.collate_fn)


class BedsoreLMDBDataModule(LightningDataModule):
    def __init__(self,
                 root,
                 batch_size,
                 trans_prob,
                 chunk_num,
                 chunk_id,
                 num_workers=8,
                 seed=32):
        super().__init__()
        self.root = root
        self.batch_size = batch_size
        self.seed = seed
        self.num_workers = num_workers

        tfmc_train = album.Compose(
            [
                album.RandomSizedBBoxSafeCrop(
                    800, 800, p=0.5, erosion_rate=0.2),
                album.HorizontalFlip(p=trans_prob),
                album.VerticalFlip(p=trans_prob),
                album.ShiftScaleRotate(p=trans_prob, rotate_limit=90),
                album.RandomBrightnessContrast(p=trans_prob),
                #  album.RGBShift(r_shift_limit=30, g_shift_limit=30, b_shift_limit=30, p=0.3),
                ToTensor()
            ],
            bbox_params=album.BboxParams(format='pascal_voc',
                                         label_fields=['category_ids']))
        tfmc_valid = album.Compose([ToTensor()],
                                   bbox_params=album.BboxParams(
                                       format='pascal_voc',
                                       label_fields=['category_ids']))

        self.train_ds = BedsoreLMDB(self.root,
                                    transforms=tfmc_train,
                                    image_set='train',
                                    val=False,
                                    chunk_id=chunk_id,
                                    chunk_num=chunk_num)

        self.valid_ds = BedsoreLMDB(self.root,
                                    transforms=tfmc_valid,
                                    image_set='train',
                                    val=True,
                                    chunk_id=chunk_id,
                                    chunk_num=chunk_num)

        self.test_ds = BedsoreLMDB(self.root,
                                      transforms=tfmc_valid,
                                      image_set='val',
                                      chunk_num=0)

    def train_dataloader(self):
        return DataLoader(self.train_ds,
                          batch_size=self.batch_size,
                          shuffle=True,
                          num_workers=self.num_workers,
                          collate_fn=utils.collate_fn)

    def val_dataloader(self):
        return DataLoader(self.valid_ds,
                          batch_size=1,
                          shuffle=False,
                          num_workers=self.num_workers,
                          collate_fn=utils.collate_fn)

    def test_dataloader(self):
        return DataLoader(self.test_ds,
                          batch_size=1,
                          shuffle=False,
                          num_workers=self.num_workers,
                          collate_fn=utils.collate_fn)


