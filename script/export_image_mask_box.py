import os
import sys
sys.path.append("/home/zrway/Projects/Bedsore")
#  sys.path.append("..")

import data
import albumentations as A
from data import BedsoreDataset
from torchvision import transforms as T
import numpy as np
from albumentations.pytorch import ToTensorV2,ToTensor
import torch
from utils import batch2pil, draw_bbox, out2detfile
from PIL import Image
from tqdm import tqdm


atfmc = A.Compose([
#     A.RandomScale(p=0.5),
#     A.RandomShadow(p=0.5),
#     A.HorizontalFlip(p=0.5),
#     A.ShiftScaleRotate(p=0.5),
#     A.RandomBrightnessContrast(p=0.3),
#     A.RGBShift(r_shift_limit=30, g_shift_limit=30, b_shift_limit=30, p=0.3),
    ToTensor(),
    ],bbox_params=A.BboxParams(format='pascal_voc', label_fields=['category_ids']))
# atfmc = None

ds = BedsoreDataset('data', transforms=atfmc)

SZ = 256
ry = Image.new("RGB",(SZ,SZ),(255,0,0))
fr = Image.new("RGB",(SZ,SZ),(255,255,0))
hs = Image.new("RGB",(SZ,SZ),(0,0,255))
tissue = {7:hs, 8:fr, 9:ry}

for i in tqdm(range(0,len(ds))):
    image,target = ds[i]
#     print(i,target['fname'])
    
    good_labels = target['labels']
    good_masks = target['masks'][good_labels>6]
    good_labels = good_labels[good_labels>6]
    img_out = draw_bbox(image,target).resize((SZ,SZ))

    for i,t in enumerate(good_labels):
        mask = batch2pil(((good_masks[i])>0.5).float()).resize((SZ,SZ))
        mask = np.array(mask)*0.5
        mask = Image.fromarray(mask.astype('uint8')).convert('L')
        img_out = Image.composite(tissue[t.item()], img_out, mask)
    img_out.save(f"check_image/{target['fname']}.jpg")
