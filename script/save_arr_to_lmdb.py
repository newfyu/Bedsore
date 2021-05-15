import os
import lmdb
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import torch
import torchvision.transforms as T
import torchvision

LMDB_NAME = 'data/arr_lmdb'

if __name__ == "__main__":
    ds = torchvision.datasets.VOCDetection('data',
                                           year='2007',
                                           image_set='trainval')
    # make byte lmdb
    env = lmdb.open(LMDB_NAME, map_size=int(3e10))
    with env.begin(write=True) as txn:
        for i, d in enumerate(tqdm(ds)):
            fname = d[1]['annotation']['filename'][:-4]
            mask_class_path = f'data/VOCdevkit/VOC2007/SegmentationClass/{fname}.png'
            mask_object_path = f'data/VOCdevkit/VOC2007/SegmentationObject/{fname}.png'
            if os.path.exists(mask_object_path):
                mask = Image.open(mask_object_path).convert('L')
                mask_class = Image.open(mask_class_path).convert('L')
                mask_byte = np.array(mask).tobytes()
                mask_class_byte = np.array(mask_class).tobytes()
                txn.put(f'mask_{i}'.encode(), mask_byte)
                txn.put(f'mask_class_{i}'.encode(), mask_class_byte)

            arr = np.array(d[0])
            byte = arr.tobytes()
            anno = str(d[1])
            txn.put(f'data_{i}'.encode(), byte)
            txn.put(f'anno_{i}'.encode(), anno.encode())
    env.close()
