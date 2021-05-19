import os
import lmdb
import numpy as np
from PIL import Image
from tqdm import tqdm
import torchvision


def make_lmdb(image_set, out_name):
    ds = torchvision.datasets.VOCDetection('data',
                                           year='2007',
                                           image_set=image_set)
    # make byte lmdb
    env = lmdb.open(out_name, map_size=int(3e10))
    for i, d in enumerate(tqdm(ds)):
        with env.begin(write=True) as txn:
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


if __name__ == "__main__":
    make_lmdb(image_set='train', out_name='data/TRAIN_LMDB')
    make_lmdb(image_set='val', out_name='data/TEST_LMDB')
