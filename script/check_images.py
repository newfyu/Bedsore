import os
import sys
sys.path.append(os.getcwd())
from tqdm import tqdm
from PIL import Image
from utils import batch2pil, draw_bbox
from albumentations.pytorch import ToTensor
import numpy as np
from data import BedsoreLMDB
import albumentations as A
import shutil


def export_image(ds):
    for i in tqdm(range(0, len(ds))):
        image, target = ds[i]

        good_labels = target['labels']
        good_masks = target['masks'][good_labels > 6]
        good_labels = good_labels[good_labels > 6]
        img_out = draw_bbox(image, target).resize((SZ, SZ))

        for i, t in enumerate(good_labels):
            mask = batch2pil(((good_masks[i]) > 0.5).float()).resize((SZ, SZ))
            mask = np.array(mask) * 0.5
            mask = Image.fromarray(mask.astype('uint8')).convert('L')
            img_out = Image.composite(tissue[t.item()], img_out, mask)
        img_out.save(f"check_image/{target['fname']}.jpg")


if __name__ == "__main__":

    SZ = 512
    ry = Image.new("RGB", (SZ, SZ), (255, 0, 0))
    fr = Image.new("RGB", (SZ, SZ), (255, 255, 0))
    hs = Image.new("RGB", (SZ, SZ), (0, 0, 255))
    tissue = {7: hs, 8: fr, 9: ry}

    out_dir = 'check_image'
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.mkdir(out_dir)

    atfmc = A.Compose([ToTensor()],
                      bbox_params=A.BboxParams(format='pascal_voc',
                                               label_fields=['category_ids']))
    train_ds = BedsoreLMDB(root='data', subset='train', transforms=atfmc)
    valid_ds = BedsoreLMDB(root='data', subset='valid', transforms=atfmc)
    test_ds = BedsoreLMDB(root='data', subset='test', transforms=atfmc)
    export_image(train_ds)
    export_image(valid_ds)
    export_image(test_ds)
