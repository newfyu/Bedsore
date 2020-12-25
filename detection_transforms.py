import random

import torch
from torchvision.transforms import functional as F
import torchvision.transforms as T
from collections import Iterable


def _flip_coco_person_keypoints(kps, width):
    flip_inds = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
    flipped_data = kps[:, flip_inds]
    flipped_data[..., 0] = width - flipped_data[..., 0]
    # Maintain COCO convention that if visibility == 0, then x, y = 0
    inds = flipped_data[..., 2] == 0
    flipped_data[inds] = 0
    return flipped_data


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms # list

    def __call__(self, image, target):
        for t in self.transforms:
            if isinstance(t, Iterable): # 如果transformers中元素是一个列表，则随机选择一个
                t = random.choice(t)
            image, target = t(image, target)
        return image, target


#  class RandomHorizontalFlip(object):
    #  def __init__(self, prob):
        #  self.prob = prob

    #  def __call__(self, image, target):
        #  if random.random() < self.prob:
        #  height, width = image.shape[-2:]
        #  image = image.flip(-1)
        #  bbox = target["boxes"]
        #  bbox[:, [0, 2]] = width - bbox[:, [2, 0]]
        #  target["boxes"] = bbox
        #  if "masks" in target:
        #  target["masks"] = target["masks"].flip(-1)
        #  if "keypoints" in target:
        #  keypoints = target["keypoints"]
        #  keypoints = _flip_coco_person_keypoints(keypoints, width)
        #  target["keypoints"] = keypoints
        #  return image, target


class RandomHorizontalFlip(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            W, H = image.size
            image = T.functional.hflip(image)
            target['masks'] = T.functional.hflip(target['masks'])
            boxes = target['boxes']
            boxes2 = boxes.clone()
            boxes[:, 0] = W - boxes2[:, 2]
            boxes[:, 2] = W - boxes2[:, 0]
        return image, target


class RandomVerticalFlip(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            W, H = image.size
            image = T.functional.vflip(image)
            boxes = target['boxes']
            boxes2 = boxes.clone()
            boxes[:, 1] = H - boxes2[:, 3]
            boxes[:, 3] = H - boxes2[:, 1]
        return image, target

class RandomRotate(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            angle = random.choice([90,180,270])
            W, H = image.size
            boxes = target['boxes']
            boxes2 = boxes.clone()
            if angle == 90:
                image = T.functional.rotate(image, 90, expand=True)
                boxes[:, 0] = boxes2[:, 1]
                boxes[:, 1] = W - boxes2[:, 2]
                boxes[:, 2] = boxes2[:, 3]
                boxes[:, 3] = W - boxes2[:, 0]
            if angle == 180:
                image = T.functional.rotate(image, 180, expand=True)
                boxes[:, 0] = W - boxes2[:, 2]
                boxes[:, 1] = H - boxes2[:, 3]
                boxes[:, 2] = W - boxes2[:, 0]
                boxes[:, 3] = H - boxes2[:, 1]
            if angle == 270:
                image = T.functional.rotate(image, 270, expand=True)
                boxes[:, 0] = H - boxes2[:, 3]
                boxes[:, 1] = boxes2[:, 0]
                boxes[:, 2] = H - boxes2[:, 1]
                boxes[:, 3] = boxes2[:, 2]
        return image, target

class RandomResize(object):
    def __init__(self, prob=0.5, w_scale=(0.5,1.5), h_scale=(0.5,1.5)):
        self.prob = prob
        self.w_scale = random.uniform(w_scale[0], w_scale[1])
        self.h_scale = random.uniform(h_scale[0], h_scale[1])

    def __call__(self, image, target):
        if random.random() < self.prob:
            W, H = image.size
            image = T.functional.resize(image, (int(H*self.h_scale),int(W*self.w_scale)))
            boxes = target['boxes']
            boxes[:, [0,2]] *= self.w_scale
            boxes[:, [1,3]] *= self.h_scale
        return image, target

class ToTensor(object):
    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target
