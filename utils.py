import demjson
import torchvision
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from torchvision import transforms as T
import torch


def collate_fn(batch):
    return tuple(zip(*batch))


def batch2pil(x, nrow=8, normalize=True, padding=1, pad_value=1, range=None):
    grid = torchvision.utils.make_grid(
        x, normalize=normalize, nrow=nrow, pad_value=pad_value, padding=padding, range=range
    )
    return torchvision.transforms.ToPILImage()(grid.cpu())


def draw_bbox(image, target, th=0.5, show_mask=True, anno_scale=1, only_return_mask=False):
    """
    在图上绘制bbox
    Args:
        image: 是一个tensor格式的图像矩阵
        target: 包含bbox关键字的字典
        anno_scale: 缩放标注
        only_return_mask: 仅返回绘制好的mask，用于计算segmentation分数
    Returns:
        image: PIL image
    """

    if isinstance(image, Image.Image):
        image = T.ToTensor()(image)
    c, h, w = image.shape

    font_size = round(((h + w) / 2) / 25) * anno_scale
    line_width = round(((h + w) / 2) / 175) * anno_scale
    image = image.cpu().permute(1, 2, 0).numpy()
    image = Image.fromarray(np.uint8(image * 255))
    boxes = target['boxes']
    labels = target['labels']
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype("Arial.ttf", size=font_size)
    for i in range(boxes.size(0)):
        if labels[i] < 7:
            if 'scores' in target.keys():
                if target['scores'][i] > th:
                    box = boxes[i].tolist()
                    draw.rectangle(box, outline=(44, 150, 120), width=line_width)
                    draw.text((box[0] + 5, box[1] + 2), str(labels[i].item()) + ': ' + str(round(target['scores'][i].item(), 3)), font=font, fill=(255, 0, 0))
            else:
                box = boxes[i].tolist()
                draw.rectangle(box, outline=(44, 150, 120), width=line_width)
                draw.text((box[0] + 5, box[1] + 2), str(labels[i].item()), font=font, fill=(255, 0, 0))

    if show_mask:

        ry = Image.new("RGB", (w, h), (255, 0, 0))
        fr = Image.new("RGB", (w, h), (255, 255, 0))
        hs = Image.new("RGB", (w, h), (0, 0, 255))
        tissue = {7: hs, 8: fr, 9: ry}
        if only_return_mask:
            image = Image.new("RGB", (w, h), (0, 0, 0))

        if 'scores' in target.keys():  # 如果有分数
            show_labels = target['labels'][target['scores'] > th]
            good_masks = target['masks'][target['scores'] > th][show_labels > 6]
            show_labels = show_labels[show_labels > 6]
        else:
            show_labels = target['labels']
            good_masks = target['masks'][show_labels > 6]
            show_labels = show_labels[show_labels > 6]
            if len(show_labels) == 0 and only_return_mask:
                return image

        for i, t in enumerate(show_labels):
            mask = batch2pil(((good_masks[i]) > 0.5).float())
            if only_return_mask:
                mask = np.array(mask)
            else:
                mask = np.array(mask) * 0.5
            mask = Image.fromarray(mask.astype('uint8')).convert('L')
            image = Image.composite(tissue[t.item()], image, mask)

    return image


def out2detfile(target, out, score_th=0.5):
    label_dict = {1: '1期', 2: '2期', 3: '3期', 4: '4期', 5: '不可分期', 6: '深部组织损伤'}
    fname = target['fname']
    out_labels = out['labels']
    out_scores = out['scores'].detach()
    bb = out['boxes']
    outs = []
    for i, t in enumerate(out_labels):
        if int(t) < 7 and out_scores[i] > score_th:
            outs.append(f'{label_dict[int(t)]} {fname} {out_scores[i]:.3} {int(bb[i][0])} {int(bb[i][1])} {int(bb[i][2])} {int(bb[i][3])}\n')
    return outs


@torch.no_grad()
def same_class_remove(out, score_th=0.5, overlap_th=0.9):
    """
    同类别重叠抑制，相同类别的框，重叠率大于overlap_th的被清除（分数置于0）
    输入的是模型的输出out(tensor)
    """
    passing_scores_ind = torch.where(out["scores"] > score_th)[0]
    for i in range(1, 7):
        labels_ind = torch.where(out["labels"] == i)[0]
        if labels_ind.size(0) > 1:
            ind = set(passing_scores_ind.tolist()) & set(labels_ind.tolist())
            ind = list(ind)
            for i in ind:
                for j in ind[i + 1:]:
                    box_i = out['boxes'][i].detach()
                    box_j = out['boxes'][j].detach()
                    box_cat = out['boxes'][[i, j]]
                    ov_x1 = box_cat[:, 0].max()
                    ov_y1 = box_cat[:, 1].max()
                    ov_x2 = box_cat[:, 2].min()
                    ov_y2 = box_cat[:, 3].min()
                    if ov_x2 > ov_x1 and ov_y2 > ov_y1:
                        area_ov = (ov_x2 - ov_x1) * (ov_y2 - ov_y1).item()
                        area_i = (box_i[2] - box_i[0]) * (box_i[3] - box_i[1]).item()
                        area_j = (box_j[2] - box_j[0]) * (box_j[3] - box_j[1]).item()

                        if area_i > area_j:
                            area_min = area_j
                            pending = j
                        else:
                            area_min = area_i
                            pending = i

                        ovmin = area_ov / area_min
                        if ovmin.item() > overlap_th:
                            out['scores'][pending] = 0.
    return out


class SegmentationMetric(object):
    """分割函数的指标"""

    def __init__(self, numClass):
        self.numClass = numClass
        self.confusionMatrix = np.zeros((self.numClass,) * 2)

    def pixelAccuracy(self):
        # return all class overall pixel accuracy
        #  PA = acc = (TP + TN) / (TP + TN + FP + TN)
        acc = np.diag(self.confusionMatrix).sum() / self.confusionMatrix.sum()
        return acc

    def classPixelAccuracy(self):
        # return each category pixel accuracy(A more accurate way to call it precision)
        # acc = (TP) / TP + FP
        classAcc = np.diag(self.confusionMatrix) / self.confusionMatrix.sum(axis=1)
        return classAcc  # 返回的是一个列表值，如：[0.90, 0.80, 0.96]，表示类别1 2 3各类别的预测准确率

    def meanPixelAccuracy(self):
        classAcc = self.classPixelAccuracy()
        meanAcc = np.nanmean(classAcc)  # np.nanmean 求平均值，nan表示遇到Nan类型，其值取为0
        return meanAcc  # 返回单个值，如：np.nanmean([0.90, 0.80, 0.96, nan, nan]) = (0.90 + 0.80 + 0.96） / 3 =  0.89

    def meanIntersectionOverUnion(self):
        # Intersection = TP Union = TP + FP + FN
        # IoU = TP / (TP + FP + FN)
        intersection = np.diag(self.confusionMatrix)  # 取对角元素的值，返回列表
        union = np.sum(self.confusionMatrix, axis=1) + np.sum(self.confusionMatrix, axis=0) - np.diag(self.confusionMatrix)  # axis = 1表示混淆矩阵行的值，返回列表； axis = 0表示取混淆矩阵列的值，返回列表
        IoU = intersection / union  # 返回列表，其值为各个类别的IoU
        mIoU = np.nanmean(IoU)  # 求各类别IoU的平均
        return mIoU

    def genConfusionMatrix(self, imgPredict, imgLabel):  # 同FCN中score.py的fast_hist()函数
        # remove classes from unlabeled pixels in gt image and predict
        mask = (imgLabel >= 0) & (imgLabel < self.numClass) # mask>=0
        label = self.numClass * imgLabel[mask] + imgPredict[mask]
        count = np.bincount(label, minlength=self.numClass**2)
        confusionMatrix = count.reshape(self.numClass, self.numClass)
        return confusionMatrix

    def Frequency_Weighted_Intersection_over_Union(self):
        # FWIOU =     [(TP+FN)/(TP+FP+TN+FN)] *[TP / (TP + FP + FN)]
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
            np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
            np.diag(self.confusion_matrix))
        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def addBatch(self, imgPredict, imgLabel):
        assert imgPredict.shape == imgLabel.shape
        self.confusionMatrix += self.genConfusionMatrix(imgPredict, imgLabel)

    def reset(self):
        self.confusionMatrix = np.zeros((self.numClass, self.numClass))
