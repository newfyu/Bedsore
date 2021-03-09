import os
import shutil
from argparse import ArgumentParser

import pytorch_lightning as pl
import torch
import torchvision
import torchvision.transforms as T
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.metrics.functional import accuracy
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import maskrcnn_resnet50_fpn

import utils
import voc_eval
from data import BedsoreDataModule
from logger import MLFlowLogger2


class MyFasterRCNN(pl.LightningModule):

    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        #  self.net = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        self.net = maskrcnn_resnet50_fpn(pretrained=True, trainable_backbone_layers=self.hparams.train_layers)
        in_features = self.net.roi_heads.box_predictor.cls_score.in_features
        self.net.roi_heads.box_predictor = FastRCNNPredictor(in_features, self.hparams.num_classes)

    def forward(self, images, targets=None):
        if targets is not None:
            return self.net(images, targets)
        else:
            return self.net(images)

    def training_step(self, batch, batch_idx):
        images, targets = batch
        loss_dict = self(images, targets)
        loss = sum(loss for loss in loss_dict.values())
        result = pl.TrainResult(minimize=loss)
        result.log_dict({'loss/train_loss': loss}, prog_bar=True)
        return result

    #  def validation_step(self, batch, batch_idx):
        #  self.net.train()  # net output loss when set train,net output label and bbox when set eval
        #  images, targets = batch
        #  loss_dict = self(images, targets)
        #  loss = sum(loss for loss in loss_dict.values())
        #  result = pl.EvalResult(checkpoint_on=loss)
        #  result.log_dict({'loss/valid_loss': loss}, prog_bar=True)
        #  return result

    def validation_step(self, batch, batch_idx):
        self.net.eval()
        image, target = batch
        output = self(image, target)
        detfile = []
        detfile.append(utils.out2detfile(target[0], output[0]))
        return detfile, target[0]['fname']

    def cal_ap(self, outputs):
        outputs = list(zip(*outputs))
        fnamelist = outputs[1]
        outputs = outputs[0]
        detfiles = []
        for i in outputs:
            for j in i:
                detfiles += j

        class_name = ['1期', '2期', '3期', '4期', '不可分期', '深部组织损伤']
        mAP = []
        class_ap = []
        for i in class_name:
            ap = voc_eval.voc_eval(detfiles, 'data/VOCdevkit/VOC2007/Annotations/{}.xml', fnamelist, i,
                                   ovthresh=0.3,
                                   use_07_metric=True)[-1]
            class_ap.append([i, ap])
            mAP.append(ap)
        mAP = sum(mAP) / len(mAP)
        return mAP, class_ap

    def validation_epoch_end(self, outputs):
        mAP, class_ap = self.cal_ap(outputs)
        result = pl.EvalResult(checkpoint_on=-torch.FloatTensor([mAP]))
        result.log_dict({"valid_map": mAP})
        return result

    def test_step(self, batch, batch_idx):
        self.net.eval()
        image, target = batch
        output = self(image, target)
        detfile = []
        detfile.append(utils.out2detfile(target[0], output[0]))
        return detfile, target[0]['fname']

    def test_epoch_end(self, outputs):
        mAP, class_ap = self.cal_ap(outputs)
        result = pl.EvalResult(checkpoint_on=torch.FloatTensor([mAP]))
        result.log_dict({"valid_mAP": mAP})
        print(class_ap)
        result = pl.EvalResult(checkpoint_on=torch.FloatTensor([mAP]))
        result.log_dict({"test_mAP": mAP})
        return result

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        return optimizer

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False, conflict_handler="resolve")
        parser.add_argument("--exp", type=str, default="Default")
        parser.add_argument("--name", type=str, default="test")
        parser.add_argument('--lr', type=float, default=1e-4)
        parser.add_argument('--num_classes', type=int, default=10)
        parser.add_argument('--batch_size', type=int, default=3)
        parser.add_argument('--num_valid', type=int, default=50)
        parser.add_argument('--seed', type=int, default=32)
        parser.add_argument('--data_root', type=str, default='data')
        parser.add_argument('--distributed_backend', type=str, default='dp')
        parser.add_argument("--amp_level", type=str, default="O0")
        parser.add_argument("--train_layers", type=int, default=3)
        parser.add_argument("--trans_prob", type=float, default=0.5)
        parser.add_argument("--max_epochs", type=int, default=40)
        parser.add_argument("--test_ckpt", type=str, default='')
        return parser


def main():
    parser = ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    parser = MyFasterRCNN.add_model_specific_args(parser)
    args = parser.parse_args()
    seed_everything(args.seed)
    model = MyFasterRCNN(args)
    dm = BedsoreDataModule(args.data_root, args.batch_size, args.num_valid, args.trans_prob, seed=args.seed)
    trainer = Trainer.from_argparse_args(args)

    if args.test_ckpt != '':  # test
        model = MyFasterRCNN.load_from_checkpoint(args.test_ckpt)
        trainer.test(model, test_dataloaders=dm.test_dataloader())
    else:  # train
        if args.name != "test":
            logger = MLFlowLogger2(experiment_name=args.exp, run_name=args.name)
            # save source files to mlflow
            save_dir = f"mlruns/{logger.experiment_id}/{logger.run_id}/artifacts"
            save_files = [i for i in os.listdir() if '.py' in i]
            for i in save_files:
                shutil.copy(i, f'{save_dir}/{i}')
        else:
            logger = None
        trainer.logger = logger
        trainer.fit(model, dm)
        trainer.test()


if __name__ == '__main__':
    main()

# python model.py --gpus '1,' --max_epochs 50 --name test
