import os
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

import utils
from data import BedsoreDataModule


class MyFasterRCNN(pl.LightningModule):

    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.net = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
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
        result.log_dict({'loss/train_loss':loss}, prog_bar=True)
        return result

    def validation_step(self, batch, batch_idx):
        self.net.train()
        images, targets = batch
        loss_dict = self(images, targets)
        loss = sum(loss for loss in loss_dict.values())
        result = pl.EvalResult(checkpoint_on=loss)
        result.log_dict({'loss/valid_loss':loss}, prog_bar=True)
        return result

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        return optimizer

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--lr', type=float, default=1e-4)
        parser.add_argument('--num_classes', type=int, default=7)
        parser.add_argument('--batch_size', type=int, default=4)
        parser.add_argument('--seed', type=int, default=32)
        parser.add_argument('--data_root', type=str, default='data')
        return parser

def main(args):
    seed_everything(args.seed)
    model = MyFasterRCNN(args)
    dm = BedsoreDataModule(args.data_root, args.batch_size, seed = args.seed)
    dm.setup()
    trainer = Trainer.from_argparse_args(args)
    trainer.fit(model,dm)

parser = ArgumentParser()
parser = MyFasterRCNN.add_model_specific_args(parser)
parser = Trainer.add_argparse_args(parser)
args = parser.parse_args()

if __name__ == '__main__':
    print(args)
    main(args)
