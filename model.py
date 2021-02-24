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
from torchvision.models.detection.mask_rcnn import maskrcnn_resnet50_fpn

import utils
from data import BedsoreDataModule
from logger import MLFlowLogger2


class MyFasterRCNN(pl.LightningModule):

    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        #  self.net = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        self.net = maskrcnn_resnet50_fpn(pretrained=True)
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

    def validation_step(self, batch, batch_idx):
        self.net.train()
        images, targets = batch
        loss_dict = self(images, targets)
        loss = sum(loss for loss in loss_dict.values())
        result = pl.EvalResult(checkpoint_on=loss)
        result.log_dict({'loss/valid_loss': loss}, prog_bar=True)
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
        parser.add_argument('--num_valid', type=int, default=100)
        parser.add_argument('--seed', type=int, default=32)
        parser.add_argument('--data_root', type=str, default='data')
        parser.add_argument('--distributed_backend', type=str, default='dp')
        parser.add_argument("--amp_level", type=str, default="O0")
        return parser


def main():
    parser = ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    parser = MyFasterRCNN.add_model_specific_args(parser)
    args = parser.parse_args()
    seed_everything(args.seed)
    model = MyFasterRCNN(args)
    dm = BedsoreDataModule(args.data_root, args.batch_size, args.num_valid, seed=args.seed)
    dm.setup()
    trainer = Trainer.from_argparse_args(args)
    if args.name != "test":
        logger = MLFlowLogger2(experiment_name=args.exp, run_name=args.name)
    else:
        logger = None
    trainer.logger = logger
    trainer.fit(model, dm)


if __name__ == '__main__':
    main()
