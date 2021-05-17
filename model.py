import os
import shutil
from argparse import ArgumentParser

import pytorch_lightning as pl
import torch
import torchvision
from pytorch_lightning import Trainer, seed_everything
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import maskrcnn_resnet50_fpn

import utils
import voc_eval
from data import BedsoreDataModule, BedsoreLMDBDataModule
from logger import MLFlowLogger2
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.anchor_utils import AnchorGenerator


class MyFasterRCNN(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        if self.hparams.backbone == "mobilenet":
            if self.hparams.backbone_ckpt != "":
                backbone = MocoV2.load_from_checkpoint(
                    self.hparams.backbone_ckpt, strict=False)
                backbone = backbone.encoder_q
                backbone = backbone.features
            else:
                backbone = torchvision.models.mobilenet_v2(
                    pretrained=True).features

            backbone.out_channels = 1280
            anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256,
                                                       512), ),
                                               aspect_ratios=((0.5, 1.0,
                                                               2.0), ))

            roi_pooler = torchvision.ops.MultiScaleRoIAlign(
                featmap_names=["0"], output_size=7, sampling_ratio=2)
            mask_roi_pooler = torchvision.ops.MultiScaleRoIAlign(
                featmap_names=["0"], output_size=14, sampling_ratio=2)
            self.net = MaskRCNN(
                backbone,
                num_classes=10,
                min_size=400,
                max_size=800,
                rpn_anchor_generator=anchor_generator,
                box_roi_pool=roi_pooler,
                mask_roi_pool=mask_roi_pooler,
            )
        elif self.hparams.backbone == "resnet50":
            if self.hparams.backbone_ckpt != "":
                backbone = MocoV2.load_from_checkpoint(
                    self.hparams.backbone_ckpt, strict=False)
                backbone = backbone.encoder_q
                backbone = torch.nn.Sequential(*(
                    list(backbone.children())[:-2]))
                backbone.out_channels = 2048
                anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256,
                                                           512), ),
                                                   aspect_ratios=((0.5, 1.0,
                                                                   2.0), ))

                roi_pooler = torchvision.ops.MultiScaleRoIAlign(
                    featmap_names=["0"], output_size=7, sampling_ratio=2)
                mask_roi_pooler = torchvision.ops.MultiScaleRoIAlign(
                    featmap_names=["0"], output_size=14, sampling_ratio=2)
                self.net = MaskRCNN(
                    backbone,
                    num_classes=10,
                    min_size=400,
                    max_size=800,
                    rpn_anchor_generator=anchor_generator,
                    box_roi_pool=roi_pooler,
                    mask_roi_pool=mask_roi_pooler,
                )
            else:
                self.net = maskrcnn_resnet50_fpn(
                    pretrained=True,
                    trainable_backbone_layers=self.hparams.train_layers,
                    min_size=400,
                    max_size=800,
                )
                in_features = self.net.roi_heads.box_predictor.cls_score.in_features
                self.net.roi_heads.box_predictor = FastRCNNPredictor(
                    in_features, self.hparams.num_classes)

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
        result.log_dict({"train_loss": loss}, prog_bar=True)
        return result

    #  def validation_step(self, batch, batch_idx):
    #  self.net.train()  # net output loss when set train,\
    # net output label and bbox when set eval
    #  images, targets = batch
    #  loss_dict = self(images, targets)
    #  loss = sum(loss for loss in loss_dict.values())
    #  result = pl.EvalResult(checkpoint_on=loss)
    #  result.log_dict({'loss/valid_loss': loss}, prog_bar=True)
    #  return result
    def cal_ap(self, outputs):
        outputs = list(zip(*outputs))
        fnamelist = outputs[1]
        outputs = outputs[0]
        detfiles = []
        for i in outputs:
            for j in i:
                detfiles += j
        class_name = ["1期", "2期", "3期", "4期", "不可分期", "深部组织损伤"]
        mAP = []
        class_ap = []
        for i in class_name:
            ap = voc_eval.voc_eval(
                detfiles,
                "data/VOCdevkit/VOC2007/Annotations/{}.xml",
                fnamelist,
                i,
                ovthresh=0.5,
                use_07_metric=True,
            )[-1]
            class_ap.append([i, ap])
            mAP.append(ap)
        mAP = sum(mAP) / len(mAP)
        return mAP, class_ap

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        self.net.eval()
        image, target = batch
        output = self(image, target)
        detfile = []
        detfile.append(utils.out2detfile(target[0], output[0]))
        return detfile, target[0]["fname"]

    def validation_epoch_end(self, outputs):
        mAP, class_ap = self.cal_ap(outputs)
        result = pl.EvalResult(checkpoint_on=-torch.FloatTensor([mAP]))
        result.log_dict({"valid_map": mAP}, prog_bar=True)
        return result

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        self.net.eval()
        image, target = batch
        output = self(image, target)
        detfile = []
        detfile.append(utils.out2detfile(target[0], output[0]))
        return detfile, target[0]["fname"]

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
        parser = ArgumentParser(parents=[parent_parser],
                                add_help=False,
                                conflict_handler="resolve")
        parser.add_argument("--exp", type=str, default="Default")
        parser.add_argument("--name", type=str, default="test")
        parser.add_argument("--chunk_num", type=int, default=10)
        parser.add_argument("--chunk_id", type=int, default=0)
        parser.add_argument("--lr", type=float, default=1e-4)
        parser.add_argument("--num_classes", type=int, default=10)
        parser.add_argument("--batch_size", type=int, default=4)
        parser.add_argument("--num_valid", type=int, default=100)
        parser.add_argument("--num_workers", type=int, default=8)
        parser.add_argument("--seed", type=int, default=32)
        parser.add_argument("--data_root", type=str, default="data")
        parser.add_argument("--distributed_backend", type=str, default="dp")
        parser.add_argument("--amp_level", type=str, default="O2")
        parser.add_argument("--train_layers", type=int, default=3)
        parser.add_argument("--trans_prob", type=float, default=0.5)
        parser.add_argument("--max_epochs", type=int, default=60)
        parser.add_argument("--test_ckpt", type=str, default="")
        parser.add_argument("--backbone", type=str, default="mobilenet")
        parser.add_argument("--backbone_ckpt", type=str, default="")
        return parser


def main():
    parser = ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    parser = MyFasterRCNN.add_model_specific_args(parser)
    args = parser.parse_args()
    seed_everything(args.seed)
    model = MyFasterRCNN(args)
    dm = BedsoreLMDBDataModule(root=args.data_root,
                               batch_size=args.batch_size,
                               trans_prob=args.trans_prob,
                               chunk_num=args.chunk_num,
                               chunk_id=args.chunk_id,
                               num_workers=args.num_workers,
                               seed=args.seed)
    trainer = Trainer.from_argparse_args(args)

    if args.test_ckpt != "":  # test
        model = MyFasterRCNN.load_from_checkpoint(args.test_ckpt)
        trainer.test(model, test_dataloaders=dm.test_dataloader())
    else:  # train
        if args.name != "test":
            logger = MLFlowLogger2(experiment_name=args.exp,
                                   run_name=args.name)
            # save source files to mlflow
            save_dir = f"mlruns/{logger.experiment_id}/{logger.run_id}/artifacts"
            save_files = [i for i in os.listdir() if ".py" in i]
            for i in save_files:
                shutil.copy(i, f"{save_dir}/{i}")
        else:
            logger = None
        trainer.logger = logger
        trainer.fit(model, dm)
        trainer.test()


if __name__ == "__main__":
    main()

# python model.py --gpus '1,' --max_epochs 50 --name test
