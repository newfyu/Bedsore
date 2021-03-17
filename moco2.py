import copy
import shutil
import random
from argparse import ArgumentParser
from typing import Union

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchvision
from PIL import ImageFilter
#  from pl_bolts.datamodules.ssl_imagenet_datamodule import SSLImagenetDataModule
from pl_bolts.metrics import mean, precision_at_k
from pl_bolts.transforms.dataset_normalizations import imagenet_normalization
from torch import nn
from torchvision import transforms
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, random_split
import os
from PIL import Image
from logger import MLFlowLogger2


class SSLImageDataset(torch.utils.data.Dataset):

    def __init__(self, root, transforms, suffix='.jpg', test_list_path='data/VOCdevkit/VOC2007/ImageSets/Main/val.txt'):
        self.root = root
        self.transforms = transforms
        self.imglist = [i for i in os.listdir(root) if suffix in i]
        with open(test_list_path,'r') as f:
            test_imglist = f.readlines()
        test_imglist = [i.replace('\n','') +'.jpg' for i in test_imglist]
        self.imglist = list(set(self.imglist) - set(test_imglist)) # exclude test sample
            

    def __len__(self):
        return len(self.imglist)

    def __getitem__(self, idx):
        img_name = self.imglist[idx]
        img = Image.open(f"{self.root}/{img_name}").convert('RGB')
        img1, img2 = self.transforms(img)
        return img1, img2


class SSLImageDataModule(LightningDataModule):

    def __init__(self, root, batch_size, num_valid, img_size=400, seed=32, **kwargs):
        super().__init__()
        ds = SSLImageDataset(root, transforms=Moco2TrainImagenetTransforms(img_size))
        self.batch_size = batch_size
        self.train_ds, self.valid_ds = torch.utils.data.random_split(
            ds, [len(ds) - num_valid, num_valid], generator=torch.Generator().manual_seed(seed))
        self.valid_ds = copy.deepcopy(self.valid_ds)
        self.valid_ds.dataset.transforms = Moco2EvalImagenetTransforms(img_size)  # 如果验证集要调整transformer

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=False, num_workers=8, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.valid_ds, batch_size=self.batch_size, shuffle=False, num_workers=8)


class Moco2TrainImagenetTransforms:
    """
    Moco 2 augmentation:
    https://arxiv.org/pdf/2003.04297.pdf

    """

    def __init__(self, height=128):
        # image augmentation functions
        self.train_transform = transforms.Compose([
            transforms.RandomResizedCrop(height, scale=(0.2, 1.)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            imagenet_normalization()
        ])

    def __call__(self, inp):
        q = self.train_transform(inp)
        k = self.train_transform(inp)
        return q, k


class Moco2EvalImagenetTransforms:
    """
    Moco 2 augmentation:
    https://arxiv.org/pdf/2003.04297.pdf

    """

    def __init__(self, height=128):
        self.test_transform = transforms.Compose([
            transforms.Resize(height + 32),
            transforms.CenterCrop(height),
            transforms.ToTensor(),
            imagenet_normalization(),
        ])

    def __call__(self, inp):
        q = self.test_transform(inp)
        k = self.test_transform(inp)
        return q, k


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=(0.1, 2.0)):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


class MocoV2(pl.LightningModule):

    def __init__(self,
                 base_encoder: Union[str, torch.nn.Module] = 'resnet18',
                 emb_dim: int = 128,
                 num_negatives: int = 65536,
                 encoder_momentum: float = 0.999,
                 softmax_temperature: float = 0.07,
                 learning_rate: float = 0.03,
                 momentum: float = 0.9,
                 weight_decay: float = 1e-4,
                 datamodule: pl.LightningDataModule = None,
                 batch_size: int = 256,
                 use_mlp: bool = False,
                 num_workers: int = 8,
                 *args, **kwargs):
        """
        Args:
            base_encoder: torchvision model name or torch.nn.Module
            emb_dim: feature dimension (default: 128)
            num_negatives: queue size; number of negative keys (default: 65536)
            encoder_momentum: moco momentum of updating key encoder (default: 0.999)
            softmax_temperature: softmax temperature (default: 0.07)
            learning_rate: the learning rate
            momentum: optimizer momentum
            weight_decay: optimizer weight decay
            datamodule: the DataModule (train, val, test dataloaders)
            batch_size: batch size
            use_mlp: add an mlp to the encoders
            num_workers: workers for the loaders
        """

        super().__init__()
        self.save_hyperparameters()

        self.datamodule = datamodule
        # create the encoders
        # num_classes is the output fc dimension
        self.encoder_q, self.encoder_k = self.init_encoders(base_encoder)

        if use_mlp:  # hack: brute-force replacement
            dim_mlp = self.encoder_q.fc.weight.shape[1]
            self.encoder_q.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_q.fc)
            self.encoder_k.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_k.fc)

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(emb_dim, num_negatives))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    def init_encoders(self, base_encoder):
        """
        Override to add your own encoders
        """

        #  backbone = torchvision.models.mobilenet_v2(pretrained=True).features
        template_model = getattr(torchvision.models, base_encoder)
        encoder_q = template_model(num_classes=self.hparams.emb_dim)
        encoder_k = template_model(num_classes=self.hparams.emb_dim)

        return encoder_q, encoder_k

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            em = self.hparams.encoder_momentum
            param_k.data = param_k.data * em + param_q.data * (1. - em)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.hparams.num_negatives % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.hparams.num_negatives  # move pointer

        self.queue_ptr[0] = ptr

    def forward(self, img_q, img_k):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """

        # compute query features
        q = self.encoder_q(img_q)  # queries: NxC
        q = nn.functional.normalize(q, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            k = self.encoder_k(img_k)  # keys: NxC
            k = nn.functional.normalize(k, dim=1)

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.hparams.softmax_temperature

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long)
        labels = labels.type_as(logits)

        # dequeue and enqueue
        self._dequeue_and_enqueue(k)

        return logits, labels

    def training_step(self, batch, batch_idx):
        img_1, img_2 = batch

        output, target = self(img_q=img_1, img_k=img_2)
        loss = F.cross_entropy(output.float(), target.long())

        acc1, acc5 = precision_at_k(output, target, top_k=(1, 5))

        result = pl.TrainResult(minimize=loss)
        result.log_dict({
            'train_loss': loss,
            'train_acc1': acc1,
            'train_acc5': acc5
        },prog_bar=True)
        return result

    def validation_step(self, batch, batch_idx):
        img_1, img_2 = batch

        output, target = self(img_q=img_1, img_k=img_2)
        loss = F.cross_entropy(output, target.long())

        acc1, acc5 = precision_at_k(output, target, top_k=(1, 5))

        results = {
            'val_loss': loss,
            'val_acc1': acc1,
            'val_acc5': acc5
        }
        return results

    def validation_epoch_end(self, outputs):
        val_loss = mean(outputs, 'val_loss')
        val_acc1 = mean(outputs, 'val_acc1')
        val_acc5 = mean(outputs, 'val_acc5')
        result = pl.EvalResult(checkpoint_on=val_loss)
        result.log_dict({
            'val_loss': val_loss,
            'val_acc1': val_acc1,
            'val_acc5': val_acc5
        },prog_bar=True)
        return result

    def configure_optimizers(self):
        #  optimizer = torch.optim.SGD(self.parameters(), self.hparams.lr,
                                    #  momentum=self.hparams.momentum,
                                    #  weight_decay=self.hparams.weight_decay)
        optimizer = torch.optim.Adam(self.parameters(), self.hparams.lr, weight_decay=self.hparams.weight_decay)
        return optimizer

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False, conflict_handler="resolve")
        parser.add_argument("--exp", type=str, default="pretrain")
        parser.add_argument("--name", type=str, default="test")
        parser.add_argument('--base_encoder', type=str, default='mobilenet_v2')
        parser.add_argument('--root', type=str, default="data/VOCdevkit/VOC2007/JPEGImages")
        parser.add_argument('--img_size', type=int, default=128)
        parser.add_argument('--emb_dim', type=int, default=128)
        parser.add_argument('--emb_dim', type=int, default=128)
        parser.add_argument('--num_workers', type=int, default=8)
        parser.add_argument('--num_negatives', type=int, default=1024) # 65536
        parser.add_argument('--encoder_momentum', type=float, default=0.999)
        parser.add_argument('--softmax_temperature', type=float, default=0.07)
        parser.add_argument('--lr', type=float, default=1e-4)
        parser.add_argument('--momentum', type=float, default=0.9)
        parser.add_argument('--weight_decay', type=float, default=1e-4)
        parser.add_argument('--batch_size', type=int, default=128)
        parser.add_argument('--num_valid', type=int, default=256)
        parser.add_argument('--use_mlp', action='store_true')

        return parser


def main():
    parser = ArgumentParser()

    # trainer args
    parser = pl.Trainer.add_argparse_args(parser)

    # model args
    parser = MocoV2.add_model_specific_args(parser)
    args = parser.parse_args()

    datamodule = SSLImageDataModule(**args.__dict__)
    model = MocoV2(**args.__dict__, datamodule=datamodule)

    trainer = pl.Trainer.from_argparse_args(args)
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
    trainer.fit(model)


if __name__ == '__main__':
    main()

# python moco2.py --gpus 4
