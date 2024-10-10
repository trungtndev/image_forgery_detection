import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from typing import Tuple
from torchmetrics import Accuracy
from torch.optim.lr_scheduler import _LRScheduler
import math
from einops import rearrange


from .model.spatial_module import SwinV1Encoder
from .model.fusion import Fusion
from .model.cnn_gru import HybridCNNGRU

class WarmupCosineAnnealingLR(_LRScheduler):
    def __init__(self, optimizer, warmup_epochs, total_epochs, eta_min=0, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            return [base_lr * (self.last_epoch + 1) / self.warmup_epochs for base_lr in self.base_lrs]
        else:
            return [self.eta_min + (base_lr - self.eta_min)
                    * 0.5
                    * (1 + math.cos(math.pi * (self.last_epoch - self.warmup_epochs)
                                    / (self.total_epochs - self.warmup_epochs)))
                    for base_lr in self.base_lrs]


class LitModel(pl.LightningModule):
    def __init__(self,
                 num_classes: int,
                 d_model: int,
                 pretrain: bool,
                 requires_grad: bool,
                 drop_rate: float,
                 proj_drop_rate: float,
                 attn_drop_rate: float,
                 drop_path_rate: float,

                 hidden_size: int,
                 image_size: int,
                 patch_size: int,

                 # training
                 learning_rate: float,
                 weight_decay: float,
                 patience: int,
                 ):
        super().__init__()
        self.spatial = SwinV1Encoder(
            pretrain=pretrain,
            requires_grad=requires_grad,
            drop_rate=drop_rate,
            proj_drop_rate=proj_drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate
        )

        self.ferquency = HybridCNNGRU(
            d_model=d_model,
            patch_size=patch_size,
            hidden_size=hidden_size,
            input_channels=3,
            drop_rate=drop_rate,
        )

        self.fusion = Fusion(
            d_model=d_model,
            mlp_ratio=4,
            dropout=drop_rate,
            num_classes=num_classes,
            input_dim_feature_1=768,
            input_dim_feature_2=hidden_size,
        )

        self.train_accuracy = Accuracy(task='multiclass', num_classes=2)
        self.val_accuracy = Accuracy(task='multiclass', num_classes=2)
        self.test_accuracy = Accuracy(task='multiclass', num_classes=2)
        self.save_hyperparameters()

    def forward(self, img):
        x_1 = self.spatial(img)
        x_2 = self.ferquency(img)

        x = self.fusion(x_1, x_2)
        return x

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(),
                                     lr=self.hparams.learning_rate,
                                     weight_decay=self.hparams.weight_decay)
        # scheduler_warmup = WarmupCosineAnnealingLR(optimizer, warmup_epochs=5, total_epochs=50)
        lateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            'max',
            factor=0.25,
            patience=self.hparams.patience // self.trainer.check_val_every_n_epoch,
        )
        scheduler = {
            "scheduler": lateau_scheduler,
            "monitor": "val_acc",
            "interval": "epoch",
            "frequency": self.trainer.check_val_every_n_epoch,
            "strict": True,
        }
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)

        loss = self.compute_loss(outputs, labels)
        self.train_accuracy(outputs, labels)

        self.log('train_loss', loss,
                 prog_bar=True,
                 sync_dist=True)
        self.log('train_acc', self.train_accuracy,
                 prog_bar=True,
                 sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)

        loss = self.compute_loss(outputs, labels)
        self.val_accuracy(outputs, labels)

        self.log('val_loss', loss,
                 prog_bar=True,
                 on_epoch=True,
                 sync_dist=True)
        self.log('val_acc', self.val_accuracy,
                 prog_bar=True,
                 on_epoch=True,
                 sync_dist=True)

    def test_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)

        loss = self.compute_loss(outputs, labels)
        self.test_accuracy(outputs, labels)

        self.log('test_loss', loss,
                 on_epoch=True,
                 sync_dist=True)
        self.log('test_acc', self.test_accuracy,
                 on_epoch=True,
                 sync_dist=True)

    def compute_loss(self, outputs, labels):
        criterion = torch.nn.CrossEntropyLoss()
        return criterion(outputs, labels)
