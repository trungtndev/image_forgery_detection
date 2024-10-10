from typing import Any
import torch.nn as nn
import torch
import timm
import numpy as np
from timm.models.swin_transformer import SwinTransformer
import pytorch_lightning as pl


class SwinV1Encoder(pl.LightningModule):
    def __init__(self,
                 pretrain: bool,
                 requires_grad=True,

                 drop_rate=0.1,
                 proj_drop_rate=0.1,
                 attn_drop_rate=0.1,
                 drop_path_rate=0.1,
                 ):
        super().__init__()
        swinv1_state_dict = (timm.create_model('swin_tiny_patch4_window7_224', pretrained=pretrain)
                             .state_dict())
        self.swinv1 = SwinTransformer(
            drop_rate=drop_rate,
            proj_drop_rate=proj_drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
        )

        self.swinv1.load_state_dict(swinv1_state_dict)

        for param in self.parameters():
            param.requires_grad = False

        self.swinv1.head = nn.Identity()


    def forward(self, img):
        x = self.swinv1(img)
        return x
