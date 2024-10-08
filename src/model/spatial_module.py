from typing import Any

import torch
import timm
import numpy as np
from timm.models.swin_transformer import SwinTransformer
import pytorch_lightning as pl


class SwinV1Encoder(pl.LightningModule):
    def __init__(self,
                 d_model: int,
                 pretrain: bool,
                 requires_grad=True,

                 drop_rate=0.1,
                 proj_drop_rate=0.1,
                 attn_drop_rate=0.1,
                 drop_path_rate=0.1,
                 ):
        super().__init__()
        swinv1_state_dict = timm.create_model('swin_tiny_patch4_window7_224', pretrained=pretrain).state_dict()
        self.swinv1 = SwinTransformer(
            drop_rate=drop_rate,
            proj_drop_rate=proj_drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
        )

        self.swinv1.load_state_dict(swinv1_state_dict)

        if pretrain:
            for param in self.swinv1.parameters():
                param.requires_grad = requires_grad


        # add a new head
        # self.swinv1.head = torch.nn.Sequential(
        #     torch.nn.Linear(256, d_model),
        #     torch.nn.LayerNorm(d_model),
        #     torch.nn.GELU(),
        #     torch.nn.Dropout(drop_rate),
        # )
        self.swinv1.head.fc = torch.nn.Linear(768, 2, bias=True)

        # make sure the head is trainable
        for param in self.swinv1.head.parameters():
            param.requires_grad = True

    def forward(self, img):
        x = self.swinv1(img)
        return x
