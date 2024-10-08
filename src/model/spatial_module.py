from typing import Any

import torch
import timm
import numpy as np
from timm.models.swin_transformer import SwinTransformer
import pytorch_lightning as pl


class SwinV1Encoder(pl.LightningModule):
    def __init__(self,
                 d_model: int,
                 requires_grad=True,

                 img_size: int = 256,
                 patch_size: int = 8,
                 in_chans: int = 1,
                 embed_dim: int = 64,
                 depths: tuple = (2, 2, 8, 2),
                 num_heads: tuple = (3, 6, 8, 12),
                 window_size: tuple = (12, 7, 6, 4),
                 mlp_ratio: int = 4,
                 drop_rate=0.1,
                 proj_drop_rate=0.1,
                 attn_drop_rate=0.1,
                 drop_path_rate=0.1,
                 ):
        super().__init__()
        self.swinv1 = SwinTransformer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,

            drop_rate=drop_rate,
            proj_drop_rate=proj_drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,

            num_classes=2,
        )

        # self.swinv1.head = torch.nn.Sequential(
        #     torch.nn.Linear(256, d_model),
        #     torch.nn.LayerNorm(d_model),
        #     torch.nn.GELU(),
        #     torch.nn.Dropout(drop_rate),
        # )
    def forward(self, img):
        x = self.swinv1(img)
        return x
