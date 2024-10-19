from typing import Any
import torch.nn as nn
import torch
import timm
import numpy as np
from timm.models.swin_transformer import SwinTransformer
import pytorch_lightning as pl
from einops.einops import rearrange


class SwinV1Encoder(pl.LightningModule):
    def __init__(self,
                 d_model,
                 requires_grad=True,

                 drop_rate=0.1,
                 proj_drop_rate=0.1,
                 attn_drop_rate=0.1,
                 drop_path_rate=0.1,
                 ):
        super().__init__()
        swinv1_state_dict = (timm.create_model('swin_tiny_patch4_window7_224',
                                               pretrained=True)
                             .state_dict())
        self.swinv1 = SwinTransformer(
            drop_rate=drop_rate,
            proj_drop_rate=proj_drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
        )
        self.swinv1.load_state_dict(swinv1_state_dict)
        self.swinv1.head = nn.Identity()

        for param in self.swinv1.patch_embed.parameters():
            param.requires_grad = False
        for param in self.swinv1.layers[0].parameters():
            param.requires_grad = False
        for param in self.swinv1.layers[1].parameters():
            param.requires_grad = False
        for param in self.swinv1.layers[2].parameters():
            param.requires_grad = False
        for param in self.swinv1.layers[3].parameters():
            param.requires_grad = requires_grad

        self.out = nn.Sequential(
            nn.Linear(768, d_model, bias=False),
            nn.ReLU(),
            nn.LayerNorm(d_model),
        )




    def forward(self, img):
        x = self.swinv1(img)
        x = self.out(x)
        x = rearrange(x, "b h w d-> b d h w")

        return x

if __name__ == '__main__':
    x = torch.randn(1, 3, 224, 224)
    model = SwinV1Encoder(d_model=256)
    out = model(x)
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))
    print(out.shape)
