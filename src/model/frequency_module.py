from typing import Any

import torch
from .cnn_gru import HybridCNNGRU
import pytorch_lightning as pl


class FrequencyModule(pl.LightningModule):
    def __init__(self,
                 d_model,
                 mlp_ratio=4,
                 input_channels=3,
                 hidden_size=128,
                 drop_rate=0.1,
                 ):
        super(FrequencyModule, self).__init__()
        self.module = HybridCNNGRU(
            d_model,
            mlp_ratio=mlp_ratio,
            input_channels=input_channels,
            hidden_size=hidden_size,
            drop_rate=drop_rate,
        )
    def forward(self, x) -> Any:
        return self.module(x)
