from typing import Any

import torch
from .densenet import Encoder
import pytorch_lightning as pl


class FrequencyModule(pl.LightningModule):
    def __init__(self,
                 d_model,
                 growth_rate,
                 num_layers,

                 ):
        super(FrequencyModule, self).__init__()
        self.fre = Encoder(
            d_model=d_model,
            growth_rate=growth_rate,
            num_layers=num_layers,
        )

    def forward(self, x) -> Any:
        return self.fre(x)
