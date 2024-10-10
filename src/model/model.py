

import pytorch_lightning as pl

class Model(pl.LightningModule):
    def __init__(self,
                 num_classes: int
                 ):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, x):
        pass

