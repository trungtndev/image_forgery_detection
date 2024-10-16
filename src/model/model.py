import pytorch_lightning as pl
from .spatial_module import SwinV1Encoder
from .fusion import Head
from .frequency_module import FrequencyModule
import torch

class Model(pl.LightningModule):
    def __init__(self,
                 num_classes: int,
                 d_model: int,

                 requires_grad: bool,
                 drop_rate: float,
                 proj_drop_rate: float,
                 attn_drop_rate: float,
                 drop_path_rate: float,

                 growth_rate: int,
                 num_layers: int,
                 ):
        super().__init__()
        self.num_classes = num_classes

        self.spatial = SwinV1Encoder(
            d_model=d_model,
            requires_grad=requires_grad,
            drop_rate=drop_rate,
            proj_drop_rate=proj_drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate
        )

        self.frequency = FrequencyModule(
            d_model=d_model,
            num_layers=num_layers,
            growth_rate=growth_rate,
        )

        self.head = Head(
            d_model=d_model,
            num_classes=num_classes,
            dropout_rate=drop_rate
        )

    def forward(self, spa, fre):
        x_1 = self.spatial(spa)
        x_2 = self.frequency(fre)
        x = self.head(x_1, x_2)
        return x

if __name__ == "__main__":
    spa = torch.randn(1, 3, 224, 224)
    fre = torch.randn(1, 3, 224, 224)
    model = Model(num_classes=2,
                  d_model=256,
                  requires_grad=True,
                  drop_rate=0.1,
                  proj_drop_rate=0.1,
                  attn_drop_rate=0.1, drop_path_rate=0.1,
                  growth_rate=48,
                  num_layers=8)
    print(model(spa, fre).shape)
    print(model)
    # print(sum(p.numel() for p in model.frequency.parameters() if p.requires_grad))
