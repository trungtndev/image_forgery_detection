from typing import Any

import pytorch_lightning as pl
import torch
from einops import rearrange

from torch import nn


class FeedForward(nn.Module):
    def __init__(self, input_size, mlp_ratio, dropout_rate):
        super(FeedForward, self).__init__()
        self.module = nn.Sequential(
            nn.Linear(input_size, input_size * mlp_ratio),
            nn.LayerNorm(input_size * mlp_ratio),
            nn.GELU(),
            nn.Dropout(dropout_rate),

            nn.Linear(input_size * mlp_ratio, input_size)
        )

    def forward(self, x):
        return self.module(x)

class Classifer(nn.Module):
    def __init__(self, input_size, dropout_rate, num_classes):
        super(Classifer, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(input_size, input_size*2)
        self.bn = nn.LayerNorm(input_size*2)
        self.output = nn.Linear(input_size*2, num_classes)

    def forward(self, x):
        x = rearrange(x, 'b w h d -> b (w h) d')
        x = x.mean(dim=1)
        x = self.dropout(x)
        x = self.fc(x)
        x = self.bn(x)
        x = nn.functional.gelu(x)
        x = self.output(x)
        return x

class Fusion(pl.LightningModule):
    def __init__(self, d_model: int,
                 dropout: float,
                 num_classes: int,
                 ):
        super(Fusion, self).__init__()
        self.fc = nn.Linear(d_model * 2, d_model)
        self.sigmoid = nn.Sigmoid()

        self.classifier = Classifer(d_model, dropout, num_classes)

    def forward(self, feature_1, feature_2):
        out = torch.cat((feature_1, feature_2), dim=-1)
        out = self.fc(out)
        attn = self.sigmoid(out)

        out = feature_1 * attn + feature_2 * (1 - attn)

        out = torch.nn.functional.gelu(out)

        out = self.classifier(out)

        return out


if __name__ == '__main__':
    fusion = Fusion(256, 0.2, 2)
    feature_1 = torch.randn(10, 7, 7, 256)
    feature_2 = torch.randn(10, 7, 7, 256)
    output = fusion(feature_1, feature_2)
    print(output.shape)
    print(output)
