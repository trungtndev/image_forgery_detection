import torch
from torch.nn import functional as F
import torch.nn as nn
from einops import rearrange

from torch_dct import dct_2d


class ImageToFrequency(nn.Module):
    def __init__(self):
        super(ImageToFrequency, self).__init__()
        self.dct = dct_2d

    def forward(self, x):
        x = self.dct(x)
        return x


class PatchExtractor(nn.Module):
    def __init__(self, patch_size):
        super(PatchExtractor, self).__init__()
        self.patch_size = patch_size

    def forward(self, x):
        x = rearrange(x, 'b c (h p1) (w p2) -> b (h w) (c p1 p2)',
                      p1=self.patch_size, p2=self.patch_size)
        return x


class CNN(nn.Module):
    def __init__(self, input_channels):
        super(CNN, self).__init__()
        self.module = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.AvgPool2d(kernel_size=3, stride=1),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.AvgPool2d(kernel_size=3, stride=1),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.AvgPool2d(kernel_size=3, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.GELU(),
            nn.AvgPool2d(kernel_size=3, stride=1),

            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.GELU(),
            nn.AvgPool2d(kernel_size=3, stride=1),

        )

    def forward(self, x):
        return self.module(x)


class GRUBlock(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(GRUBlock, self).__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRUCell(input_size, hidden_size)

    def forward(self, x):
        # Assuming x has shape (batch_size, sequence_length, input_size)
        batch_size, sequence_length, _ = x.size()
        h = torch.zeros(batch_size, self.hidden_size, device=x.device)

        # Iterate through the sequence
        outputs = []
        for i in range(sequence_length):
            h = self.gru(x[:, i, :], h)
            outputs.append(h.unsqueeze(1))

        # Concatenate along the sequence dimension
        outputs = torch.cat(outputs, dim=1)
        return outputs

class MLP(nn.Module):
    def __init__(self, input_size, mlp_ratio, dropout_rate):
        super(MLP, self).__init__()
        self.module = nn.Sequential(
            nn.Linear(input_size, input_size*mlp_ratio),
            nn.LayerNorm(input_size*mlp_ratio),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(input_size*mlp_ratio, input_size)
        )

    def forward(self, x):
        return self.module(x) + x

class HybridCNNGRU(nn.Module):
    def __init__(self,
                 d_model,
                 mlp_ratio=4,
                 input_channels=3,
                 hidden_size=128,
                 drop_rate=0.1,
                 ):
        super(HybridCNNGRU, self).__init__()

        self.module = nn.Sequential(
            ImageToFrequency(),
            CNN(input_channels),
            PatchExtractor(7),
            nn.LazyLinear(d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(drop_rate),

            GRUBlock(input_size=d_model, hidden_size=hidden_size),
            nn.LayerNorm(hidden_size),
            nn.Dropout(drop_rate),

            nn.Linear(hidden_size, d_model),
        )

    def forward(self, x):
        x = self.module(x)
        return x
