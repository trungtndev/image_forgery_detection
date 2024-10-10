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
        x = self.dct(x, norm='ortho')
        return x


class PatchExtractor(nn.Module):
    def __init__(self, patch_size, num_patches):
        super(PatchExtractor, self).__init__()
        self.patch_size = patch_size
        self.num_patches = num_patches

    def forward(self, x):
        x = rearrange(x, 'b c (h p1) (w p2) -> b (h w) (c p1 p2)',
                      p1=self.num_patches[0],
                      p2=self.num_patches[1])
        return x


class CNN(nn.Module):
    def __init__(self, input_channels):
        super(CNN, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.AvgPool2d(kernel_size=2, stride=2, padding=1),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.AvgPool2d(kernel_size=2, stride=2, padding=1),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.AvgPool2d(kernel_size=2, stride=2, padding=1),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.GELU(),
            nn.AvgPool2d(kernel_size=2, stride=1, padding=0),

            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.GELU(),
        )

    def forward(self, x):
        x = self.cnn(x)
        return x


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



class HybridCNNGRU(nn.Module):
    def __init__(self,
                 d_model,
                 patch_size=7,
                 input_channels=3,
                 hidden_size=128,
                 drop_rate=0.1,
                 ):
        super(HybridCNNGRU, self).__init__()
        image_size = 28

        self.patch_size = patch_size
        self.image_size = image_size
        assert image_size % patch_size == 0, 'image size must be divisible by patch size'
        n_patch = (image_size // patch_size)
        self.num_patches = (n_patch, n_patch)

        self.cnn = nn.Sequential(
            ImageToFrequency(),
            CNN(input_channels),
        )
        self.patch_emb = nn.Sequential(
            PatchExtractor(self.patch_size, self.num_patches),
            nn.Linear(in_features=8192, out_features=d_model),
            nn.LayerNorm(d_model),
            nn.SiLU(),
            nn.Dropout(drop_rate),
        )
        self.gru = nn.Sequential(
            GRUBlock(input_size=d_model, hidden_size=hidden_size),
            nn.LayerNorm(hidden_size),
        )

    def forward(self, x):
        out = self.cnn(x)
        out = self.patch_emb(out)

        out = self.gru(out)

        out = rearrange(out, 'b (w h) d -> b w h d',
                      w=self.patch_size,
                      h=self.patch_size)
        return out

