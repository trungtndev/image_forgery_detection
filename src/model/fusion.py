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

class AttentionFusionModule(nn.Module):
    def __init__(self,
                 input_dim_feature_1,
                 input_dim_feature_2,
                 output_dim):
        super(AttentionFusionModule, self).__init__()

        self.fc1 = nn.Sequential(
            nn.Linear(input_dim_feature_1, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(),
        )

        self.fc2 = nn.Sequential(
            nn.Linear(input_dim_feature_2, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(),
        )

        self.attention = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.Tanh(),
            nn.Linear(output_dim, 1),
            nn.Softmax(dim=1)
        )

    def forward(self, feature_1, feature_2):
        out1 = self.fc1(feature_1)
        out2 = self.fc2(feature_2)

        attn_weights1 = self.attention(out1)
        attn_weights2 = self.attention(out2)

        fused = attn_weights1 * out1 + attn_weights2 * out2
        return fused

class Fusion(pl.LightningModule):
    def __init__(self, d_model: int,
                 mlp_ratio: int,
                 dropout: float,
                 num_classes: int,
                 input_dim_feature_1: int,
                 input_dim_feature_2: int,
                 ):
        super(Fusion, self).__init__()
        self.attention_fusion = AttentionFusionModule(input_dim_feature_1,
                                                      input_dim_feature_2,
                                                      d_model)
        self.mlp = FeedForward(d_model, mlp_ratio, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, feature_1, feature_2):
        feature_1 = rearrange(feature_1, 'b h w c -> b (h w) c')
        feature_2 = rearrange(feature_2, 'b h w c -> b (h w) c')

        fused_features = self.attention_fusion(feature_1, feature_2)
        x = fused_features + self.mlp(fused_features)
        x = self.norm2(x)
        x = x.mean(dim=1)
        x = self.classifier(x)
        return x
