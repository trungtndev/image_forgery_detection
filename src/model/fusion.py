from typing import Any

import pytorch_lightning as pl
import torch
from torch import nn
from timm.layers import ClassifierHead
from .cnn_gru import FeedForward

class AttentionFusionModule(nn.Module):
    def __init__(self, input_dim_feature_1, input_dim_feature_2, output_dim):
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
                 input_dim_feature_1: int,
                 input_dim_feature_2: int,
                 num_classes: int):
        super(Fusion, self).__init__()
        self.attention_fusion = AttentionFusionModule(input_dim_feature_1,
                                                      input_dim_feature_2,
                                                      d_model)
        self.mlp = FeedForward(d_model, 4, 0.1)
        self.norm = nn.LayerNorm(d_model)

        self.classifier = ClassifierHead(d_model, num_classes)

    def forward(self, feature_1, feature_2):
        fused_features = self.attention_fusion(feature_1, feature_2)
        x = self.mlp(fused_features)
        x = self.norm(x)

        x = self.classifier(x)
        return x
