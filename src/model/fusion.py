from typing import Any

import pytorch_lightning as pl
import torch
from einops import rearrange
import torch.nn.functional as F
from torch import nn


class ConvForward(nn.Module):
    def __init__(self, d_model: int, dropout_rate: float):
        super(ConvForward, self).__init__()
        self.conv1 = nn.Conv2d(d_model, d_model * 2, kernel_size=1)
        self.bn = nn.BatchNorm2d(d_model * 2)
        self.drop = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(d_model * 2, d_model, kernel_size=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.drop(x)
        x = self.conv2(x)
        return x


class Classifer(nn.Module):
    def __init__(self, input_size, num_classes):
        super(Classifer, self).__init__()
        # self.pooling = nn.AdaptiveMaxPool2d(1)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(input_size, num_classes)

    def forward(self, x):
        kernel_size = x.size()[2:]
        x = F.avg_pool2d(x, kernel_size)
        x = self.flatten(x)
        x = self.fc(x)
        return x


class Fusion(nn.Module):
    def __init__(self, d_model: int):
        super(Fusion, self).__init__()
        self.conv = nn.Conv2d(d_model * 2, d_model, kernel_size=1)
        self.bn = nn.BatchNorm2d(d_model)
        self.sigmoid = nn.Sigmoid()

    def forward(self, feature_1, feature_2):
        out = torch.cat((feature_1, feature_2), dim=1)
        out = self.conv(out)
        out = self.bn(out)
        attn = self.sigmoid(out)
        out = feature_1 * attn + feature_2 * (1 - attn)
        return out


class Head(nn.Module):
    def __init__(self, d_model: int, num_classes: int, dropout_rate: float):
        super(Head, self).__init__()
        self.fusion = Fusion(d_model)
        self.conv = ConvForward(d_model, dropout_rate)
        self.classifier = Classifer(d_model, num_classes)

    def forward(self, x1, x2):
        x = self.fusion(x1, x2)
        x = x + self.conv(x)
        return self.classifier(x)


if __name__ == '__main__':
    fusion = Head(256, 2, 0.2)
    feature_1 = torch.randn(10, 256, 7, 7)
    feature_2 = torch.randn(10, 256, 7, 7)
    output = fusion(feature_1, feature_2)
    print(output.shape)
    print(output)
