from typing import Any

import pytorch_lightning as pl
import torch
from einops import rearrange
import torch.nn.functional as F
from torch import nn


class ConvForward(nn.Module):
    def __init__(self, d_model: int, dropout_rate: float):
        super(ConvForward, self).__init__()
        self.conv1 = nn.Conv2d(d_model, d_model * 2, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(d_model * 2)
        self.drop = nn.Dropout(dropout_rate)
        self.ge = nn.SiLU()
        self.conv2 = nn.Conv2d(d_model * 2, d_model, kernel_size=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        x = self.ge(x)
        x = self.drop(x)
        x = self.conv2(x)
        return x

class FeedForward(nn.Module):
    def __init__(self, d_model: int):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_model * 2, bias=True)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(d_model * 2, d_model, bias=True)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class Classifer(pl.LightningModule):
    def __init__(self, input_size, num_classes, dropout_rate):
        super(Classifer, self).__init__()
        self.pool = nn.MaxPool2d((7, 7))
        self.flatten = nn.Flatten()
        # self.ffd = FeedForward(input_size)
        self.act = nn.ReLU()
        self.fc = nn.Linear(input_size, num_classes)

    def forward(self, x):
        out = self.pool(x)
        out = self.flatten(out)
        # out = out + self.ffd(out)
        out = self.act(out)
        out = self.fc(out)
        return out


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
        self.classifier = Classifer(d_model, num_classes, dropout_rate)

    def forward(self, x):
        return self.classifier(x)


if __name__ == '__main__':
    fusion = Head(256, 2, 0.2)
    feature_1 = torch.randn(10, 256, 7, 7)
    feature_2 = torch.randn(10, 256, 7, 7)
    output = fusion(feature_1, feature_2)
    print(output.shape)
    print(output)
