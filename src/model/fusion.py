import pytorch_lightning as pl
import torch
from torch import nn
from .cbam import CBAM

class FeedForward(nn.Module):
    def __init__(self, d_model: int, dropout_rate: float):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_model * 2, bias=True)
        self.fc2 = nn.Linear(d_model, d_model * 2, bias=True)
        self.fc3 = nn.Linear(d_model * 2, d_model, bias=True)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.act = nn.SiLU()


    def forward(self, x):
        x1 = self.fc1(x)
        x2 = self.fc2(x)
        x = self.act(x1) * x2
        x = self.dropout1(x)
        x = self.fc3(x)
        return x


class Classifer(pl.LightningModule):
    def __init__(self, input_size, num_classes, dropout_rate):
        super(Classifer, self).__init__()
        self.pool = nn.MaxPool2d((7, 7), stride=(1, 1))
        self.flatten = nn.Flatten()
        self.ffd = FeedForward(input_size, dropout_rate)
        self.act = nn.ReLU()
        self.fc = nn.Linear(input_size, num_classes)

    def forward(self, x):
        out = self.pool(x)
        out = self.flatten(out)
        out = out + self.ffd(out)
        out = self.act(out)
        out = self.fc(out)
        return out


class Fusion(nn.Module):
    def __init__(self, d_model: int):
        super(Fusion, self).__init__()
        self.d_model = d_model
        self.conv1 = nn.Conv2d(d_model * 2, d_model, kernel_size=1)
        self.act = nn.ReLU()
        self.conv2 = nn.Conv2d(d_model, d_model, kernel_size=1)
        self.sigmoid = nn.Tanh()

        self.cbam = CBAM(channels=d_model, reduction_rate=2, kernel_size=3)

    def forward(self, feature_1, feature_2):
        out = torch.cat((feature_1, feature_2), dim=1)
        out = self.conv1(out)
        out = self.act(out)
        out = self.conv2(out)
        attn = self.sigmoid(out)
        out = feature_1 * attn + feature_2 * (1 - attn)

        out = self.cbam(out)

        return out


class Head(nn.Module):
    def __init__(self, d_model: int, num_classes: int, dropout_rate: float):
        super(Head, self).__init__()
        self.fusion = Fusion(d_model)
        self.classifier = Classifer(d_model, num_classes, dropout_rate)

    def forward(self, x1, x2):
        x = self.fusion(x1, x2)
        return self.classifier(x)


if __name__ == '__main__':
    fusion = Head(256, 2, 0.2)
    feature_1 = torch.randn(10, 256, 7, 7)
    feature_2 = torch.randn(10, 256, 7, 7)
    output = fusion(feature_1, feature_2)
    print(output.shape)
    print(output)
