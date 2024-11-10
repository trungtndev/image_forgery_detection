import pytorch_lightning as pl
import torch
from einops import rearrange
from torch import nn
# from .cbam import CBAM

class FeedForward(nn.Module):
    def __init__(self, d_model: int, dropout_rate: float):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_model * 2, bias=True)
        self.fc2 = nn.Linear(d_model * 2, d_model, bias=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.act = nn.GELU()


    def forward(self, x):
        out = self.fc1(x)
        out = self.act(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out


class Classifer(pl.LightningModule):
    def __init__(self, input_size, num_classes, dropout_rate):
        super(Classifer, self).__init__()
        self.gru = GRUBlock(input_size, input_size)
        self.flatten = nn.Flatten()
        self.ffd = FeedForward(input_size, dropout_rate)
        self.act = nn.GELU()
        self.fc = nn.Linear(input_size, num_classes)

    def forward(self, x):
        out = self.gru(x)
        out = out[:, -1, :]
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
        self.tanh = nn.Tanh()
        self.posnorm = nn.BatchNorm2d(d_model)

        # self.cbam = CBAM(channels=d_model, reduction_rate=2, kernel_size=3)

    def forward(self, feature_1, feature_2):
        out = torch.cat((feature_1, feature_2), dim=1)
        out = self.conv1(out)
        out = self.act(out)
        out = self.conv2(out)
        attn = (self.tanh(out) + 1) / 2
        out = feature_1 * attn + feature_2 * (1 - attn)
        out = self.posnorm(out)
        out = self.cbam(out)

        return out

class GRUBlock(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(GRUBlock, self).__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRUCell(input_size, hidden_size)

    def forward(self, x):
        # Assuming x has shape (batch_size, sequence_length, input_size)
        batch_size, sequence_length, input_size = x.size()
        h = torch.zeros(batch_size, self.hidden_size, device=x.device)

        # Iterate through the sequence
        outputs = []
        for i in range(sequence_length):
            h = self.gru(x[:, i, :], h)
            outputs.append(h.unsqueeze(1))

        # Concatenate along the sequence dimension
        outputs = torch.cat(outputs, dim=1)
        return outputs

class Head(nn.Module):
    def __init__(self, d_model: int, num_classes: int, dropout_rate: float):
        super(Head, self).__init__()
        self.fusion = Fusion(d_model)
        self.classifier = Classifer(d_model, num_classes, dropout_rate)

    def forward(self, x1, x2):
        x = self.fusion(x1, x2)
        x = rearrange(x, 'b c h w -> b (h w) c')
        return self.classifier(x)


if __name__ == '__main__':
    fusion = Head(256, 2, 0.2)
    feature_1 = torch.randn(10, 256, 7, 7)
    feature_2 = torch.randn(10, 256, 7, 7)
    output = fusion(feature_1, feature_2)
    print(output.shape)
    print(output)
