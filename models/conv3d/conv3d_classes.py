import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F


class ResBlock(nn.Module):
    def __init__(self, n_chans=8):
        super(ResBlock, self).__init__()
        self.conv = nn.Conv3d(in_channels=n_chans, out_channels=n_chans,
                              kernel_size=(3, 3, 3), padding='same')
        self.batch_norm = nn.BatchNorm3d(num_features=n_chans)
        torch.nn.init.kaiming_normal_(self.conv.weight, nonlinearity='relu')
        torch.nn.init.constant_(self.batch_norm.weight, 0.5)
        torch.nn.init.zeros_(self.batch_norm.bias)

    def forward(self, x):
        out = self.conv(x)
        out = self.batch_norm(out)
        out = torch.relu(out)
        return out + x


def base_model(n_chans=8, n_hidden=16, n_blocks=5):
    return nn.Sequential(
        nn.Conv3d(in_channels=n_chans, out_channels=n_hidden,
                  kernel_size=(3, 3, 3), padding='same'),
        nn.ReLU(),
        *(n_blocks * [ResBlock(n_chans=n_hidden)]),
        nn.Conv3d(in_channels=n_hidden, out_channels=n_chans,
                  kernel_size=(3, 3, 3), padding='same'),
    )


class Conv3dModel(nn.Module):

    def __init__(self, n_chans=8, n_hidden=16, n_blocks=5):
        super(Conv3dModel, self).__init__()

        self.model1 = base_model(n_chans, n_hidden, n_blocks)
        self.model2 = base_model(n_chans, n_hidden, n_blocks)
        self.model3 = base_model(n_chans, n_hidden, n_blocks)

    def forward(self, x):
        y = self.model1(x)
        z = self.model2(y)
        w = self.model3(z)

        out = [y, z, w]

        return torch.cat(out, dim=2)[:, :, :30, :, :]
