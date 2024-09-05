import numpy as np
import torch
import torch.nn as nn


class NoModel(nn.Module):
    def __init__(self):
        super(NoModel, self).__init__()

    def forward(self, x, future_steps=30, hidden_state=None):
        return torch.stack(future_steps * [x[:, -1, :, :, :]], 1)
