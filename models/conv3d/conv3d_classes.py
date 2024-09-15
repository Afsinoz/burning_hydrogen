import numpy as np
import torch
import torch.nn as nn


class ResBlock(nn.Module):
  """
  A residual network based on a single Conv3d layer.
  Input shape = (b, c, t, h, w)
  b = batch_size, c = channels, t = time, h = height, w = width
  """

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


def base_model(in_chans, out_chans, n_hidden, n_blocks, dropout_prob=0.5):
  """
  A basic sequential model consisting of a Conv3d, n_blocks layers of ResBlocks,
  followed by a final Conv3d.
  """

  return nn.Sequential(
      nn.Conv3d(in_channels=in_chans, out_channels=n_hidden,
                kernel_size=(3, 3, 3), padding='same'),
      nn.ReLU(),
      *(n_blocks * [ResBlock(n_chans=n_hidden), nn.Dropout(p=dropout_prob)]),
      nn.Conv3d(in_channels=n_hidden, out_channels=out_chans,
                kernel_size=(3, 3, 3), padding='same'),
  )


class Conv3dModel(nn.Module):
  """
  A model which uses a base_model to predict the errors from the baseline
  model which predicts the last image in a sequence to stay constant.
  """

  def __init__(self, in_chans=8, out_chans=1, n_hidden=16, n_blocks=5, pred_chan=1):
    """
    Parameters:
      in_chans: number of channels in the input tensor
      out_chans: number of channels in the output tensors
      n_hidden: the number of channels in the hidden layers for the ResBlocks
      n_blocks: number of ResBlocks in the model
      pred_chan: the channel to predict in the output
    """
    super(Conv3dModel, self).__init__()

    self.model1 = base_model(in_chans, in_chans, n_hidden, n_blocks)
    self.model2 = base_model(in_chans, in_chans, n_hidden, n_blocks)
    self.model3 = base_model(in_chans, in_chans, n_hidden, n_blocks)

    self.pred_chan = pred_chan

  def forward(self, x, future_steps=30):
    # The tensor which just stacks the last image seen in the input sequence
    x_id_future = torch.stack(future_steps * [x[:, 1, -1]], 1)

    # Train model2 on the predictions of model1, etc.
    y = self.model1(x)
    z = self.model2(y)
    w = self.model3(z)

    # Concatenate y,z,w, truncate to predct only the channel pred_chan,
    # for future_steps time steps into the future.
    out = torch.cat([y, z, w], dim=2)[:, self.pred_chan, :future_steps]

    # outputs will fit to the difference between
    # future images and the last image seen
    return out + x_id_future
