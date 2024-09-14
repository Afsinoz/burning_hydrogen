import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.nn import functional as F
import torch.optim as optim
import tensorboard
from google.colab import drive


class ConvLSTMCell(nn.Module):
    """
    A ConvLSTM cell as implemented in https://github.com/ndrplz/ConvLSTM_pytorch
    """

    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        """
        Initialize ConvLSTM cell.

        Parameters
        ----------
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """
        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        # concatenate along channel axis
        combined = torch.cat([input_tensor, h_cur], dim=1)

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(
            combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width,
                            device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width,
                            device=self.conv.weight.device))


class ConvLSTM(nn.Module):
    """
    ConvLSTM network as modified from https://github.com/ndrplz/ConvLSTM_pytorch
    """

    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers,
                 batch_first=False, bias=True, return_all_layers=False,
                 dropout_prob=0.5, output_size=30, output_channel=1):
        """
        Parameters:
            input_dim: Number of channels in input
            hidden_dim: Number of hidden channels
            kernel_size: Size of kernel in convolutions
            num_layers: Number of LSTM layers stacked on each other
            batch_first: Whether or not dimension 0 is the batch or not
            bias: Bias or no bias in Convolution
            return_all_layers: Return the list of computations for all layers
            dropout_prob: Probability of an element being zeroed by dropout layers
            output_size: Length of the sequence to output
            output_channel: the channel to predict in the output
        """
        super(ConvLSTM, self).__init__()

        self._check_kernel_size_consistency(kernel_size)

        # Make sure that both `kernel_size` and `hidden_dim` are
        # lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers
        self.output_size = output_size
        self.output_channel = output_channel

        cell_list, batchnorm_list1, dropout_list1, batchnorm_list2, dropout_list2 = \
            [], [], [], [], []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]

            cell_list.append(ConvLSTMCell(input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias))

            batchnorm_list1.append(torch.nn.BatchNorm2d(cur_input_dim))
            dropout_list1.append(torch.nn.Dropout(p=dropout_prob))
            batchnorm_list2.append(torch.nn.BatchNorm2d(cur_input_dim))
            dropout_list2.append(torch.nn.Dropout(p=dropout_prob))

        self.cell_list = nn.ModuleList(cell_list)
        self.batchnorm_list1 = nn.ModuleList(batchnorm_list1)
        self.dropout_list1 = nn.ModuleList(dropout_list1)
        self.batchnorm_list2 = nn.ModuleList(batchnorm_list2)
        self.dropout_list2 = nn.ModuleList(dropout_list2)

    def forward(self, input_tensor, hidden_state=None):
        """
        Parameters
        ----------
        input_tensor: 5-D Tensor either of shape (b, t, c, h, w) or (t, b, c, h, w)

        Returns
        ----------
        Predicted sequence of images in self.output_channel
        """
        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        b, _, _, h, w = input_tensor.size()

        # Since the init is done in forward. Can send image size here.
        hidden_state = self._init_hidden(batch_size=b, image_size=(h, w))

        layer_output_list = []
        last_state_list = []

        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):

            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                # Add batch norm and dropout to the previous hidden state.
                h = self.batchnorm_list1[layer_idx](h)
                h = self.dropout_list1[layer_idx](h)
                c = self.batchnorm_list2[layer_idx](c)
                c = self.dropout_list2[layer_idx](c)
                h, c = self.cell_list[layer_idx](
                    input_tensor=cur_layer_input[:, t, :, :, :],
                    cur_state=[h, c])
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]

        # Concatenate output of all ConvLSTM layers. Truncate to achieve
        # desired output sequence length.
        out = torch.cat(layer_output_list, dim=1)[:, :self.output_size]
        out = out[:, :, self.output_channel]

        # The tensor which just stacks the last image seen in the input sequence
        id = torch.stack(
            self.output_size * [input_tensor[:, -1, self.output_channel, :, :]], dim=1)

        # outputs will fit to the difference between
        # future images and the last image seen
        return out + id

    def _init_hidden(self, batch_size, image_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(
                self.cell_list[i].init_hidden(batch_size, image_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple)
                                                        for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param
