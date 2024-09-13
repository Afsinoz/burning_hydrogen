import numpy as np
import torch
import torch.nn as nn

# An encoder-decoder model using ConvLSTM cells


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


class EncoderDecoderConvLSTM(nn.Module):
    """
        Encoder-decoder network using ConvLSTM cells as modified from
        https://github.com/holmdk/Video-Prediction-using-PyTorch
    """

    def __init__(self, nf=24, in_chan=8, out_chan=1, past_steps=11,
                 future_steps=30, dropout_prob=0.5):
        """
            The architecture is as follows.
            Encoder: ConvLSTM
            Encoder vector: final hidden state
            Decoder: ConvLSTM (takes encoder vector as input
            Decoder: 3D CNN (produces regression predictions for the model)
        """
        super(EncoderDecoderConvLSTM, self).__init__()

        self.past_steps = past_steps
        self.future_steps = future_steps

        self.encoder_1_convlstm = ConvLSTMCell(input_dim=in_chan,
                                               hidden_dim=nf,
                                               kernel_size=(3, 3),
                                               bias=True)

        self.encoder_2_convlstm = ConvLSTMCell(input_dim=nf,
                                               hidden_dim=nf,
                                               kernel_size=(3, 3),
                                               bias=True)

        # Add batchnorm between encoder and decoder
        self.batchnorm1 = nn.BatchNorm2d(num_features=nf)

        self.decoder_1_convlstm = ConvLSTMCell(input_dim=nf,
                                               hidden_dim=nf,
                                               kernel_size=(3, 3),
                                               bias=True)

        self.decoder_2_convlstm = ConvLSTMCell(input_dim=nf,
                                               hidden_dim=nf,
                                               kernel_size=(3, 3),
                                               bias=True)

        # Add batchnorm between decoder ConvLSTM cells and Conv3d
        self.batchnorm2 = nn.BatchNorm3d(num_features=nf)

        # Add dropout at each step of input to encoder ConvLSTMs
        self.encoder_dropouts = [nn.Dropout(
            p=dropout_prob) for i in range(self.past_steps)]

        # Add dropout at each step of input to decoder ConvLSTMs
        self.decoder_dropouts = [nn.Dropout(
            p=dropout_prob) for i in range(self.future_steps)]

        self.decoder_CNN = nn.Conv3d(in_channels=nf,
                                     out_channels=out_chan,
                                     kernel_size=(1, 3, 3),
                                     padding=(0, 1, 1))

    def autoencoder(self, x, h_t, c_t, h_t2, c_t2, h_t3, c_t3, h_t4, c_t4):
        outputs = []

        # Encoder
        for i, t in enumerate(range(self.past_steps)):
            # Add dropout to hidden state
            h_t = self.encoder_dropouts[i](h_t)

            h_t, c_t = self.encoder_1_convlstm(input_tensor=x[:, t, :, :],
                                               cur_state=[h_t, c_t])
            h_t2, c_t2 = self.encoder_2_convlstm(input_tensor=h_t,
                                                 cur_state=[h_t2, c_t2])

        # Encoder vector
        encoder_vector = h_t2

        # Apply batchnorm to encoder vector
        encoder_vector = self.batchnorm1(encoder_vector)

        # Decoder
        for i, t in enumerate(range(self.future_steps)):
            # Add dropout to encoder vector
            encoder_vector = self.decoder_dropouts[i](encoder_vector)

            h_t3, c_t3 = self.decoder_1_convlstm(input_tensor=encoder_vector,
                                                 cur_state=[h_t3, c_t3])
            h_t4, c_t4 = self.decoder_2_convlstm(input_tensor=h_t3,
                                                 cur_state=[h_t4, c_t4])
            encoder_vector = h_t4
            outputs.append(h_t4)

        outputs = torch.stack(outputs, 1)
        outputs = outputs.permute(0, 2, 1, 3, 4)
        outputs = self.batchnorm2(outputs)
        outputs = self.decoder_CNN(outputs)
        outputs = outputs.permute(0, 2, 1, 3, 4)

        return outputs

    def forward(self, x, hidden_state=None):
        """
        Parameters
        ----------
        Input_tensor:
            5-D Tensor of shape (b, t, c, h, w)
            b = batch size, t = time, c = channels,
            h = height, w = width
        """

        # The tensor which just stacks the last image seen in the input sequence
        x_id_future = torch.stack(
            self.future_steps * [x[:, -1, 1, :, :].unsqueeze(1)], 1)

        # Find size of different input dimensions
        b, _, _, h, w = x.size()

        # Initialize hidden states
        h_t, c_t = self.encoder_1_convlstm.init_hidden(
            batch_size=b, image_size=(h, w))
        h_t2, c_t2 = self.encoder_2_convlstm.init_hidden(
            batch_size=b, image_size=(h, w))
        h_t3, c_t3 = self.decoder_1_convlstm.init_hidden(
            batch_size=b, image_size=(h, w))
        h_t4, c_t4 = self.decoder_2_convlstm.init_hidden(
            batch_size=b, image_size=(h, w))

        # Autoencoder forward
        outputs = self.autoencoder(x, h_t, c_t, h_t2, c_t2,
                                   h_t3, c_t3, h_t4, c_t4)

        # outputs will fit to the difference between
        # future images and the last image seen
        return outputs + x_id_future
