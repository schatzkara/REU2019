# phase 2

import torch
import torch.nn as nn
from torchsummary import summary


class Transformer(nn.Module):
    """
    Class representing the Transformer network to be used.
    """

    def __init__(self, in_channels, trans_name='Transformer Network'):
        super(Transformer, self).__init__()
        self.trans_name = trans_name

        self.in_channels = in_channels

        # definition of all network layers
        self.conv2d_1a = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=(3, 3),
                                   stride=(1, 1), padding=(1, 1))
        self.relu_1a = nn.ReLU(inplace=True)
        self.conv2d_1b = nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=(3, 3),
                                   stride=(1, 1), padding=(1, 1))
        self.relu_1b = nn.ReLU(inplace=True)
        self.conv2d_1c = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(3, 3),
                                   stride=(1, 1), padding=(1, 1))
        self.relu_1c = nn.ReLU(inplace=True)

        # print('%s Model Successfully Built \n' % self.trans_name)

    def forward(self, x):
        """
        Function to compute a single forward pass through the network, according to the architecture.
        :param x: (tensor) The input tensor from which a video will be generated.
                   Must be a tensor of shape: (bsz, 1536, 7, 7) for this application.
        :return: A tensor representing the video generated by the network.
                 Shape of output is: (bsz, 3, 8/16, 112, 112) for this application.
        """
        x = self.relu_1a(self.conv2d_1a(x))
        x = self.relu_1b(self.conv2d_1b(x))
        x = self.relu_1c(self.conv2d_1c(x))

        return x


if __name__ == "__main__":
    print_summary = True

    ex = Transformer(in_channels=33)

    if print_summary:
        summary(ex, input_size=(33, 14, 14))
