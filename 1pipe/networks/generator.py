# phase 3

import torch
import torch.nn as nn
import torch.nn.functional as f
from torchsummary import summary


class Generator(nn.Module):
    """
    Class representing the Generator network to be used.
    """

    VALID_OUT_FRAMES = (8, 16)

    def __init__(self, in_channels, out_frames, gen_name='Video Generator'):
        """
        Initializes the Generator network.
        :param in_channels: (int) The number of channels in the input tensor.
        :param out_frames: (int) The number of frames desired in the generated output video.
                            Legal values: 8, 16
        :param gen_name: (str, optional) The name of the network (default 'Video Generator').
        Raises:
            ValueError: if 'out_frames' is not a legal value.
        """
        if out_frames not in self.VALID_OUT_FRAMES:
            raise ValueError('Invalid number of frames in desired output: %d' % out_frames)

        super(Generator, self).__init__()
        self.gen_name = gen_name
        self.out_frames = out_frames

        # definition of all network layers
        self.conv3d_1a = nn.Conv3d(in_channels=in_channels, out_channels=128, kernel_size=(3, 3, 3),
                                   stride=(1, 1, 1), padding=(1, 1, 1))
        self.relu_1a = nn.ReLU(inplace=True)
        self.conv3d_1b = nn.Conv3d(in_channels=128, out_channels=64, kernel_size=(3, 3, 3),
                                   stride=(1, 1, 1), padding=(1, 1, 1))
        self.relu_1b = nn.ReLU(inplace=True)

        self.conv3d_2a = nn.Conv3d(in_channels=64, out_channels=32, kernel_size=(3, 3, 3),
                                   stride=(1, 1, 1), padding=(1, 1, 1))
        self.relu_2a = nn.ReLU(inplace=True)
        self.conv3d_2b = nn.Conv3d(in_channels=32, out_channels=16, kernel_size=(3, 3, 3),
                                   stride=(1, 1, 1), padding=(1, 1, 1))
        self.relu_2b = nn.ReLU(inplace=True)

        self.conv3d_3a = nn.Conv3d(in_channels=16, out_channels=8, kernel_size=(3, 3, 3),
                                   stride=(1, 1, 1), padding=(1, 1, 1))
        self.relu_3a = nn.ReLU(inplace=True)
        self.conv3d_3b = nn.Conv3d(in_channels=8, out_channels=4, kernel_size=(3, 3, 3),
                                   stride=(1, 1, 1), padding=(1, 1, 1))
        self.relu_3b = nn.ReLU(inplace=True)

        self.conv3d_4a = nn.Conv3d(in_channels=4, out_channels=4, kernel_size=(3, 3, 3),
                                   stride=(1, 1, 1), padding=(1, 1, 1))
        self.relu_4a = nn.ReLU(inplace=True)
        self.conv3d_4b = nn.Conv3d(in_channels=4, out_channels=3, kernel_size=(3, 3, 3),
                                   stride=(1, 1, 1), padding=(1, 1, 1))
        self.sigmoid = nn.Sigmoid()

        # print('%s Model Successfully Built \n' % self.gen_name)

    def forward(self, x):
        """
        Function to compute a single forward pass through the network, according to the architecture.
        :param x: (tensor) The input tensor from which a video will be generated.
                   Must be a tensor of shape: (bsz, 64, 16, 28, 28) for this application.
        :return: A tensor representing the video generated by the network.
                 Shape of output is: (bsz, 3, 8/16, 112, 112) for this application.
        """
        # x = f.interpolate(x, size=(8, 14, 14), mode='trilinear')
        x = self.conv3d_1a(x)
        x = self.relu_1a(x)
        x = self.conv3d_1b(x)
        x = self.relu_1b(x)

        x = f.interpolate(x, size=(8, 28, 28), mode='trilinear')

        x = self.conv3d_2a(x)
        x = self.relu_2a(x)
        x = self.conv3d_2b(x)
        x = self.relu_2b(x)

        x = f.interpolate(x, size=(16, 56, 56), mode='trilinear')

        x = self.conv3d_3a(x)
        x = self.relu_3a(x)
        x = self.conv3d_3b(x)
        x = self.relu_3b(x)

        x = f.interpolate(x, size=(16, 112, 112), mode='trilinear')

        x = self.conv3d_4a(x)
        x = self.relu_4a(x)
        x = self.conv3d_4b(x)
        x = self.sigmoid(x)

        return x


if __name__ == "__main__":
    print_summary = True

    gen = Generator(in_channels=1, out_frames=16)

    if print_summary:
        summary(gen, input_size=(1, 4, 14, 14))
