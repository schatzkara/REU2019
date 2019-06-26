# phase 2

import torch
import torch.nn as nn
import torch.nn.functional as f
# from torchsummary import summary


class Deconv(nn.Module):
    """
    Class representing the Deconvolutional network that is used to estimate the video keypoints.
    """

    VALID_OUT_FRAMES = (8, 16)

    def __init__(self, in_channels, out_frames, deconv_name='Deconvolutional Network'):
        """
        Initializes the Deconvolutional network.
        :param in_channels: (int) The number of channels in the input tensor.
        :param out_frames: (int) The number of frames desired in the generated output video.
                            Legal values: 8, 16
        :param deconv_name: (str, optional) The name of the network (default 'Deconvolutinal Network').
        Raises:
            ValueError: if 'out_frames' is not a legal value.
        """
        if out_frames not in self.VALID_OUT_FRAMES:
            raise ValueError('Invalid number of frames in desired output: %d' % out_frames)

        super(Deconv, self).__init__()
        self.deconv_name = deconv_name

        self.out_frames = out_frames

        # definition of all network layers
        self.conv3d_1a = nn.Conv3d(in_channels=in_channels, out_channels=256, kernel_size=(3, 3, 3),
                                   stride=(1, 1, 1), padding=(1, 1, 1))
        self.conv3d_1b = nn.Conv3d(in_channels=256, out_channels=128, kernel_size=(3, 3, 3),
                                   stride=(1, 1, 1), padding=(1, 1, 1))

        self.conv3d_2a = nn.Conv3d(in_channels=128, out_channels=64, kernel_size=(3, 3, 3),
                                   stride=(1, 1, 1), padding=(1, 1, 1))
        self.conv3d_2b = nn.Conv3d(in_channels=64, out_channels=32, kernel_size=(3, 3, 3),
                                   stride=(1, 1, 1), padding=(1, 1, 1))

        # print('%s Model Successfully Built \n' % self.deconv_name)

    def forward(self, x):
        """
        Function to compute a single forward pass through the network, according to the architecture.
        :param x: (tensor) The input tensor from which to generate the keypoints.
                   Must be a tensor of shape: (bsz, 256, 4, 7, 7) for this application.
        :return: A tensor representing the keypoints of the input video.
                 Shape of output is: (bsz, 32, 8/16, 28, 28) for this application.
        """
        x = self.conv3d_1a(x)
        x = self.conv3d_1b(x)

        x = f.interpolate(x, size=(int(self.out_frames/2), 28, 28), mode='nearest')
        x = self.conv3d_2a(x)
        x = self.conv3d_2b(x)

        x = f.interpolate(x, size=(self.out_frames, 28, 28), mode='nearest')

        return x


# if __name__ == "__main__":
#     print_summary = True
#
#     deconv = Deconv(in_channels=256, out_frames=16)
#
#     if print_summary:
#         summary(deconv, input_size=(256, 4, 7, 7))
