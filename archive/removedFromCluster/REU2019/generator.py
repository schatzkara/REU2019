# DO NOT CHANGE
# THIS VERSION WORKS FOR 8 AND 16 FRAMES

import torch
import torch.nn as nn


class Generator(nn.Module):
    """
    Class representing the Generator network to be used.
    """

    VALID_OUT_FRAMES = (8, 16)

    def __init__(self, in_channels, out_frames=8, gen_name='Video Generator'):
        """
        Initializes the Generator network.
        :param in_channels: (int) The number of channels in the input tensor.
        :param out_frames: (int, optional) The number of frames desired in the generated output video (default 8).
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
        self.conv2d = nn.Conv2d(in_channels=in_channels, out_channels=1024, kernel_size=(3, 3),
                                stride=(1, 1), padding=(1, 1))

        self.upsamp1 = nn.Upsample(scale_factor=2, mode='nearest')

        self.conv3d_1a = nn.Conv3d(in_channels=1024, out_channels=256, kernel_size=(3, 3, 3),
                                   stride=(1, 1, 1), padding=(1, 1, 1))
        self.conv3d_1b = nn.Conv3d(in_channels=256, out_channels=256, kernel_size=(3, 3, 3),
                                   stride=(1, 1, 1), padding=(1, 1, 1))

        self.upsamp2 = nn.Upsample(scale_factor=2, mode='nearest')

        self.conv3d_2a = nn.Conv3d(in_channels=256, out_channels=128, kernel_size=(3, 3, 3),
                                   stride=(1, 1, 1), padding=(1, 1, 1))
        self.conv3d_2b = nn.Conv3d(in_channels=128, out_channels=128, kernel_size=(3, 3, 3),
                                   stride=(1, 1, 1), padding=(1, 1, 1))

        self.upsamp3 = nn.Upsample(scale_factor=2, mode='nearest')

        self.conv3d_3a = nn.Conv3d(in_channels=128, out_channels=64, kernel_size=(3, 3, 3),
                                   stride=(1, 1, 1), padding=(1, 1, 1))
        self.conv3d_3b = nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(3, 3, 3),
                                   stride=(1, 1, 1), padding=(1, 1, 1))

        self.upsamp4 = nn.Upsample(scale_factor=2, mode='nearest')

        self.conv3d_4 = nn.Conv3d(in_channels=64, out_channels=3, kernel_size=(1, 1, 1),
                                  stride=(1, 1, 1), padding=0)

        # print('%s Model Successfully Built \n' % self.gen_name)

    def forward(self, x):
        """
        Function to compute a single forward pass through the network, according to the architecture.
        :param x: (tensor) The input tensor from which a video will be generated.
                   Must be a tensor of shape: (bsz, 1536, 7, 7) for this application.
        :return: A tensor representing the video generated by the network.
                 Shape of output is: (bsz, 3, 8/16, 112, 112) for this application.
        """
        x = self.conv2d(x)

        if self.out_frames == 8:
            x = self.upsamp1(x)

        # add in the temporal dimension
        x = torch.unsqueeze(x, dim=2)

        if self.out_frames == 16:
            x = self.upsamp1(x)

        x = self.conv3d_1a(x)
        x = self.conv3d_1b(x)

        x = self.upsamp2(x)
        x = self.conv3d_2a(x)
        x = self.conv3d_2b(x)

        x = self.upsamp3(x)
        x = self.conv3d_3a(x)
        x = self.conv3d_3b(x)

        x = self.upsamp4(x)
        x = self.conv3d_4(x)

        return x