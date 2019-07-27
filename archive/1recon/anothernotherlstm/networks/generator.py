import torch
import torch.nn as nn
import torch.nn.functional as f
from torchsummary import summary

device = 'cuda' if torch.cuda.is_available() else 'cpu'


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
        layer_out_channels = {'conv_1a': 256,
                              'conv_1b': 128,
                              'conv_2a': 128,
                              'conv_2b': 64,
                              'conv_3a': 64,
                              'conv_3b': 32,
                              'conv_4a': 16,
                              'conv_4b': 3
                              }  # key: layer name, value: layer_out_channels
        layer_in_channels = {'conv_1a': sum(in_channels),
                             'conv_1b': layer_out_channels['conv_1a'],
                             'conv_2a': layer_out_channels['conv_1b'] + in_channels[0],  # + 256
                             'conv_2b': layer_out_channels['conv_2a'],
                             'conv_3a': layer_out_channels['conv_2b'] + in_channels[0],  # + 128
                             'conv_3b': layer_out_channels['conv_3a'],
                             'conv_4a': layer_out_channels['conv_3b'],
                             'conv_4b': layer_out_channels['conv_4a']
                             }  # key: layer name, value: layer_in_channels

        # block 1
        layer = 'conv_1a'
        self.conv3d_1a = nn.Conv3d(in_channels=layer_in_channels[layer], out_channels=layer_out_channels[layer],
                                   kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        self.relu_1a = nn.ReLU(inplace=True)
        layer = 'conv_1b'
        self.conv3d_1b = nn.Conv3d(in_channels=layer_in_channels[layer], out_channels=layer_out_channels[layer],
                                   kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        self.relu_1b = nn.ReLU(inplace=True)

        # block 2
        layer = 'conv_2a'
        self.conv3d_2a = nn.Conv3d(in_channels=layer_in_channels[layer], out_channels=layer_out_channels[layer],
                                   kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        self.relu_2a = nn.ReLU(inplace=True)
        layer = 'conv_2b'
        self.conv3d_2b = nn.Conv3d(in_channels=layer_in_channels[layer], out_channels=layer_out_channels[layer],
                                   kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        self.relu_2b = nn.ReLU(inplace=True)

        # block 3
        layer = 'conv_3a'
        self.conv3d_3a = nn.Conv3d(in_channels=layer_in_channels[layer], out_channels=layer_out_channels[layer],
                                   kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        self.relu_3a = nn.ReLU(inplace=True)
        layer = 'conv_3b'
        self.conv3d_3b = nn.Conv3d(in_channels=layer_in_channels[layer], out_channels=layer_out_channels[layer],
                                   kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        self.relu_3b = nn.ReLU(inplace=True)

        # block 4
        layer = 'conv_4a'
        self.conv3d_4a = nn.Conv3d(in_channels=layer_in_channels[layer], out_channels=layer_out_channels[layer],
                                   kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        self.relu_4a = nn.ReLU(inplace=True)
        layer = 'conv_4b'
        self.conv3d_4b = nn.Conv3d(in_channels=layer_in_channels[layer], out_channels=layer_out_channels[layer],
                                   kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0))
        self.sigmoid = nn.Sigmoid()

        # print('%s Model Successfully Built \n' % self.gen_name)

    def forward(self, app):
        """
        Function to compute a single forward pass through the network, according to the architecture.
        :param app: (tensor) The input tensor from which a video will be generated.
                   Must be a tensor of shape: (bsz, in_channels, 4, 14, 14) for this application.
        :return: A tensor representing the video generated by the network.
                 Shape of output is: (bsz, 3, 8/16, 112, 112) for this application.
        """
        # block 1
        block = 1
        app_block_input = app[-block]

        x = self.conv3d_1a(app_block_input)
        x = self.relu_1a(x)
        x = self.conv3d_1b(x)
        x = self.relu_1b(x)

        x = f.interpolate(x, size=(8, 28, 28), mode='trilinear')

        # block 2
        block = 2
        app_block_input = app[-block]
        x = torch.cat([app_block_input, x], dim=1)

        x = self.conv3d_2a(x)
        x = self.relu_2a(x)
        x = self.conv3d_2b(x)
        x = self.relu_2b(x)

        x = f.interpolate(x, size=(8, 56, 56), mode='trilinear')

        # block 3
        block = 3
        app_block_input = app[-block]
        x = torch.cat([app_block_input, x], dim=1)

        x = self.conv3d_3a(x)
        x = self.relu_3a(x)
        x = self.conv3d_3b(x)
        x = self.relu_3b(x)

        x = f.interpolate(x, size=(16, 112, 112), mode='trilinear')

        # block 4
        x = self.conv3d_4a(x)
        x = self.relu_4a(x)
        x = self.conv3d_4b(x)
        x = self.sigmoid(x)

        return x


if __name__ == "__main__":
    print_summary = False

    gen = Generator(in_channels=[128], out_frames=16)

    print(gen)

    if print_summary:
        summary(gen, input_size=[[(1, 56, 56), (1, 28, 28), (1, 14, 14)], (1, 4, 14, 14)])