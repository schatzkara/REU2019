import torch
import torch.nn as nn
import torch.nn.functional as f
from torchsummary import summary

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class Discriminator(nn.Module):
    """
    Class representing the Discriminator network to be used.
    """

    def __init__(self, in_channels=3, gen_name='Video Discriminator'):
        """
        Initializes the Discriminator network.
        :param in_channels: (int) The number of channels in the input tensor.
        :param gen_name: (str, optional) The name of the network (default 'Video Generator').
        Raises:
            ValueError: if 'out_frames' is not a legal value.
        """

        super(Discriminator, self).__init__()
        self.gen_name = gen_name

        # definition of all layer channels
        layer_out_channels = {'conv_1': 16,
                              'conv_2': 32,
                              'conv_3': 64,
                              'conv_4': 128,
                              'conv_5': 256,
                              'linear': 1
                              }  # key: layer name, value: layer_out_channels
        layer_in_channels = {'conv_1': in_channels,
                             'conv_2': layer_out_channels['conv_1'],
                             'conv_3': layer_out_channels['conv_2'],
                             'conv_4': layer_out_channels['conv_3'],
                             'conv_5': layer_out_channels['conv_4'],
                             'linear': layer_out_channels['conv_5']
                             }  # key: layer name, value: layer_in_channels

        # definition of all network layers
        layer = 'conv_1'
        self.conv3d_1 = nn.Conv3d(in_channels=layer_in_channels[layer], out_channels=layer_out_channels[layer],
                                  kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        self.relu_1 = nn.ReLU(inplace=True)
        layer = 'conv_2'
        self.conv3d_2 = nn.Conv3d(in_channels=layer_in_channels[layer], out_channels=layer_out_channels[layer],
                                  kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        self.relu_2 = nn.ReLU(inplace=True)
        layer = 'conv_3'
        self.conv3d_3 = nn.Conv3d(in_channels=layer_in_channels[layer], out_channels=layer_out_channels[layer],
                                  kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        self.relu_3 = nn.ReLU(inplace=True)
        layer = 'conv_4'
        self.conv3d_4 = nn.Conv3d(in_channels=layer_in_channels[layer], out_channels=layer_out_channels[layer],
                                  kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        self.relu_4 = nn.ReLU(inplace=True)
        layer = 'conv_5'
        self.conv3d_5 = nn.Conv3d(in_channels=layer_in_channels[layer], out_channels=layer_out_channels[layer],
                                  kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0))
        self.relu_5 = nn.ReLU(inplace=True)
        layer = 'linear'
        self.fc_1 = nn.Linear(in_features=layer_in_channels[layer], out_features=layer_out_channels[layer])

        # print('%s Model Successfully Built \n' % self.gen_name)

    def forward(self, vid):
        """
        Function to compute a single forward pass through the network, according to the architecture.
        :param vid: (tensor) .
        :return: .
        """
        x = self.relu_1(self.conv3d_1(vid))
        x = self.relu_2(self.conv3d_2(x))
        x = self.relu_3(self.conv3d_3(x))
        x = self.relu_4(self.conv3d_4(x))
        x = self.relu_5(self.conv3d_5(x))

        bsz, channels, height, width = x.size()
        x = x.reshape(bsz, -1)

        x = self.fc_1(x).squeeze()

        return x


if __name__ == "__main__":
    print_summary = True

    dis = Discriminator()

    if print_summary:
        summary(dis, input_size=(16, 3, 112, 112))
