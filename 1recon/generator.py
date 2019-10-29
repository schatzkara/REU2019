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

        # definition of all layer channels
        layer_out_channels = {'conv_1a': 256,
                              'conv_1b': 256,
                              'conv_2a': 256,
                              'conv_2b': 256,
                              'rdn_block': sum(in_channels),
                              'inc_block': 256,
                              'conv_3a': 128,
                              'conv_3b': 128,
                              'conv_4a': 128,
                              'conv_4b': 3
                              }  # key: layer name, value: layer_out_channels
        layer_in_channels = {'conv_1a': sum(in_channels),
                             'conv_1b': layer_out_channels['conv_1a'],
                             'conv_2a': layer_out_channels['conv_1b'] + in_channels[0] + in_channels[1],  # + 256
                             'conv_2b': layer_out_channels['conv_2a'],
                             'rdn_block': sum(in_channels),
                             'inc_block': layer_out_channels['rdn_block'],
                             'conv_3a': layer_out_channels['conv_2b'] + in_channels[0] + in_channels[1],  # + 128
                             'conv_3b': layer_out_channels['conv_3a'],
                             'conv_4a': layer_out_channels['conv_3b'],
                             'conv_4b': layer_out_channels['conv_4a']
                             }  # key: layer name, value: layer_in_channels

        # definition of all network layers
        # block 1
        self.avg_pool_1 = nn.AvgPool3d(kernel_size=(4, 4, 4), stride=(4, 4, 4))
        # layer = 'conv_1a'
        # self.conv3d_1a = nn.Conv3d(in_channels=layer_in_channels[layer], out_channels=layer_out_channels[layer],
        #                            kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        # self.relu_1a = nn.ReLU(inplace=True)
        # layer = 'conv_1b'
        # self.conv3d_1b = nn.Conv3d(in_channels=layer_in_channels[layer], out_channels=layer_out_channels[layer],
        #                            kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        # self.relu_1b = nn.ReLU(inplace=True)
        layer = 'res_block'
        self.res_block = ResidualBlock(in_features=layer_in_channels[layer])

        # block 2
        self.avg_pool_2 = nn.AvgPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        # layer = 'conv_2a'
        # self.conv3d_2a = nn.Conv3d(in_channels=layer_in_channels[layer], out_channels=layer_out_channels[layer],
        #                            kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        # self.relu_2a = nn.ReLU(inplace=True)
        # layer = 'conv_2b'
        # self.conv3d_2b = nn.Conv3d(in_channels=layer_in_channels[layer], out_channels=layer_out_channels[layer],
        #                            kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        # self.relu_2b = nn.ReLU(inplace=True)
        layer = 'inc_block'
        self.inc_block = InceptionModule(in_channels=layer_in_channels[layer], out_channels=[32, 64, 64, 32, 32, 32])


        # block 3
        self.avg_pool_3 = nn.AvgPool3d(kernel_size=(2, 1, 1), stride=(2, 1, 1))
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

    def forward(self, app, kp):
        """
        Function to compute a single forward pass through the network, according to the architecture.
        :param app: (tensor) The input appearance encoding for the desired view of the generated video.
                     Must be a tensor of shape: (bsz, in_channels[0], 4, 14, 14) for this application.
        :param rep: (tensor) The input motion representation for the generated video.
                     Must be a tensor of shape: (bsz, in_channels[1], 4, 14, 14) for this application.
        :return: A tensor representing the video generated by the network.
                 Shape of output is: (bsz, 3, out_frames, 112, 112) for this application.
        """
        # block 1
        block = 1
        app_block_input = app[-block]
        kp_block_input = self.avg_pool_1(kp)
        x = torch.cat([app_block_input, kp_block_input], dim=1)  # dim=channels

        # x = self.conv3d_1a(x)
        # x = self.relu_1a(x)
        # x = self.conv3d_1b(x)
        # x = self.relu_1b(x)
        x = self.res_block(x)
        x = f.interpolate(x, size=(8, 28, 28), mode='trilinear')

        # block 2
        block = 2
        app_block_input = app[-block]
        kp_block_input = self.avg_pool_2(kp)
        x = torch.cat([app_block_input, kp_block_input, x], dim=1)

        # x = self.conv3d_2a(x)
        # x = self.relu_2a(x)
        # x = self.conv3d_2b(x)
        # x = self.relu_2b(x)
        x = self.inc_block(x)
        x = f.interpolate(x, size=(8, 56, 56), mode='trilinear')

        # block 3
        block = 3
        app_block_input = app[-block]
        kp_block_input = self.avg_pool_3(kp)
        x = torch.cat([app_block_input, kp_block_input, x], dim=1)

        x = self.conv3d_3a(x)
        x = self.relu_3a(x)
        x = self.conv3d_3b(x)
        x = self.relu_3b(x)

        x = f.interpolate(x, size=(self.out_frames, 56, 56), mode='trilinear')

        # block 4
        x = self.conv3d_4a(x)
        x = self.relu_4a(x)
        x = self.conv3d_4b(x)
        x = self.sigmoid(x)

        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_features, in_features, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_features, 0.8),
            nn.PReLU(),
            nn.Conv2d(in_features, in_features, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_features, 0.8),
        )

    def forward(self, x):
        return x + self.conv_block(x)


class InceptionModule(nn.Module):
    """
    Class representing a single Inception Module that is part of the I3D Network design.
    """

    def __init__(self, in_channels, out_channels, name):
        super(InceptionModule, self).__init__()

        self.b0 = Unit3D(in_channels=in_channels, output_channels=out_channels[0], kernel_shape=[1, 1, 1], padding=0,
                         name=name + '/Branch_0/Conv3d_0a_1x1')
        self.b1a = Unit3D(in_channels=in_channels, output_channels=out_channels[1], kernel_shape=[1, 1, 1], padding=0,
                          name=name + '/Branch_1/Conv3d_0a_1x1')
        self.b1b = Unit3D(in_channels=out_channels[1], output_channels=out_channels[2], kernel_shape=[3, 3, 3],
                          name=name + '/Branch_1/Conv3d_0b_3x3')
        self.b2a = Unit3D(in_channels=in_channels, output_channels=out_channels[3], kernel_shape=[1, 1, 1], padding=0,
                          name=name + '/Branch_2/Conv3d_0a_1x1')
        self.b2b = Unit3D(in_channels=out_channels[3], output_channels=out_channels[4], kernel_shape=[3, 3, 3],
                          name=name + '/Branch_2/Conv3d_0b_3x3')
        self.b3a = MaxPool3dSamePadding(kernel_size=[3, 3, 3],
                                        stride=(1, 1, 1), padding=0)
        self.b3b = Unit3D(in_channels=in_channels, output_channels=out_channels[5], kernel_shape=[1, 1, 1], padding=0,
                          name=name + '/Branch_3/Conv3d_0b_1x1')
        self.name = name

    def forward(self, x):
        b0 = self.b0(x)
        b1 = self.b1b(self.b1a(x))
        b2 = self.b2b(self.b2a(x))
        b3 = self.b3b(self.b3a(x))
        return torch.cat([b0, b1, b2, b3], dim=1)


class MaxPool3dSamePadding(nn.MaxPool3d):
    """
    Class respresenting a 3D Max Pooling layer that computes padding necessary for legal computation with kernel.
    """

    def compute_pad(self, dim, s):
        if s % self.stride[dim] == 0:
            return max(self.kernel_size[dim] - self.stride[dim], 0)
        else:
            return max(self.kernel_size[dim] - (s % self.stride[dim]), 0)

    def forward(self, x):
        # compute 'same' padding
        (batch, channel, t, h, w) = x.size()
        # print t,h,w
        out_t = np.ceil(float(t) / float(self.stride[0]))
        out_h = np.ceil(float(h) / float(self.stride[1]))
        out_w = np.ceil(float(w) / float(self.stride[2]))
        # print out_t, out_h, out_w
        pad_t = self.compute_pad(0, t)
        pad_h = self.compute_pad(1, h)
        pad_w = self.compute_pad(2, w)
        # print pad_t, pad_h, pad_w

        pad_t_f = pad_t // 2
        pad_t_b = pad_t - pad_t_f
        pad_h_f = pad_h // 2
        pad_h_b = pad_h - pad_h_f
        pad_w_f = pad_w // 2
        pad_w_b = pad_w - pad_w_f

        pad = (pad_w_f, pad_w_b, pad_h_f, pad_h_b, pad_t_f, pad_t_b)
        # print x.size()
        # print pad
        x = F.pad(x, pad)
        return super(MaxPool3dSamePadding, self).forward(x)


class Unit3D(nn.Module):
    """
    Class Representing a 3D Convolutional Unit that computes padding necessary for legal convolution with kernel.
    """

    def __init__(self, in_channels,
                 output_channels,
                 kernel_shape=(1, 1, 1),
                 stride=(1, 1, 1),
                 padding=0,
                 activation_fn=F.relu,
                 use_batch_norm=True,
                 use_bias=False,
                 name='unit_3d'):

        """Initializes Unit3D module."""
        super(Unit3D, self).__init__()

        self._output_channels = output_channels
        self._kernel_shape = kernel_shape
        self._stride = stride
        self._use_batch_norm = use_batch_norm
        self._activation_fn = activation_fn
        self._use_bias = use_bias
        self.name = name
        self.padding = padding

        self.conv3d = nn.Conv3d(in_channels=in_channels,
                                out_channels=self._output_channels,
                                kernel_size=self._kernel_shape,
                                stride=self._stride,
                                padding=0,
                                # we always want padding to be 0 here.
                                # We will dynamically pad based on input size in forward function
                                bias=self._use_bias)

        if self._use_batch_norm:
            self.bn = nn.BatchNorm3d(self._output_channels, eps=0.001, momentum=0.01)

    def compute_pad(self, dim, s):
        if s % self._stride[dim] == 0:
            return max(self._kernel_shape[dim] - self._stride[dim], 0)
        else:
            return max(self._kernel_shape[dim] - (s % self._stride[dim]), 0)

    def forward(self, x):
        # compute 'same' padding
        (batch, channel, t, h, w) = x.size()
        # print t,h,w
        out_t = np.ceil(float(t) / float(self._stride[0]))
        out_h = np.ceil(float(h) / float(self._stride[1]))
        out_w = np.ceil(float(w) / float(self._stride[2]))
        # print out_t, out_h, out_w
        pad_t = self.compute_pad(0, t)
        pad_h = self.compute_pad(1, h)
        pad_w = self.compute_pad(2, w)
        # print pad_t, pad_h, pad_w

        pad_t_f = pad_t // 2
        pad_t_b = pad_t - pad_t_f
        pad_h_f = pad_h // 2
        pad_h_b = pad_h - pad_h_f
        pad_w_f = pad_w // 2
        pad_w_b = pad_w - pad_w_f

        pad = (pad_w_f, pad_w_b, pad_h_f, pad_h_b, pad_t_f, pad_t_b)
        # print x.size()
        # print pad
        x = F.pad(x, pad)
        # print x.size()

        x = self.conv3d(x)
        if self._use_batch_norm:
            x = self.bn(x)
        if self._activation_fn is not None:
            x = self._activation_fn(x)
        return x


if __name__ == "__main__":
    print_summary = True

    gen = Generator(layer_in_channels=[1, 1], out_frames=16)

    if print_summary:
        summary(gen, input_size=[[(1, 56, 56), (1, 28, 28), (1, 14, 14)], (1, 4, 14, 14)])
