# phase 3

import torch
import torch.nn as nn
import torch.nn.functional as f
from torchsummary import summary


class Generator(nn.Module):
    """
    Class representing the Generator network to be used.
    """

    VALID_OUT_FRAMES = (16,)

    def __init__(self, in_channels, out_frames, gen_name='Video Generator'):
        """
        Initializes the Generator network.
        :param in_channels: (list of ints) The number of channels in each input tensor respectively
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
        out_channels = {'conv_1a': 256,
                        'conv_1b': 128,
                        'conv_2a': 64,
                        'conv_2b': 32,
                        'conv_3a': 16,
                        'conv_3b': 8,
                        'conv_4a': 4,
                        'conv_4b': 4,
                        'conv_5a': 4,
                        'conv_5b': 3
                        }  # key: layer name, value: out_channels
        in_channels = {'conv_1a': sum(in_channels),
                       'conv_1b': out_channels['conv_1a'],
                       'conv_2a': out_channels['conv_1b'] + 128 + in_channels[1] + in_channels[2],  # + 288
                       'conv_2b': out_channels['conv_2a'],
                       'conv_3a': out_channels['conv_2b'] + 64 + in_channels[1] + in_channels[2],  # + 288
                       'conv_3b': out_channels['conv_3a'],
                       'conv_4a': out_channels['conv_3b'] + in_channels[1],  # + 32
                       'conv_4b': out_channels['conv_4a'],
                       'conv_5a': out_channels['conv_4b'],  # + 32
                       'conv_5b': out_channels['conv_5a']
                       }  # key: layer name, value: in_channels

        # block 1
        self.avg_pool_kp_1 = nn.AvgPool3d(kernel_size=(16, 2, 2), stride=(16, 2, 2))
        self.avg_pool_rep_1 = nn.AvgPool3d(kernel_size=(4, 1, 1), stride=(4, 1, 1))

        layer = 'conv_1a'
        self.conv3d_1a = nn.Conv3d(in_channels=in_channels[layer], out_channels=out_channels[layer],
                                   kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        self.relu_1a = nn.ReLU(inplace=True)
        layer = 'conv_1b'
        self.conv3d_1b = nn.Conv3d(in_channels=in_channels[layer], out_channels=out_channels[layer],
                                   kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        self.relu_1b = nn.ReLU(inplace=True)

        # block 2
        self.avg_pool_kp_2 = nn.AvgPool3d(kernel_size=(8, 1, 1), stride=(8, 1, 1))
        self.avg_pool_rep_2 = nn.AvgPool3d(kernel_size=(2, 1, 1), stride=(2, 1, 1))

        layer = 'conv_2a'
        self.conv3d_2a = nn.Conv3d(in_channels=in_channels[layer], out_channels=out_channels[layer],
                                   kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        self.relu_2a = nn.ReLU(inplace=True)
        layer = 'conv_2b'
        self.conv3d_2b = nn.Conv3d(in_channels=in_channels[layer], out_channels=out_channels[layer],
                                   kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        self.relu_2b = nn.ReLU(inplace=True)

        # block 3
        self.avg_pool_kp_3 = nn.AvgPool3d(kernel_size=(4, 1, 1), stride=(4, 1, 1))

        layer = 'conv_3a'
        self.conv3d_3a = nn.Conv3d(in_channels=in_channels[layer], out_channels=out_channels[layer],
                                   kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        self.relu_3a = nn.ReLU(inplace=True)
        layer = 'conv_3b'
        self.conv3d_3b = nn.Conv3d(in_channels=in_channels[layer], out_channels=out_channels[layer],
                                   kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        self.relu_3b = nn.ReLU(inplace=True)

        # block 4
        self.avg_pool_kp_4 = nn.AvgPool3d(kernel_size=(2, 1, 1), stride=(2, 1, 1))

        layer = 'conv_4a'
        self.conv3d_4a = nn.Conv3d(in_channels=in_channels[layer], out_channels=out_channels[layer],
                                   kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        self.relu_4a = nn.ReLU(inplace=True)
        layer = 'conv_4b'
        self.conv3d_4b = nn.Conv3d(in_channels=in_channels[layer], out_channels=out_channels[layer],
                                   kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        self.relu_4b = nn.ReLU(inplace=True)

        # block 5
        layer = 'conv_5a'
        self.conv3d_5a = nn.Conv3d(in_channels=in_channels[layer], out_channels=out_channels[layer],
                                   kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        self.relu_5a = nn.ReLU(inplace=True)
        layer = 'conv_5b'
        self.conv3d_5b = nn.Conv3d(in_channels=in_channels[layer], out_channels=out_channels[layer],
                                   kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0))

        self.sigmoid = nn.Sigmoid()

        # print('%s Model Successfully Built \n' % self.gen_name)

    def forward(self, app, kp, rep):
        """
        Function to compute a single forward pass through the network, according to the architecture.
        :param app: (tensor) The appearance features for the desired view of the output video.
                     Must be a tensor of shape: (bsz, 256, 1, 14, 14) for this application.
        :param kp: (tensor) The keypoints for the video action.
                    Must be a tensor of shape: (bsz, 32, 16, 28, 28) for this application.
        :param rep: (tensor) The motion representation/features for the video action.
                     Must be a tensor of shape: (bsz, 256, 4, 14, 14) for this application.
        :return: A tensor representing the video generated by the network.
                 Shape of output is: (bsz, 3, 8/16, 112, 112) for this application.
        """
        # if len(app.size()) == 4:
        #     app = torch.unsqueeze(app, dim=2)  # dim=frames

        # block 1
        app_block_input = app[-1]
        if len(app_block_input.size()) == 4:
            app_block_input = torch.unsqueeze(app_block_input, dim=2)  # dim=frames
        kp_block_input = self.avg_pool_kp_1(kp)
        rep_block_input = self.avg_pool_rep_1(rep)
        x = torch.cat([app_block_input, kp_block_input, rep_block_input], dim=1)  # dim=channels
        x = self.conv3d_1a(x)
        x = self.relu_1a(x)
        x = self.conv3d_1b(x)
        x = self.relu_1b(x)

        x = f.interpolate(x, size=(2, 28, 28), mode='trilinear')

        # block 2
        app_block_input = app[-2]
        if len(app_block_input.size()) == 4:
            app_block_input = torch.unsqueeze(app_block_input, dim=2)  # dim=frames
        app_block_input = torch.cat([app_block_input, app_block_input], dim=2)  # dim=frames
        kp_block_input = self.avg_pool_kp_2(kp)
        rep_block_input = self.avg_pool_rep_2(rep)
        rep_block_input = f.interpolate(rep_block_input, size=(2, 28, 28), mode='trilinear')
        x = torch.cat([x, app_block_input, kp_block_input, rep_block_input], dim=1)  # dim=channels
        x = self.conv3d_2a(x)
        x = self.relu_2a(x)
        x = self.conv3d_2b(x)
        x = self.relu_2b(x)
        x = f.interpolate(x, size=(4, 56, 56), mode='trilinear')

        # block 3
        app_block_input = app[-3]
        if len(app_block_input.size()) == 4:
            app_block_input = torch.unsqueeze(app_block_input, dim=2)  # dim=frames
        for i in range(2):
            app_block_input = torch.cat([app_block_input, app_block_input], dim=2)  # dim=frames
        kp_block_input = self.avg_pool_kp_3(kp)
        kp_block_input = f.interpolate(kp_block_input, size=(4, 56, 56), mode='trilinear')
        rep_block_input = f.interpolate(rep, size=(4, 56, 56), mode='trilinear')
        x = torch.cat([x, app_block_input, kp_block_input, rep_block_input], dim=1)  # dim=channels
        x = self.conv3d_3a(x)
        x = self.relu_3a(x)
        x = self.conv3d_3b(x)
        x = self.relu_3b(x)

        x = f.interpolate(x, size=(8, 112, 112), mode='trilinear')

        # block 4
        kp_block_input = self.avg_pool_kp_4(kp)
        kp_block_input = f.interpolate(kp_block_input, size=(8, 112, 112), mode='trilinear')
        x = torch.cat([x, kp_block_input], dim=1)  # dim=channels
        x = self.conv3d_4a(x)
        x = self.relu_4a(x)
        x = self.conv3d_4b(x)
        x = self.relu_4b(x)

        x = f.interpolate(x, size=(16, 112, 112), mode='trilinear')

        # block 5
        # kp_block_input = f.interpolate(kp, size=(16, 112, 112), mode='trilinear')
        # x = torch.cat([x, kp_block_input], dim=1)  # dim=channels
        x = self.conv3d_5a(x)
        x = self.relu_5a(x)
        x = self.conv3d_5b(x)
        x = self.sigmoid(x)

        return x


if __name__ == "__main__":
    print_summary = True

    gen = Generator(in_channels=[1, 1, 1], out_frames=16)

    if print_summary:
        summary(gen, input_size=[(1, 14, 14), (1, 16, 28, 28), (1, 4, 14, 14)])
