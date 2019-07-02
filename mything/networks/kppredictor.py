# phase 2

import torch
import torch.nn as nn
import torch.nn.functional as f
from torchsummary import summary
import tensorflow as tf


class KPPredictor(nn.Module):
    """
    Class representing the Keypoint Predictor network that is used to estimate the video keypoints.
    """

    VALID_OUT_FRAMES = (8, 16)

    def __init__(self, in_channels, out_frames, out_size, kpp_name='Keypoint Predictor'):
        """
        Initializes the Keypoint Predictor network.
        :param in_channels: (int) The number of channels in the input tensor.
        :param out_frames: (int) The number of frames desired in the generated output video.
                            Legal values: 8, 16
        :param kpp_name: (str, optional) The name of the network (default 'Keypoint Predictor').
        Raises:
            ValueError: if 'out_frames' is not a legal value.
        """
        if out_frames not in self.VALID_OUT_FRAMES:
            raise ValueError('Invalid number of frames in desired output: %d' % out_frames)

        super(KPPredictor, self).__init__()
        self.kpp_name = kpp_name

        self.out_frames = out_frames
        self.out_size = out_size

        # definition of all network layers
        self.conv3d_1a = nn.Conv3d(in_channels=in_channels, out_channels=256, kernel_size=(3, 3, 3),
                                   stride=(1, 1, 1), padding=(1, 1, 1))
        self.relu_1a = nn.ReLU(inplace=True)
        self.conv3d_1b = nn.Conv3d(in_channels=256, out_channels=128, kernel_size=(3, 3, 3),
                                   stride=(1, 1, 1), padding=(1, 1, 1))
        self.relu_1b = nn.ReLU(inplace=True)

        self.conv3d_2a = nn.Conv3d(in_channels=128, out_channels=64, kernel_size=(3, 3, 3),
                                   stride=(1, 1, 1), padding=(1, 1, 1))
        self.relu_2a = nn.ReLU(inplace=True)
        self.conv3d_2b = nn.Conv3d(in_channels=64, out_channels=32, kernel_size=(3, 3, 3),
                                   stride=(1, 1, 1), padding=(1, 1, 1))
        self.relu_2b = nn.ReLU(inplace=True)

        self.softmax_layer = nn.Softmax(dim=3)

        # print('%s Model Successfully Built \n' % self.deconv_name)

    def forward(self, x):
        """
        Function to compute a single forward pass through the network, according to the architecture.
        :param x: (tensor) The input tensor from which to generate the keypoints.
                   Must be a tensor of shape: (bsz, 256, 4, 7, 7) for this application.
        :return: A tensor representing the keypoints of the input video.
                 Shape of output is: (bsz, 32, 8/16, 28, 28) for this application.
        """
        x = self.relu_1a(self.conv3d_1a(x))
        x = self.relu_1b(self.conv3d_1b(x))

        x = f.interpolate(x, size=(int(self.out_frames / 2), 14, 14), mode='trilinear')
        x = self.relu_2a(self.conv3d_2a(x))
        x = self.relu_2b(self.conv3d_2b(x))

        x = f.interpolate(x, size=(self.out_frames, self.out_size, self.out_size), mode='trilinear')

        x = self.softmax(x)  # output: bsz, #kp, frames, height, width

        x = self.get_kp_heatmaps(x)

        return x

    def softmax(self, x):
        bsz, channels, frames, height, width = x.size()
        x = torch.reshape(x, (bsz, channels, frames, height * width))
        x = self.softmax_layer(x)  # dim=flattened height x width
        x = torch.reshape(x, (bsz, channels, frames, height, width))

        return x

    def get_kp_heatmaps(self, x):
        bsz, channels, frames, height, width = x.size()
        heatmaps = torch.zeros(bsz, channels, frames, height, width)
        for f in range(frames):
            tensor = x[:, :, f, :, :]
            tensor = torch.squeeze(tensor, dim=2)
            tensor = tensor.permute(0, 2, 3, 1)  # now bsz,h,w,c

            def get_coord(other_axis, axis_size):
                # get "x-y" coordinates:
                g_c_prob = tf.reduce_mean(x, axis=other_axis)  # B,W,NMAP
                g_c_prob = tf.nn.softmax(g_c_prob, axis=1)  # B,W,NMAP
                coord_pt = tf.to_float(tf.linspace(-1.0, 1.0, axis_size))  # W
                coord_pt = tf.reshape(coord_pt, [1, axis_size, 1])
                g_c = tf.reduce_sum(g_c_prob * coord_pt, axis=1)
                return g_c, g_c_prob

            gauss_y, gauss_y_prob = get_coord(2, height)  # B,NMAP
            gauss_x, gauss_x_prob = get_coord(1, width)  # B,NMAP
            gauss_mu = tf.stack([gauss_y, gauss_x], axis=2)

            gauss_xy_ = KPPredictor.get_gaussian_maps(gauss_mu, [self.out_size, self.out_size],
                                                      # choose some fixed std
                                                      inv_std=0.1,  # 1.0 / self._config.gauss_std,
                                                      mode='ankush')  # [B, H, W, NMAPS]
            gauss_xy_.permute(0, 3, 1, 2)
            gauss_xy_ = torch.unsqueeze(gauss_xy_, dim=2)  # bsz, nmaps, 1, h, w

            heatmaps[:, :, f, :, :] = gauss_xy_

    '''def get_kp_heatmaps(self, x):
        gauss_y, gauss_y_prob = KPPredictor.get_coord(x=x, other_axis=5, axis_size=self.out_size)
        gauss_x, gauss_x_prob = KPPredictor.get_coord(x=x, other_axis=4, axis_size=self.out_size)

        gauss_mu = tf.stack([gauss_y, gauss_x], axis=2)

        gauss_xy = []
        gauss_xy_ = KPPredictor.get_gaussian_maps(mu=gauss_mu, shape_hw=[self.out_size, self.out_size],
                                                  # come up with some fixed std to use
                                                  inv_std=0.1, mode='ankush')
        gauss_xy.append(gauss_xy_)

    # SHOULD BE DONE
    # adapted version: added x param as input tensor, added temporal dim
    @staticmethod
    def get_coord(x, other_axis, axis_size):
        # get "x-y" coordinates:
        g_c_prob = tf.reduce_mean(x, axis=other_axis)  # B,NMAP,FRAMES,W
        g_c_prob = tf.nn.softmax(g_c_prob, axis=3)  # B,NMAP,FRAMES,W
        coord_pt = tf.to_float(tf.linspace(-1.0, 1.0, axis_size))  # W
        coord_pt = tf.reshape(coord_pt, [1, 1, 1, axis_size])
        g_c = tf.reduce_sum(g_c_prob * coord_pt, axis=3)
        return g_c, g_c_prob'''

    # https: // github.com / tomasjakab / imm / blob / dev / imm / models / imm_model.py
    @staticmethod
    def pose_encoder(self, x, training_pl, n_maps=1, filters=32,
                     gauss_mode='ankush', map_sizes=None,
                     reuse=False, var_device='/cpu:0'):
        """
        Regresses a N_MAPSx2 (2 = (row, col)) tensor of gaussian means.
        These means are then used to generate 2D "heat-maps".
        Standard deviation is assumed to be fixed.
        """
        with tf.variable_scope('pose_encoder', reuse=reuse):
            opts = self._get_opts(training_pl)
            block_features = self.encoder(x, training_pl, var_device=var_device)
            x = block_features[-1]

            xshape = x.shape.as_list()
            x = self.conv(x, n_maps, [1, 1], opts, stride=1, batch_norm=False,
                          var_device=var_device, activation=None, name='conv_1')

            tf.add_to_collection('tensors', ('heatmaps', x))

            def get_coord(other_axis, axis_size):
                # get "x-y" coordinates:
                g_c_prob = tf.reduce_mean(x, axis=other_axis)  # B,W,NMAP
                g_c_prob = tf.nn.softmax(g_c_prob, axis=1)  # B,W,NMAP
                coord_pt = tf.to_float(tf.linspace(-1.0, 1.0, axis_size))  # W
                coord_pt = tf.reshape(coord_pt, [1, axis_size, 1])
                g_c = tf.reduce_sum(g_c_prob * coord_pt, axis=1)
                return g_c, g_c_prob

            xshape = x.shape.as_list()
            gauss_y, gauss_y_prob = get_coord(2, xshape[1])  # B,NMAP
            gauss_x, gauss_x_prob = get_coord(1, xshape[2])  # B,NMAP
            gauss_mu = tf.stack([gauss_y, gauss_x], axis=2)

            tf.add_to_collection('tensors', ('gauss_y_prob', gauss_y_prob))
            tf.add_to_collection('tensors', ('gauss_x_prob', gauss_x_prob))

            gauss_xy = []
            for map_size in map_sizes:
                gauss_xy_ = get_gaussian_maps(gauss_mu, [map_size, map_size],
                                              1.0 / self._config.gauss_std,
                                              mode=gauss_mode)
                gauss_xy.append(gauss_xy_)

            return gauss_mu, gauss_xy

    # https: // github.com / tomasjakab / imm / blob / dev / imm / models / imm_model.py
    @staticmethod
    def get_gaussian_maps(mu, shape_hw, inv_std, mode='ankush'):
        """
        Generates [B,SHAPE_H,SHAPE_W,NMAPS] tensor of 2D gaussians,
        given the gaussian centers: MU [B, NMAPS, 2] tensor.
        STD: is the fixed networks dev.
        """
        with tf.name_scope(None, 'gauss_map', [mu]):
            mu_y, mu_x = mu[:, :, 0:1], mu[:, :, 1:2]

            y = tf.to_float(tf.linspace(-1.0, 1.0, shape_hw[0]))

            x = tf.to_float(tf.linspace(-1.0, 1.0, shape_hw[1]))

        if mode in ['rot', 'flat']:
            mu_y, mu_x = tf.expand_dims(mu_y, -1), tf.expand_dims(mu_x, -1)

            y = tf.reshape(y, [1, 1, shape_hw[0], 1])
            x = tf.reshape(x, [1, 1, 1, shape_hw[1]])

            g_y = tf.square(y - mu_y)
            g_x = tf.square(x - mu_x)
            dist = (g_y + g_x) * inv_std ** 2

            if mode == 'rot':
                g_yx = tf.exp(-dist)
            else:
                g_yx = tf.exp(-tf.pow(dist + 1e-5, 0.25))

        elif mode == 'ankush':
            y = tf.reshape(y, [1, 1, shape_hw[0]])
            x = tf.reshape(x, [1, 1, shape_hw[1]])

            g_y = tf.exp(-tf.sqrt(1e-4 + tf.abs((mu_y - y) * inv_std)))
            g_x = tf.exp(-tf.sqrt(1e-4 + tf.abs((mu_x - x) * inv_std)))

            g_y = tf.expand_dims(g_y, axis=3)
            g_x = tf.expand_dims(g_x, axis=2)
            g_yx = tf.matmul(g_y, g_x)  # [B, NMAPS, H, W]

        else:
            raise ValueError('Unknown mode: ' + str(mode))

        g_yx = tf.transpose(g_yx, perm=[0, 2, 3, 1])
        return g_yx


if __name__ == "__main__":
    print_summary = True

    kpp = KPPredictor(in_channels=256, out_frames=16, out_size=14)

    if print_summary:
        summary(kpp, input_size=(256, 4, 7, 7))
