# phase 3

import torch
import torch.nn as nn
import torch.nn.functional as f
from .modifiedVGG import vgg16
from .modifiedI3D import InceptionI3d
from .kpp import KPPredictor
from .expander import Expander
from .transformer import Transformer
from .generator import Generator
from torchsummary import summary

"""
Pipeline:
    i1 = single frame view2
    i2 = 8 frames view1
    i3 = viewpoint change

    app = VGG(i1)

    rep = I3D(i2)
    vp = expander(i3)
    rep' = trans(rep + vp)

    o_con = GEN(app_2 + rep)
    o_real = GEN(app + rep')
"""

vgg_weights_path = '/home/yogesh/kara/REU2019/weights/vgg16-397923af.pth'
# 'C:/Users/Owner/Documents/UCF/Project/REU2019/weights/vgg16-397923af.pth'
i3d_weights_path = '/home/yogesh/kara/REU2019/weights/rgb_charades.pt'
# 'C:/Users/Owner/Documents/UCF/Project/REU2019/weights/rgb_charades.pt'

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class FullNetwork(nn.Module):
    """
    Class combining the full Cross-View Action Synthesis network architecture.
    """

    VALID_VP_VALUE_COUNTS = (1, 3)
    VALID_FRAME_COUNTS = (8, 16)

    def __init__(self, vp_value_count, output_shape, stdev, name='Full Network'):
        """
        Initializes the Full Network.
        :param output_shape: (5-tuple) The desired output shape for generated videos. Must match video input shape.
                              Legal values: (bsz, 3, 8, 112, 112) and (bsz, 3, 16, 112, 112)
        :param name: (str, optional) The name of the network (default 'Full Network').
        Raises:
            ValueError: if 'vp_value_count' is not a legal value count
            ValueError: if 'output_shape' does not contain a legal number of frames.
        """
        if vp_value_count not in self.VALID_VP_VALUE_COUNTS:
            raise ValueError('Invalid number of vp values: %d' % vp_value_count)
        if output_shape[2] not in self.VALID_FRAME_COUNTS:
            raise ValueError('Invalid number of frames in desired output: %d' % output_shape[2])

        super(FullNetwork, self).__init__()

        self.net_name = name
        self.vp_value_count = vp_value_count
        # self.output_shape = output_shape
        self.out_frames = output_shape[2]
        # self.stdev = stdev

        self.vgg = vgg16(pretrained=True, weights_path=vgg_weights_path)
        self.i3d = InceptionI3d(final_endpoint='Mixed_5c_small', in_frames=self.out_frames,
                                pretrained=True, weights_path=i3d_weights_path)

        self.kpp = KPPredictor(stdev=stdev)

        self.exp = Expander(vp_value_count=self.vp_value_count, out_frames=4, out_size=14)
        self.trans = Transformer(in_channels=32 + self.vp_value_count)

        self.gen = Generator(in_channels=32 + 32 + 32, out_frames=self.out_frames)
        # print('%s Model Successfully Built \n' % self.net_name)

    def forward(self, vp_diff, vid1, vid2, img1, img2):
        """
        Function to compute a single forward pass through the network, according to the architecture.
        :param vp_diff (tensor) The difference between the two viewpoints i.e. the vp change to get from v1 to v2.
                        A scalar value for the NTU Dataset; a 3-tuple for the panoptic Dataset.
                        Must be a tensor of shape: (bsz, 1/3) for this application.
        :param vid1: (tensor) A video of the scene from the first viewpoint.
                      Must be a tensor of shape: (bsz, 3, 8/16, 112, 112) for this application.
        :param vid2: (tensor) A video of the scene from the second viewpoint.
                      Must be a tensor of shape: (bsz, 3, 8/16, 112, 112) for this application.
        :param img1: (tensor) An image of the scene from the first viewpoint to use for appearance conditioning.
                      Must be a tensor of shape: (bsz, 3, 112, 112) for this application.
        :param img2: (tensor) An image of the scene from the second viewpoint to use for appearance conditioning.
                      Must be a tensor of shape: (bsz, 3, 112, 112) for this application.
        :return: The two output videos, one from each viewpoint and key-point heatmaps.
                 Shape of two output videos is: (bsz, 3, 8/16, 112, 112) for this application.
                 Shape of two key-point feature maps is: (bsz, 32, 16, 28, 28) for this application.
        """
        vp1_to_2, vp2_to_1 = self.exp(vp_diff), self.exp(-vp_diff)  # bsz,1/3,14,14

        exp_app_v1, exp_app_v2 = self.appearance_pipeline(img1, img2)  # bsz,32,16,14,14

        action_output = self.action_pipeline(vp1_to_2, vp2_to_1, vid1, vid2)  # bsz,32,8/16,14,14
        rep_v1, rep_v2, rep_v1_est, rep_v2_est, kp_v1, kp_v2, kp_v1_est, kp_v2_est = action_output

        # appearance encoding + video features
        gen_input1, gen_input2 = torch.cat([exp_app_v1, rep_v1_est, kp_v1_est], dim=1), \
                                 torch.cat([exp_app_v2, rep_v2_est, kp_v2_est], dim=1)  # dim=channels

        # these are the videos that get returned
        gen_v1, gen_v2 = self.gen(gen_input1), self.gen(gen_input2)  # bsz,3,8/16,112,112

        return gen_v1, gen_v2, rep_v1, rep_v2, rep_v1_est, rep_v2_est, kp_v1, kp_v2, kp_v1_est, kp_v2_est

    def appearance_pipeline(self, img1, img2):
        app_v1, app_v2 = self.vgg(img1), self.vgg(img2)  # bsz,32,14,14

        exp_app_v1, exp_app_v2 = self.expand_app_enc(app_v1, 4), self.expand_app_enc(app_v2, 4)  # bsz,32,8/16,14,14

        return exp_app_v1, exp_app_v2

    def action_pipeline(self, vp1_to_2, vp2_to_1, vid1, vid2):
        rep_v1, rep_v2 = self.i3d(vid1), self.i3d(vid2)  # bsz,32,4,14,14

        kp_v1 = self.kpp(rep_v1)  # bsz,32,16,14,14; bsz,32,16,28,28
        kp_v2 = self.kpp(rep_v2)

        # viewpoint change + action representation
        trans_input1, trans_input2 = torch.cat([vp2_to_1, rep_v2], dim=1), torch.cat([vp1_to_2, rep_v1], dim=1)  # dim=c

        rep_v1_est, rep_v2_est = self.trans(trans_input1), self.trans(trans_input2)  # bsz,32,8/16,14,14

        kp_v1_est = self.kpp(rep_v1_est)  # bsz,32,16,14,14; bsz,32,16,28,28
        kp_v2_est = self.kpp(rep_v2_est)

        return rep_v1, rep_v2, rep_v1_est, rep_v2_est, kp_v1, kp_v2, kp_v1_est, kp_v2_est

    '''def transform_kp(self, vp, kp):
        bsz, channels, frames, height, width = kp.size()
        buffer = torch.zeros(bsz, channels, frames, height, width)
        for i in range(frames):
            kp_frame = torch.squeeze(kp[:, :, i, :, :], dim=2)  # eliminate temporal dim
            # key points + view point
            trans_input = torch.cat([vp, kp_frame], dim=1)  # dim=channels
            kp_frame_est = self.trans(trans_input)  # bsz,32,14,14
            buffer[:, :, i, :, :] = kp_frame_est
        buffer = buffer.to(device)

        return buffer'''

    def expand_app_enc(self, app, frames):
        """
        Function to repeat the appearance encoding along the depth. Makes the depth equivalent to the input video depth.
        :param app: (tensor) The appearance encoding to expand.
        :return: The expanded appearance encoding.
        """
        bsz, channels, height, width = app.size()
        buffer = torch.zeros(bsz, channels, frames, height, width)
        for frame in range(frames):
            buffer[:, :, frame, :, :] = app
        buffer = buffer.to(device)

        return buffer


if __name__ == "__main__":
    print_summary = True

    net = FullNetwork(vp_value_count=1, output_shape=(20, 3, 8, 112, 112), stdev=0.1)

    if print_summary:
        summary(net, input_size=[(3, 8, 112, 112), (3, 8, 112, 122), (3, 112, 122), (3, 112, 122), (1)])
