# phase 3

import torch
import torch.nn as nn
from networks.modifiedVGG import vgg16
from networks.modifiedI3D import InceptionI3d
from networks.expander import Expander
from networks.transformer import Transformer
from networks.generator import Generator
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

    recon = gen(app_2 + rep')
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

    def __init__(self, vp_value_count, output_shape, name='Full Network'):
        """
        Initializes the Full Network.
        :param vp_value_count: (int) The number of values that identify the viewpoint.
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
        self.output_shape = output_shape
        self.out_frames = output_shape[2]

        # specs of the motion features from I3D
        self.rep_channels = 256
        self.rep_frames = 4
        self.rep_size = 14

        self.vgg = vgg16(pretrained=True, weights_path=vgg_weights_path)
        self.i3d = InceptionI3d(final_endpoint='Mixed_5c_small', in_frames=self.out_frames,
                                pretrained=True, weights_path=i3d_weights_path)

        self.exp = Expander(vp_value_count=self.vp_value_count, out_frames=self.rep_frames, out_size=self.rep_size)
        self.trans = Transformer(in_channels=self.rep_channels + self.vp_value_count, out_channels=self.rep_channels)

        self.gen = Generator(in_channels=self.rep_channels + self.rep_channels, out_frames=self.out_frames)
        # print('%s Model Successfully Built \n' % self.net_name)

    def forward(self, vp_diff, vid1, img2):
        """
        Function to compute a single forward pass through the network, according to the architecture.
        :param vp_diff (tensor) The difference between the two viewpoints.
                        A scalar value for the NTU Dataset; a 3-tuple for the panoptic Dataset.
                        Must be a tensor of shape: (bsz, 1/3) for this application.
        :param vid1: (tensor) A video of the scene from the first viewpoint.
                      Must be a tensor of shape: (bsz, 3, 8/16, 112, 112) for this application.
        :param img2: (tensor) An image of the scene from the second viewpoint to use for appearance conditioning.
                      Must be a tensor of shape: (bsz, 3, 112, 112) for this application.
        :return: The reconstructed video and the transformed representation of the motion features.
                 Shape of output video is: (bsz, 3, out_frames, 112, 112) for this application.
                 Shape of motion features is: (bsz, rep_channels, rep_frames, rep_size, rep_size) for this application.
        """
        vp1_to_2 = self.exp(vp_diff)  # bsz,1/3,4,14,14

        exp_app_v2 = self.appearance_pipeline(img2)  # bsz,256,4,14,14

        rep_v2_est = self.action_pipeline(vp1_to_2, vid1)  # bsz,256,4,14,14

        # appearance encoding + video features
        gen_input2 = torch.cat([exp_app_v2, rep_v2_est], dim=1)

        # these are the videos that get returned
        gen_v2 = self.gen(gen_input2)  # bsz,3,8/16,112,112

        return gen_v2, rep_v2_est

    def appearance_pipeline(self, img2):
        app_v2 = self.vgg(img2)  # bsz,256,14,14

        exp_app_v2 = FullNetwork.expand_app_enc(app=app_v2, frames=self.rep_frames)  # bsz,256,4,14,14

        return exp_app_v2

    def action_pipeline(self, vp_1_to_2, vid1):
        rep_v1 = self.i3d(vid1)  # bsz,256,4,14,14

        # viewpoint change + video features
        trans_input2 = torch.cat([vp_1_to_2, rep_v1], dim=1)  # dim=channels

        rep_v2_est = self.trans(trans_input2)  # bsz,256,4,14,14

        return rep_v2_est

    @staticmethod
    def expand_app_enc(app, frames):
        """
        Function to repeat the appearance encoding along the depth. Makes the depth equivalent to the input video depth.
        :param app: (tensor) The appearance encoding to expand.
        :param frames: (int) The number of frames for which to repeat the appearance encoding.
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

    net = FullNetwork(vp_value_count=1, output_shape=(20, 3, 8, 112, 112))

    if print_summary:
        summary(net, input_size=[(3, 8, 112, 112), (3, 8, 112, 122), (3, 112, 122), (3, 112, 122), (1)])
