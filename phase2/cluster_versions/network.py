# phase 2

import torch
import torch.nn as nn
from modifiedVGG import vgg16
from modifiedI3D import InceptionI3d
from deconvolutional import Deconv
from expander import Expander
from transformer import Transformer
from generator import Generator
# from torchsummary import summary

"""
Pipeline:
    i1 = single frame view2
    i2 = 8 frames view1
    i3 = viewpoint change, scalar
    app = VGG(i1)
    
    rep = I3D(i2)
    kp = deconv(rep)
    vp = expander(i3)
    kp' = trans(kp + vp)
    
    o3 = GEN(app + kp)
"""

vgg_weights_path = '/home/yogesh/kara/REU2019/weights/vgg16-397923af.pth'
# 'C:/Users/Owner/Documents/UCF/Project/REU2019/weights/vgg16-397923af.pth'
i3d_weights_path = '/home/yogesh/kara/REU2019/weights/rgb_charades.pt'


# 'C:/Users/Owner/Documents/UCF/Project/REU2019/weights/rgb_charades.pt'


class FullNetwork(nn.Module):
    """
    Class combining the full Cross-View Action Synthesis network architecture.
    """

    VALID_FRAME_COUNTS = (8, 16)

    def __init__(self, output_shape, name='Full Network'):
        """
        Initializes the Full Network.
        :param output_shape: (5-tuple) The desired output shape for generated videos. Must match video input shape.
                              Legal values: (bsz, 3, 8, 112, 112) and (bsz, 3, 16, 112, 112)
        :param name: (str, optional) The name of the network (default 'Full Network').
        Raises:
            ValueError: if 'output_shape' does not contain a legal number of frames.
        """
        if output_shape[2] not in self.VALID_FRAME_COUNTS:
            raise ValueError('Invalid number of frames in desired output: %d' % output_shape[2])

        super(FullNetwork, self).__init__()

        self.net_name = name
        self.output_shape = output_shape
        self.out_frames = output_shape[2]

        # self.vgg = VGG('VGG16')
        self.vgg = vgg16()
        self.vgg.load_state_dict(torch.load(vgg_weights_path))
        self.i3d = InceptionI3d(final_endpoint='Mixed_5c', in_frames=self.out_frames)
        self.i3d.load_state_dict(torch.load(i3d_weights_path))

        self.deconv = Deconv(in_channels=1024, out_frames=self.out_frames)
        self.exp = Expander()
        self.trans = Transformer(in_channels=40)

        self.gen = Generator(in_channels=1536, out_frames=self.out_frames)
        # print('%s Model Successfully Built \n' % self.net_name)

    def forward(self, vp_diff, vid1, vid2, img1, img2):
        """
        Function to compute a single forward pass through the network, according to the architecture.
        :param vid1: (tensor) A video of the scene from the first viewpoint.
                      Must be a tensor of shape: (bsz, 3, 8/16, 112, 112) for this application.
        :param vid2: (tensor) A video of the scene from the second viewpoint.
                      Must be a tensor of shape: (bsz, 3, 8/16, 112, 112) for this application.
        :param img1: (tensor) An image of the scene from the first viewpoint to use for appearance conditioning.
                      Must be a tensor of shape: (bsz, 3, 112, 112) for this application.
        :param img2: (tensor) An image of the scene from the second viewpoint to use for appearance conditioning.
                      Must be a tensor of shape: (bsz, 3, 112, 112) for this application.
        :param vp_diff (tensor) The difference between the two viewpoints. A scalar value for the NTU Dataset.
                        Must be a tensor of shape: (bsz, 1) for this application.
        :return: The two output videos, one from each viewpoint, and the two representation feature maps from I3D.
                 Shape of two output videos is: (bsz, 3, 8/16, 112, 112) for this application.
                 Shape of two representation feature maps is: (bsz, 1024, 7, 7) for this application.
        """
        # print(vp_diff.size())
        app_v1, app_v2 = self.vgg(img1), self.vgg(img2)

        app_v1, app_v2 = self.expand_app_encoding(app=app_v1), self.expand_app_encoding(app=app_v2)

        rep_v1, rep_v2 = self.i3d(vid1), self.i3d(vid2)

        kp_v1, kp_v2 = self.deconv(rep_v1), self.deconv(rep_v2)

        vp1, vp2 = self.exp(vp_diff), self.exp(-vp_diff)

        # key points + view point
        trans_input1, trans_input2 = torch.cat([vp1, kp_v2], dim=1), torch.cat([vp2, kp_v1], dim=1)  # dim=channels

        kp_v1_est, kp_v2_est = self.trans(trans_input1), self.trans(trans_input2)

        # appearance encoding + key points
        gen_input1, gen_input2 = torch.cat([app_v1, kp_v2], dim=1), torch.cat([app_v2, kp_v1], dim=1)  # dim=channels

        # these are the videos that get returned
        output_v1, output_v2 = self.gen(gen_input1), self.gen(gen_input2)

        return output_v1, output_v2, kp_v1, kp_v2, kp_v1_est, kp_v2_est

    def expand_app_encoding(self, app):
        bsz, channels, height, width = app.size()
        buffer = torch.zeros(bsz, channels, self.out_frames, height, width)
        for frame in range(self.out_frames):
            buffer[:, :, frame, :, :] = app

        return buffer.to('cuda')


# if __name__ == "__main__":
#     print_summary = True
#
#     net = FullNetwork(output_shape=(20, 3, 8, 112, 112))
#
#     if print_summary:
#         summary(net, input_size=[(3, 8, 112, 112), (3, 8, 112, 122), (3, 112, 122), (3, 112, 122), (1)])
