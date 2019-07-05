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

        self.vgg = vgg16(pretrained=True, weights_path=vgg_weights_path)
        self.i3d = InceptionI3d(final_endpoint='Mixed_5c_small', in_frames=self.out_frames,
                                pretrained=True, weights_path=i3d_weights_path)

        self.deconv = Deconv(in_channels=256, out_frames=self.out_frames)
        self.exp = Expander(vp_value_count=self.vp_value_count, out_frames=self.out_frames, out_size=28)
        self.trans = Transformer(in_channels=32 + self.vp_value_count)

        self.gen = Generator(in_channels=32 + 32, out_frames=self.out_frames)
        # print('%s Model Successfully Built \n' % self.net_name)

    def forward(self, vp_diff, vid1, vid2, img1, img2):
        """
        Function to compute a single forward pass through the network, according to the architecture.
        :param vp_diff (tensor) The difference between the two viewpoints.
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
        :return: The two output videos, one from each viewpoint, the two keypoint feature maps, and the two transformed
                 keypoint feature maps.
                 Shape of two output videos is: (bsz, 3, 8/16, 112, 112) for this application.
                 Shape of two keypoint feature maps is: (bsz, 32, 16, 28, 28) for this application.
        """
        app_v1, app_v2 = self.vgg(img1), self.vgg(img2)  # bsz,32,28,28

        app_v1, app_v2 = self.expand_app_encoding(app=app_v1), self.expand_app_encoding(app=app_v2)  # bsz,32,8/16,28,28

        rep_v1, rep_v2 = self.i3d(vid1), self.i3d(vid2)  # bsz,256,4,7,7

        # these are the keypoint feature maps that get returned
        kp_v1, kp_v2 = self.deconv(rep_v1), self.deconv(rep_v2)  # bsz,32,8/16,28,28

        vp1, vp2 = self.exp(vp_diff), self.exp(-vp_diff)  # bsz,1/3,8/16,28,28

        # key points + view point
        trans_input1, trans_input2 = torch.cat([vp1, kp_v2], dim=1), torch.cat([vp2, kp_v1], dim=1)  # dim=channels

        # these are the transformed keypoint feature maps that get returned
        kp_v1_est, kp_v2_est = self.trans(trans_input1), self.trans(trans_input2)  # bsz,32,8/16,28,28

        # appearance encoding + key points
        gen_input1, gen_input2 = torch.cat([app_v1, kp_v2], dim=1), torch.cat([app_v2, kp_v1], dim=1)  # dim=channels

        # these are the videos that get returned
        output_v1, output_v2 = self.gen(gen_input1), self.gen(gen_input2)  # bsz,3,8/16,112,112

        return output_v1, output_v2, kp_v1, kp_v2, kp_v1_est, kp_v2_est

    def expand_app_encoding(self, app):
        """
        Function to repeat the appearance encoding along the depth. Makes the depth equivalent to the input video depth.
        :param app: (tensor) The appearance encoding to expand.
        :return: The expanded appearance encoding.
        """
        bsz, channels, height, width = app.size()
        buffer = torch.zeros(bsz, channels, self.out_frames, height, width)
        for frame in range(self.out_frames):
            buffer[:, :, frame, :, :] = app
        buffer = buffer.to(device)

        return buffer


# if __name__ == "__main__":
#     print_summary = True
#
#     net = FullNetwork(output_shape=(20, 3, 8, 112, 112))
#
#     if print_summary:
#         summary(net, input_size=[(3, 8, 112, 112), (3, 8, 112, 122), (3, 112, 122), (3, 112, 122), (1)])
