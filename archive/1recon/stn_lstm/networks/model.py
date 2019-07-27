# phase 3
64262
import torch
import torch.nn as nn
from networks.modifiedVGG import vgg16
from networks.modifiedI3D import InceptionI3d
from networks.expander import Expander
from networks.transformer import Transformer
from networks.convLSTM import ConvLSTM
from networks.generator import Generator
from torchsummary import summary

"""
Pipeline:
    i1 = single frame view2
    i2 = 8 frames view1
    i3 = viewpoint change
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

        # params
        self.net_name = name
        self.vp_value_count = vp_value_count
        self.output_shape = output_shape
        self.out_frames = output_shape[2]
        self.rep_feat = 128
        self.app_feat = 256

        # networks
        self.vgg = vgg16(pretrained=True, weights_path=vgg_weights_path)
        self.i3d = InceptionI3d(final_endpoint='Mixed_5c', in_frames=self.out_frames,
                                pretrained=True, weights_path=i3d_weights_path)
        self.exp = Expander(vp_value_count=self.vp_value_count)
        self.trans = Transformer(in_channels=self.rep_feat + self.vp_value_count, out_channels=self.rep_feat)
        self.gen = Generator(in_channels=[self.app_feat, self.rep_feat], out_frames=self.out_frames)

        self.conv_lstms = {
            56: ConvLSTM(input_dim=self.rep_feat, hidden_dim=[self.app_feat], kernel_size=(3, 3), num_layers=1,
                         in_shape=(56, 56), batch_first=True, bias=False, return_all_layers=False),
            28: ConvLSTM(input_dim=self.rep_feat, hidden_dim=[self.app_feat], kernel_size=(3, 3), num_layers=1,
                         in_shape=(28, 28), batch_first=True, bias=False, return_all_layers=False),
            14: ConvLSTM(input_dim=self.rep_feat, hidden_dim=[self.app_feat], kernel_size=(3, 3), num_layers=1,
                         in_shape=(14, 14), batch_first=True, bias=False, return_all_layers=False)
        }

        # convs
        self.app_conv128 = nn.Conv2d(in_channels=128, out_channels=self.app_feat, kernel_size=(3, 3),
                                     stride=(1, 1), padding=(1, 1))
        self.app_conv256 = nn.Conv2d(in_channels=256, out_channels=self.app_feat, kernel_size=(3, 3),
                                     stride=(1, 1), padding=(1, 1))
        self.app_conv512 = nn.Conv2d(in_channels=512, out_channels=self.app_feat, kernel_size=(3, 3),
                                     stride=(1, 1), padding=(1, 1))
        self.app_convs = {128: self.app_conv128,
                          256: self.app_conv256,
                          512: self.app_conv512}

        self.hconv = nn.Conv2d(in_channels=self.app_feat, out_channels=128, kernel_size=(3, 3),
                               stride=(1, 1), padding=(1, 1))
        self.cconv = nn.Conv2d(in_channels=self.app_feat, out_channels=128, kernel_size=(3, 3),
                               stride=(1, 1), padding=(1, 1))

        self.rep_conv64 = nn.Conv3d(in_channels=64, out_channels=self.rep_feat, kernel_size=(3, 3, 3),
                                    stride=(1, 1, 1), padding=(1, 1, 1))
        self.rep_conv192 = nn.Conv3d(in_channels=192, out_channels=self.rep_feat, kernel_size=(3, 3, 3),
                                     stride=(1, 1, 1), padding=(1, 1, 1))
        self.rep_conv256 = nn.Conv3d(in_channels=256, out_channels=self.rep_feat, kernel_size=(3, 3, 3),
                                     stride=(1, 1, 1), padding=(1, 1, 1))
        self.rep_convs = {64: self.rep_conv64,
                          192: self.rep_conv192,
                          256: self.rep_conv256}

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
        :return: The two output videos, one from each viewpoint, the two keypoint feature maps, and the two transformed
                 keypoint feature maps.
                 Shape of two output videos is: (bsz, 3, 8/16, 112, 112) for this application.
                 Shape of two keypoint feature maps is: (bsz, 32, 16, 28, 28) for this application.
        """
        rep_v2_est = self.action_pipeline(vp_diff, vid1)  # bsz,128,8,56,56, bsz,128,8,28,28, bsz,128,4,14,14

        app_v2_est = self.appearance_pipeline(img2, rep_v2_est)  # bsz,128,8,56,56, bsz,128,8,28,28, bsz,128,4,14,14

        gen_v2 = self.gen(app_v2_est, rep_v2_est)  # bsz,3,16,112,112

        return gen_v2

    def action_pipeline(self, vp_diff, vid1):
        rep_v1 = self.i3d(vid1)

        rep_v2_est = []
        for rep in rep_v1:
            bsz, channels, frames, height, width = rep.size()
            rep = self.rep_convs[channels](rep)  # bsz,128,f,h,w
            vp_1_to_2 = self.exp(vp_diff, out_frames=frames, out_size=(height, width))  # bsz,vp_count,f,h,w
            trans_input2 = torch.cat([vp_1_to_2, rep], dim=1)  # dim=channels
            rep_est = self.trans(trans_input2)  # bsz,128,f,h,w
            rep_v2_est.append(rep_est)

        return rep_v2_est  # bsz,128,8,56,56, bsz,128,8,28,28, bsz,128,4,14,14

    def appearance_pipeline(self, img2, rep_v2_est):
        app_v2 = self.vgg(img2)  # bsz,128,56,56, bsz,256,28,28, bsz,512,14,14

        app_v2_est = []
        for i in range(len(app_v2)):
            bsz, channels, height, width = app_v2[i].size()
            app = self.app_convs[channels](app_v2[i])  # bsz,256,56,56, bsz,256,28,28, bsz,256,14,14
            h, c = self.hconv(app), self.cconv(app)  # bsz,128,56,56, bsz,128,28,28, bsz,128,14,14
            output, last_state = self.conv_lstms[height](input_tensor=rep_v2_est[i].permute(0, 2, 1, 3, 4),
                                                         hidden_state=[(h, c)])
            output = output[0].permute(0, 2, 1, 3, 4)

            trans_app = output.to(device)
            app_v2_est.append(trans_app)

        return app_v2_est  # bsz,128,8,56,56, bsz,128,8,28,28, bsz,128,4,14,14


if __name__ == "__main__":
    print_summary = True

    net = FullNetwork(vp_value_count=1, output_shape=(20, 3, 8, 112, 112))

    if print_summary:
        summary(net, input_size=[(3, 8, 112, 112), (3, 8, 112, 122), (3, 112, 122), (3, 112, 112), (1)])
