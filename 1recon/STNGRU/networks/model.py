# phase 3
import torch
import torch.nn as nn
from networks.modifiedVGG import vgg16
from networks.modifiedI3D import InceptionI3d
from networks.expander import Expander
from networks.transformer import Transformer
# from networks.roipool import ROIPooling2d as roi_pool
# from networks.STN import STN
from networks.kpp import KPPredictor
from networks.vpp import VPPredictor
from networks.convGRU import ConvGRU
from networks.generator import Generator
from torchsummary import summary

"""
Pipeline:
    i1 = single frame view2
    i2 = 8 frames view1
    i3 = viewpoint change

    rep = I3D(i2)
    vp = expander(i3)
    rep' = trans(rep + vp)
    kp' = kpp(rep')

    app = VGG(i1)
    app' = gru(app, kp')

    recon = gen(app' + kp')
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

    def __init__(self, vp_value_count, stdev, output_shape, name='Full Network'):
        """
        Initializes the Full Network.
        :param vp_value_count: (int) The number of values that identify the viewpoint.
        :param output_shape: (5-tuple) The desired output shape for generated videos. Must match video input shape.
                              Legal values: (bsz, 3, 8/16, 112, 112) and (bsz, 3, 16, 112, 112)
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
        self.stdev = stdev
        self.output_shape = output_shape
        self.out_frames = output_shape[2]

        # specs of various features
        self.app_feat = 128
        self.rep_feat = 128
        self.rep_frames = 4
        self.rep_size = 14
        self.nkp = 32

        self.vgg = vgg16(pretrained=True, weights_path=vgg_weights_path)
        self.i3d = InceptionI3d(final_endpoint='Mixed_5c', in_frames=self.out_frames,
                                pretrained=True, weights_path=i3d_weights_path)

        self.exp = Expander(vp_value_count=self.vp_value_count)

        # convs to make all appearance encodings have same number of channels, so they can be used in the same convGRU
        self.app_conv128 = nn.Conv2d(in_channels=128, out_channels=self.app_feat, kernel_size=(3, 3),
                                     stride=(1, 1), padding=(1, 1))
        self.app_conv256a = nn.Conv2d(in_channels=256, out_channels=self.app_feat, kernel_size=(3, 3),
                                      stride=(1, 1), padding=(1, 1))
        self.app_conv256b = nn.Conv2d(in_channels=256, out_channels=self.app_feat, kernel_size=(3, 3),
                                      stride=(1, 1), padding=(1, 1))
        self.app_convs = [nn.Sequential(self.app_conv128),  # , nn.ReLU(inplace=True)),
                          nn.Sequential(self.app_conv256a),  # , nn.ReLU(inplace=True)),
                          nn.Sequential(self.app_conv256b)]  # , nn.ReLU(inplace=True))]

        # convs to make all motion features have the same number of channels, so they can be used in the same trans net
        self.rep_conv64 = nn.Conv3d(in_channels=64, out_channels=self.rep_feat, kernel_size=(3, 3, 3), stride=(1, 1, 1),
                                    padding=(1, 1, 1))
        self.rep_conv192 = nn.Conv3d(in_channels=192, out_channels=self.rep_feat, kernel_size=(3, 3, 3),
                                     stride=(1, 1, 1),
                                     padding=(1, 1, 1))
        self.rep_conv256 = nn.Conv3d(in_channels=256, out_channels=self.rep_feat, kernel_size=(3, 3, 3),
                                     stride=(1, 1, 1),
                                     padding=(1, 1, 1))
        self.rep_convs = [nn.Sequential(self.rep_conv64, nn.ReLU(inplace=True)),
                          nn.Sequential(self.rep_conv192, nn.ReLU(inplace=True)),
                          nn.Sequential(self.rep_conv256, nn.ReLU(inplace=True))]

        self.trans = Transformer(in_channels=self.rep_feat + self.vp_value_count, out_channels=self.rep_feat)

        # self.roi_pool = roi_pool(output_size=(3, 3))
        # self.STN = STN()

        self.kpp = KPPredictor(in_channels=self.rep_feat, nkp=self.nkp, stdev=self.stdev)

        self.kp_poolers = [nn.AvgPool3d(kernel_size=(2, 1, 1), stride=(2, 1, 1)),
                           nn.AvgPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)),
                           nn.AvgPool3d(kernel_size=(4, 4, 4), stride=(4, 4, 4))]

        self.vpp = VPPredictor(in_channels=256)

        self.gru = ConvGRU(input_dim=self.rep_feat, hidden_dim=[self.app_feat], kernel_size=(3, 3),
                           num_layers=1, batch_first=True, bias=False, return_all_layers=False)

        self.gen = Generator(in_channels=[self.app_feat, self.nkp], out_frames=self.out_frames)
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
        :return: The reconstructed video and the transformed motion features.
                 Shape of the output video is: (bsz, 3, out_frames, 112, 112) for this application.
        """
        est_reps_v2, est_kp_v2, gauss_means, rep_v1 = self.action_pipeline(vp_diff, vid1)
        # [bsz,128,8,56,56, bsz,128,8,28,28, bsz,128,4,14,14], bsz,32,16,56,56, bsz,256,4,14,14

        est_apps_v2, app_v2 = self.appearance_pipeline(img2, est_reps_v2, est_kp_v2, gauss_means)
        # [bsz,128,8,56,56, bsz,128,8,28,28, bsz,128,4,14,14], bsz,256,14,14

        vp_est = self.viewpoint_pipeline(app_v2, rep_v1)  # bsz

        # these are the videos that get returned
        gen_v2 = self.gen(est_apps_v2, est_kp_v2)  # bsz,3,out_frames,112,112

        return gen_v2, vp_est

    def action_pipeline(self, vp_diff, vid):
        """
        Function to run all networks that extract motion features of some sort from the input video.
        :param vp_diff: (tensor) The rotational viewpoint change that will be used to transform the motion features.
        :param vid: (tensor) The input video from which to extract features.
        :return: A list of 3 tensors of different sizes that containing the transformed motion features used to extract
                 keypoints, a tensor of gaussian heatmaps representing the keypoints, and a tensor containing the motion
                 features from I3D.
        """
        reps = self.i3d(vid)  # bsz,64,8,56,56, bsz,192,8,28,28, bsz,256,4,14,14
        reps = [self.rep_convs[i](reps[i]) for i in range(len(reps))]

        est_reps = []  # bsz,128,8,56,56, bsz,128,8,28,28, bsz,128,4,14,14
        for i in range(len(reps)):
            r = reps[i]
            # reps, kp, means = self.kpp(r)  # bsz,32,16,56,56
            # r = reps[i]
            bsz, channels, frames, height, width = r.size()
            # r = self.rep_convs[i](r)
            vp_1_to_2 = self.exp(vp_diff, out_frames=frames, out_size=height)
            trans_input2 = torch.cat([vp_1_to_2, r], dim=1)  # dim=channels
            est_reps.append(self.trans(trans_input2))

        est_reps, est_kp, gauss_means = self.kpp(est_reps)  # bsz,32,16,56,56

        return est_reps, est_kp, gauss_means, reps[-1]


    def invert_stn_input_prep(self, r, means):
        bsz, nkp, frames, height, width = means.size()
        means = means.reshape(bsz * nkp, frames, height, width)
        bsz, channels, frames, height, width = r.size()
        r = r.reshape(bsz * frames, channels, height, width)

        # inverse ROI pool
        output = torch.zeros(bsz * frames, channels, height, width)
        for elem in range(bsz * frames):
            y_coord, x_coord = means[elem, 0:1], means[elem, 0:1]
            left, right = x_coord - 2, width - x_coord + 2
            top, bottom = y_coord - 2, height - y_coord + 2
            padded_elem = nn.ZeroPad2d(left, right, top, bottom)
            output = torch.sum(output, padded_elem)

        return output

    def roi_pooling_helper(self, trans_feat, kp):
        kp = nn.Softmax(dim=1)(kp)  # dim=kp
        print(trans_feat.size())
        print(kp.size())
        trans_feat = trans_feat * kp

        return trans_feat

    def appearance_pipeline(self, img, est_reps, est_kp, gauss_means):
        """
        Function to run all networks that extract appearance features from the input image.
        :param img: (tensor) The input image from which to extract features.
        :param est_reps: (tensor) The motion features that will be used to transform the appearance features.
        :return: A list of 3 tensors of different sizes that contain the transformed appearance features and a tensor
                 containing the appearance features from VGG.
        """
        apps = self.vgg(img)  # bsz,128,56,56, bsz,256,28,28, bsz,256,14,14
        apps = [self.app_convs[i](apps[i]) for i in range(len(apps))]

        est_apps = []  # bsz,256,8,56,56, bsz,256,8,28,28, bsz,256,4,14,14
        for i in range(len(apps)):
            a, r = torch.tanh(apps[i]), torch.tanh(est_reps[i])
            kp = self.kp_poolers[i](est_kp)
            # cell_input = torch.cat([r, kp], dim=1).permute(0, 2, 1, 3, 4)  # dim=channels
            output, last_state = self.gru(input_tensor=(r, kp, gauss_means),  # r.permute(0, 2, 1, 3, 4),
                                          hidden_state=[a])
            output = output[0].permute(0, 2, 1, 3, 4)
            trans_app = output.to(device)
            est_apps.append(trans_app)

        return est_apps, apps[-1]

    def viewpoint_pipeline(self, app, rep):
        """
        Function to run the network to estimate the rotational viewpoint change.
        :param app: (tensor) The VGG appearance features extracted from the input image.
        :param rep: (tensor) The I3D motion features extracted from the input video.
        :return: A tensor of scalars representing the estimated rotational viewpoint change.
        """
        rep = nn.AvgPool3d(kernel_size=(4, 1, 1), stride=(4, 1, 1))(rep).squeeze()
        vpp_input = torch.cat([app, rep], dim=1)  # dim=channels

        vp_est = self.vpp(vpp_input)  # bsz

        return vp_est


if __name__ == "__main__":
    print_summary = True

    net = FullNetwork(vp_value_count=1, stdev=0.1, output_shape=(20, 3, 8, 112, 112))

    if print_summary:
        summary(net, input_size=[(3, 8, 112, 112), (3, 8, 112, 122), (3, 112, 122), (3, 112, 112), (1)])
