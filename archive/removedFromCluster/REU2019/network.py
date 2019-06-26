import torch
import torch.nn as nn
from modifiedVGG import VGG, vgg16
from modifiedI3D import InceptionI3d
from generator import Generator

"""
Pipeline:
    i1 = single frame view2
    i2 = 8 frames view1
    o1 = VGG(i1)
    o2 = I3D(i2)
    o3 = GEN(o1 + o2)
"""

vgg_weights_path = '/home/yogesh/kara/REU2019/weights/vgg16-397923af.pth'  # 'C:/Users/Owner/Documents/UCF/Project/REU2019/weights/vgg16-397923af.pth'
i3d_weights_path = '/home/yogesh/kara/REU2019/weights/rgb_charades.pt'  # 'C:/Users/Owner/Documents/UCF/Project/REU2019/weights/rgb_charades.pt'


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

        # self.vgg = VGG('VGG16')
        self.vgg = vgg16()
        self.vgg.load_state_dict(torch.load(vgg_weights_path))
        self.i3d = InceptionI3d(final_endpoint='Mixed_5c', in_frames=output_shape[2])
        self.i3d.load_state_dict(torch.load(i3d_weights_path))
        self.gen = Generator(in_channels=1536, out_frames=output_shape[2])
        # print('%s Model Successfully Built \n' % self.net_name)

    def forward(self, vid1, vid2, img1, img2):
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
        :return: The two output videos, one from each viewpoint, and the two representation feature maps from I3D.
                 Shape of two output videos is: (bsz, 3, 8/16, 112, 112) for this application.
                 Shape of two representation feature maps is: (bsz, 1024, 7, 7) for this application.
        """
        app_v1 = self.vgg(img1)
        app_v2 = self.vgg(img2)
        # print('VGG runs complete.')
        # add the temporal dimension so they can be concatenated with rep_v1 and rep_v2
        # app_v1 = torch.unsqueeze(app_v1, dim=2)
        # app_v2 = torch.unsqueeze(app_v2, dim=2)

        rep_v1 = self.i3d(vid1)
        rep_v2 = self.i3d(vid2)
        # print('I3D runs complete.')

        # these are the representation feature maps that get returned
        # eliminate the temporal dimension so they can be concatenated with app_v1 and app_v2
        rep_v1 = torch.squeeze(rep_v1, dim=2)
        rep_v2 = torch.squeeze(rep_v2, dim=2)
        # print('Eliminated temporal dimension.')

        # appearance encoding + video representation
        # input1 = torch.cat([app_v1, rep_v2], dim=2)
        # input2 = torch.cat([output1_v2, rep_v1], dim=2)

        input1 = torch.cat([app_v1, rep_v2], dim=1)
        input2 = torch.cat([app_v2, rep_v1], dim=1)

        # these are the videos that get returned
        output_v1 = self.gen(input1)
        output_v2 = self.gen(input2)
        # print('Generator runs complete.')

        return output_v1, output_v2, rep_v1, rep_v2

    '''def forward(self, vid1, vid2):
        """
        Function to compute a single forward pass through the network, according to the architecture.
        :param vid1: (tensor) A video of the scene from the first viewpoint.
                      Must be a tensor of shape: (bsz, 3, 8/16, 112, 112) for this application.
        :param vid2: (tensor) A video of the scene from the second viewpoint.
                      Must be a tensor of shape: (bsz, 3, 8/16, 112, 112) for this application.
        :return: The two output videos, one from each viewpoint, and the two representation feature maps from I3D.
                 Shape of two output videos is: (bsz, 3, 8/16, 112, 112) for this application.
                 Shape of two representation feature maps is: (bsz, 1024, 7, 7) for this application.
        """
        # extract first frame of each vid for appearance conditioning
        img1, img2 = FullNetwork.get_first_frame(vid1), FullNetwork.get_first_frame(vid2)
        img1, img2 = img1.to(device), img2.to(device)

        app_v1 = self.vgg(img1)
        app_v2 = self.vgg(img2)
        # print('VGG runs complete.')
        # add the temporal dimension so they can be concatenated with rep_v1 and rep_v2
        # app_v1 = torch.unsqueeze(app_v1, dim=2)
        # app_v2 = torch.unsqueeze(app_v2, dim=2)

        rep_v1 = self.i3d(vid1)
        rep_v2 = self.i3d(vid2)
        # print('I3D runs complete.')

        # these are the representation feature maps that get returned
        # eliminate the temporal dimension so they can be concatenated with app_v1 and app_v2
        rep_v1 = torch.squeeze(rep_v1, dim=2)
        rep_v2 = torch.squeeze(rep_v2, dim=2)

        # appearance encoding + video representation
        # input1 = torch.cat([app_v1, rep_v2], dim=2)
        # input2 = torch.cat([output1_v2, rep_v1], dim=2)

        # appearance encoding + video representation
        input1 = torch.cat([app_v1, rep_v2], dim=1)
        input2 = torch.cat([app_v2, rep_v1], dim=1)

        # these are the videos that get returned
        output_v1 = self.gen(input1)
        output_v2 = self.gen(input2)
        # print('Generator runs complete.')

        return output_v1, output_v2, rep_v1, rep_v2'''

    '''@staticmethod
    def get_first_frame(vid_batch):
        """
        Function to extract the first frame from a batch of input videos.
        We extract the first frame from each of the videos input to the network so that the network can learn appearance
        conditioning from the desired views.
        :param vid_batch: (tensor) A batch of videos from which to extract only the first frame of each.
        :return: A tensor that holds all the first frames.
        """
        # get the first frame fom each vid in the batch and eliminate temporal dimension
        frames = [torch.squeeze(vid[:, :1, :, :]) for vid in vid_batch]
        # extract the batch size from the input vid_batch
        batch_size = vid_batch.size()[0]
        # create empty tensor containing batch_size images of the correct shape (matching the frames)
        imgs = torch.zeros(batch_size, *frames[0].size())
        # put all the first frames into the tensor
        for sample in range(batch_size):
            imgs[sample] = frames[sample]

        return imgs'''
