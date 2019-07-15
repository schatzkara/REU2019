# phase 3

import torch
import torch.nn as nn
from torchsummary import summary

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class Expander(nn.Module):
    """
    Class representing the Expander network to be used to expand the viewpoint ID to the desired size.
    """

    def __init__(self, vp_value_count, out_frames, out_size, ex_name='Viewpoint Expander'):
        """
        Initializes the Expander network.
        :param vp_value_count: (int) The number of values that identify the viewpoint
        :param out_size: (int) The height and width desired in the output viewpoint tensor.
        :param ex_name: (str, optional) The name of the network (default 'Viewpoint Expander').
        """
        super(Expander, self).__init__()
        self.ex_name = ex_name

        self.vp_value_count = vp_value_count
        self.out_frames = out_frames
        self.out_size = out_size

        # definition of all network layers
        self.conv3d_1a = nn.Conv3d(in_channels=self.vp_value_count, out_channels=4, kernel_size=(3, 3, 3),
                                   stride=(1, 1, 1), padding=(1, 1, 1))
        self.conv3d_1b = nn.Conv3d(in_channels=4, out_channels=self.vp_value_count, kernel_size=(3, 3, 3),
                                   stride=(1, 1, 1), padding=(1, 1, 1))

    # print('%s Model Successfully Built \n' % self.ex_name)

    def forward(self, x):
        """
        Function to compute a single forward pass through the network, according to the architecture.
        :param x: (tensor) The input tensor (viewpoint ID) to expand.
                   Must be a tensor of shape: (bsz, 1/3) for this application.
        :return: A tensor representing the viewpoint.
                 Shape of output is: (bsz, 1/3, 8/16, 28, 28) for this application.
        """
        x = self.expand_vp(x)

        x = self.conv3d_1a(x)
        x = self.conv3d_1b(x)

        return x

    def expand_vp(self, vp):
        """
        Function to expand the size of the viewpoint to the desired size.
        :param vp: (tensor) The input tensor (viewpoint ID) to expand.
        :return: The expanded viewpoint tensor.
        """
        bsz = vp.size()[0]

        if self.vp_value_count == 1:
            vp = torch.unsqueeze(vp, dim=1)

        buffer = torch.zeros(bsz, self.vp_value_count, self.out_frames, self.out_size, self.out_size)
        for f in range(self.out_frames):
            for h in range(self.out_size):
                for w in range(self.out_size):
                    buffer[:, :, f, h, w] = vp
        buffer = buffer.to(device)

        return buffer


if __name__ == "__main__":
    print_summary = True

    ex = Expander(vp_value_count=1, out_frames=4, out_size=14)

    if print_summary:
        summary(ex, input_size=(3))
