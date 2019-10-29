import torch
import torch.nn as nn
import torch.nn.functional as f
from torchsummary import summary

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class OFPredictor(nn.Module):
    """
    Class representing the Optical Flow Predictor to be used.
    """

    def __init__(self, in_channels, ofp_name='Optical Flow Predictor'):
        """
        Initializes the Generator network.
        :param in_channels: (int) The number of channels in the input tensor.
        :param ofp_name: (str, optional) The name of the network (default 'Optical Flow Predictor').
        """
        super(OFPredictor, self).__init__()
        self.ofp_name = ofp_name

        # definition of all layer channels
        self.layer_out_channels = {1: 128,
                                   2: 64,
                                   3: 32,
                                   4: 16,
                                   5: 2}
        self.layer_in_channels = {1: in_channels,
                                  2: self.layer_out_channels[1],
                                  3: self.layer_out_channels[2],
                                  4: self.layer_out_channels[3],
                                  5: self.layer_out_channels[4]}

        self.layers = self.build_layers()

        # print('%s Model Successfully Built \n' % self.kpp_name)

    def build_layers(self):
        layers = []
        for x in range(1, len(self.layer_in_channels) + 1):
            conv = nn.Conv3d(in_channels=self.layer_in_channels[x], out_channels=self.layer_out_channels[x],
                             kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
            if x == len(self.layer_in_channels):
                layers += [conv, nn.Tanh()]
            else:
                layers += [conv, nn.ReLU(inplace=True)]

        return nn.Sequential(*layers)

    def forward(self, rep, kp):
        """
        Function to compute a single forward pass through the network, according to the architecture.
        :param x: (tensor) The input tensors from which to estimate the viewpoint change.
                   Must be a tensor of shape: (bsz, in_channels, 14, 14) for this application.
        :return: A scalar value representing the estimated viewpoint change between views 1 and 2.
        """
        x = torch.cat([rep, kp], dim=1)  # dim=channel
        x = self.layers(x)

        return x  # bsz


if __name__ == "__main__":
    print_summary = True

    ofp = OFPredictor(in_channels=128 + 32)
    print(ofp)

    if print_summary:
        summary(ofp, input_size=[(128, 4, 14, 14), (32, 4, 14, 14)])
