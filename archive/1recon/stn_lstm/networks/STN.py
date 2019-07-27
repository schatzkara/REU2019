# modifided from https://pytorch.org/tutorials/intermediate/spatial_transformer_tutorial.html

# from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
# import torch.optim as optim
# import torchvision
# from torchvision import datasets, transforms
# import matplotlib.pyplot as plt
# import numpy as np


class STN(nn.Module):
    def __init__(self, in_channels, in_shape):
        super(STN, self).__init__()
        # self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        # self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        # self.conv2_drop = nn.Dropout2d()
        # self.fc1 = nn.Linear(320, 50)
        # self.fc2 = nn.Linear(50, 10)
        self.out_shape = (int(self.compute_out_size(int(self.compute_out_size(in_shape[0], 7, 0, 1) / 2), 5, 0, 1) / 2),
                          int(self.compute_out_size(int(self.compute_out_size(in_shape[1], 7, 0, 1) / 2), 5, 0, 1) / 2))

        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * self.out_shape[0] * self.out_shape[1], 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    # Spatial transformer network forward function
    def stn(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 10 * self.out_shape[0] * self.out_shape[1])
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)

        return x

    def forward(self, x):
        # transform the input
        x = self.stn(x)

        '''# Perform the usual forward pass
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)'''

        return x

    def compute_out_size(self, in_size, filter_size, padding, stride):
        out_size = ((in_size - filter_size + 2 * padding) / stride) + 1
        return out_size
