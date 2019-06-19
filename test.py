import torch
import torch.nn as nn

split = '/home/c2-2/yogesh/datasets/panoptic/rgb_data/160906_ian2/samples/vga_11_21/7500_7625'.split('/')

print(split)

x = torch.zeros(32, 4, 16, 28, 28)

x = nn.functional.interpolate(x, size=(16, 56, 56), mode='nearest')

print(x.size())
