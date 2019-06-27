import torch
import torch.nn as nn
import torch.nn.functional as f
from torchsummary import summary

if __name__ == "__main__":
    # x = torch.randint(1, 3, (2, 2, 2, 2, 2))
    # x = torch.tensor(x, dtype=torch.float32)
    # x = torch.tensor([[[[[1., 2.], [2., 2.]],
    #                     [[1., 2.], [2., 2.]]],
    #                    [[[1., 2.], [2., 2.]],
    #                     [[1., 2.], [2., 2.]]]],
    #                   [[[[1., 2.], [2., 2.]],
    #                     [[1., 2.], [2., 2.]]],
    #                    [[[1., 2.], [2., 2.]],
    #                     [[1., 2.], [2., 2.]]]]])
    # print(x)
    # x = f.interpolate(x, size=(4, 4, 4), mode='trilinear', align_corners=False)
    # print(x)

    x = torch.ones(2, 3, 4, 8, 8)
    # print(torch.sum(x))
    bsz, channels, frames, height, width = x.size()
    x = torch.reshape(x, (bsz, channels, frames, height * width))
    print(x.size())
    x = nn.Softmax(dim=3)(x)
    print(x.size())
    x = torch.reshape(x, (bsz, channels, frames, height, width))
    print(x.size())
    for i in range(height * width):
        print(x[:, :, :, i].size())
        print(torch.sum(torch.squeeze(x[i, i, i, :])))
