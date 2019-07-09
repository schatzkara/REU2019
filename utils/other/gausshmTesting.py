import torch
import torch.nn as nn
import cv2
import numpy as np


# bsz,nkp,frames,height(y),width(x)

# get xy coordinates
def get_coord(tensor, other_axis, axis_size):
    coord_probs = torch.mean(tensor, dim=other_axis, keepdim=False)  # bsz, nkp, frames, h/w
    # print(coord_probs)
    coord_probs = nn.Softmax(dim=3)(coord_probs)  # softmax along h/w
    # print(coord_probs)

    # step = 2 / (axis_size - 1)
    coords = torch.linspace(start=-1.0, end=1.0, steps=axis_size, dtype=torch.float32)  # h/w
    # print(coords)
    coords = torch.reshape(coords, shape=(1, 1, 1, axis_size))  # bsz=1, nkp=1, frames=1, h/w
    coord = torch.sum(coord_probs * coords, dim=3)  # bsz, nkp, frames
    # print(coord)

    return coord


# make gaussian heatmaps
def get_gaussian_heatmaps(mean, stdev, map_size):
    height, width = map_size
    mean_y, mean_x = mean[:, :, :, 0:1], mean[:, :, :, 1:2]  # bsz, nkp, frames, h/w=1
    # print(mean_y.size(), mean_x.size())
    # print(mean_y, mean_x)
    y = torch.linspace(start=-1.0, end=1.0, steps=height, dtype=torch.float32)  # h
    x = torch.linspace(start=-1.0, end=1.0, steps=width, dtype=torch.float32)  # w
    # print(y, '\n', x)
    # print(y.size(), x.size())
    y = torch.reshape(y, shape=(1, 1, 1, height))  # bsz=1, nkp=1, frames=1, h
    x = torch.reshape(x, shape=(1, 1, 1, width))  # bsz=1, nkp=1, frames=1, w
    # print(y)
    # print(y.size(), x.size())

    gauss_y = torch.exp(-(((y - mean_y) * (y - mean_y)) / (2 * stdev * stdev)))  # bsz, nkp, frames, h
    gauss_x = torch.exp(-(((x - mean_x) * (x - mean_x)) / (2 * stdev * stdev)))  # bsz, nkp, frames, w
    # print(gauss_y, '\n', gauss_x)
    # print(gauss_y.size(), gauss_x.size())

    gauss_y = torch.unsqueeze(gauss_y, dim=4)  # bsz, nkp, frames, h, w=1
    gauss_x = torch.unsqueeze(gauss_x, dim=3)  # bsz, nkp, frames, h=1, w
    # print(gauss_y.size(), gauss_x.size())

    gauss_yx = torch.matmul(gauss_y, gauss_x)  # bsz, nkp, frames, h, w
    # print(gauss_yx.size())

    return gauss_yx


# main
'''x = None
bsz, nkp, frames, height, width = x.size()
y_coord = get_coord(x, other_axis=4, axis_size=height)  # bsz, nkp, frames
x_coord = get_coord(x, other_axis=3, axis_size=width)  # bsz, nkp, frames
y_coord, x_coord = torch.unsqueeze(y_coord, dim=3), torch.unsqueeze(x_coord, dim=3)
gaussian_mean = torch.cat([y_coord, x_coord], dim=3)

heatmaps = get_gaussian_heatmaps(mean=gaussian_mean, stdev=0.01, map_size=(14, 14))'''

if __name__ == '__main__':
    bsz, nkp, frames = 20, 32, 16
    height = width = 28
    mheight = mwidth = 28

    x = torch.randint(0, 256, (bsz, nkp, frames, height, width), dtype=torch.float32)
    # print(x)
    y_coord = get_coord(x, other_axis=4, axis_size=height)  # bsz, nkp, frames
    x_coord = get_coord(x, other_axis=3, axis_size=width)  # bsz, nkp, frames
    y_coord, x_coord = torch.unsqueeze(y_coord, dim=3), torch.unsqueeze(x_coord, dim=3)
    # y_coord = torch.randint(0, height, (bsz, nkp, frames, 1), dtype=torch.float32)
    # x_coord = torch.randint(0, width, (bsz, nkp, frames, 1), dtype=torch.float32)
    # print(y_coord, x_coord)
    gaussian_mean = torch.cat([y_coord, x_coord], dim=3)
    ghm = get_gaussian_heatmaps(mean=gaussian_mean, stdev=0.05, map_size=(mheight, mwidth))
    # print(ghm.size())
    # print(ghm)

    maxs, indices = torch.max(ghm, dim=3)
    # print(maxs, '\n', indices)

    maxs, indices = torch.max(ghm, dim=4)
    # print(maxs, '\n', indices)

    for b in range(bsz):
        for k in range(nkp):
            for f in range(frames):
                hmap = ghm[b, k, f, :, :].numpy()
                hmap = np.multiply(hmap, 255.0)
                cv2.namedWindow('image', cv2.WINDOW_NORMAL)
                cv2.resizeWindow('image', 400, 400)
                cv2.imshow('image', hmap)
                cv2.waitKey(300)
