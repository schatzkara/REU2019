import torch
import torch.nn as nn


# bsz,nkp,frames,height(y),width(x)

# get xy coordinates
def get_coord(tensor, other_axis, axis_size):
    coord_probs = torch.mean(tensor, dim=other_axis, keepdim=False)  # bsz,nkp,frames,h/w
    coord_probs = nn.Softmax(dim=3)(coord_probs)  # softmax along h/w

    step = 2 / (axis_size - 1)
    coords = torch.range(start=-1.0, end=1.0, step=step, dtype=torch.float32)  # h/w
    coords = torch.reshape(coords, shape=(1, 1, 1, axis_size))  # bsz,nkp,frames,h/w
    coord = torch.sum(coord_probs * coords, dim=3)  # bsz, nkp, frames

    return coord


# make gaussian heatmaps
def get_gaussian_heatmaps(mean, stdev, map_size):
    mean_y, mean_x = mean[:, :, :, 0], mean[:, :, :, 1]
    

    return 0


# main
x = None
bsz, nkp, frames, height, width = x.size()
y_coord = get_coord(x, other_axis=4, axis_size=height)  # bsz, nkp, frames
x_coord = get_coord(x, other_axis=3, axis_size=width)  # bsz, nkp, frames
y_coord, x_coord = torch.unsqueeze(y_coord, dim=3), torch.unsqueeze(x_coord, dim=3)
gaussian_mean = torch.cat([y_coord, x_coord], dim=3)

heatmaps = get_gaussian_heatmaps(mean=gaussian_mean, stdev=0.01, map_size=(14, 14))
