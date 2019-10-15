import torch


def make_coordinate_grid(spatial_size, type):
    """
    Create a meshgrid [-1,1] x [-1,1] of given spatial_size.
    """
    h, w = spatial_size
    x = torch.arange(w).type(type)
    y = torch.arange(h).type(type)

    x = (2 * (x / (w - 1)) - 1)
    y = (2 * (y / (h - 1)) - 1)

    yy = y.view(-1, 1).repeat(1, w)
    xx = x.view(1, -1).repeat(h, 1)

    meshed = torch.cat([xx.unsqueeze_(2), yy.unsqueeze_(2)], 2)

    return meshed


x = make_coordinate_grid((5, 5), type=torch.float32)
print(x)

y = make_coordinate_grid((5, 5), type=torch.float32)
print(y)

z = x + y
print(z)
print(z.size())


a = {1: 1, 2: 2, 3: 3}
print(len(a))






