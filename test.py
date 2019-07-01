import torch
import torch.nn as nn
import tensorflow as tf
import sys

# tf.compat.v1.enable_eager_execution()


# x = torch.zeros(2, 2)
# x = torch.tensor([[1., 2.],
#                   [1., 2.],
#                   [1., 2.]])
# print(x.size())
# y = torch.mean(x, dim=1)
# print(y)
# print(y.size())

# a = torch.range(start=-1.0, end=1.0, step=2/7)
# print(a)
# tf.compat.v1.enable_eager_execution()
# b = tf.linspace(start=-1.0, stop=1.0, num=8)
# tf.print(b, output_stream=sys.stdout)

# x = torch.tensor([2.0, 2.0])
# x = torch.reshape(x, shape=(1, 1, 1, 2))
# print(x)
# print(x.size())
#
# y = torch.randn(3, 4, 5, 2)
# print(y.size())
#
# z = x * y
# print(z.size())
#
# zz = torch.sum(z, dim=3)
# print(zz.size())


def my_get_coord(tensor, other_axis, axis_size):
    coord_probs = torch.mean(tensor, dim=other_axis, keepdim=False)  # bsz,nkp,frames,h/w
    coord_probs = nn.Softmax(dim=3)(coord_probs)  # softmax along h/w

    step = 2 / (axis_size - 1)
    coords = torch.range(start=-1.0, end=1.0, step=step, dtype=torch.float32)  # h/w
    coords = torch.reshape(coords, shape=(1, 1, 1, axis_size))  # bsz,nkp,frames,h/w
    coord = torch.sum(coord_probs * coords, dim=3)  # bsz, nkp, frames

    return coord


def get_coord(other_axis, axis_size):
    # get "x-y" coordinates:
    g_c_prob = tf.reduce_mean(x, axis=other_axis)  # B,W,NMAP
    g_c_prob = tf.nn.softmax(g_c_prob, axis=1)  # B,W,NMAP
    coord_pt = tf.to_float(tf.linspace(-1.0, 1.0, axis_size))  # W
    coord_pt = tf.reshape(coord_pt, [1, axis_size, 1])
    g_c = tf.reduce_sum(g_c_prob * coord_pt, axis=1)
    return g_c, g_c_prob


# print(torch.randint(1, 5, (1, 1, 1, 2, 2)))

# tensor = torch.tensor([[[[[6., 0.],
#                           [0.5, 8.],
#                           [6., 0.],
#                           [0.5, 8.]
#                           ]]]])
# torch.tensor([[[[[2.0, 2.0]]]],[[[[2.0, 2.0]]]]])
# print(tensor.size())
# ans1 = my_get_coord(tensor, 4, 4)
#
# x = tf.convert_to_tensor([[[[6., 0.],
#                             [0.5, 8.],
#                             [6., 0.],
#                             [0.5, 8.]
#                             ]]])
# x = tf.transpose(x, perm=[0, 2, 3, 1])
# [[[[2.0, 2.0]]],[[[2.0, 2.0]]]])
# print(x.get_shape())
# ans2, _ = get_coord(2, 4)
#
# print(ans1)
# tf.print(ans2, output_stream=sys.stdout)

x = torch.zeros(2, 2, 2)
y = torch.ones(2, 2, 2)
z = torch.cat([torch.unsqueeze(x, dim=3), torch.unsqueeze(y, dim=3)], dim=3)
print(z.size())
