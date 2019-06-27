# phase 2

import os
import numpy as np
import _pickle as pickle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# data_root_dir = '/home/c2-2/yogesh/datasets/panoptic/rgb_data/'
data_root_dir = 'C:/Users/Owner/Documents/UCF/Project/panoptic/rgb_data/'
sample = '150303_celloScene1'  # '150821_dance5'
width = height = 128
resize_width = resize_height = 112


def get_camera_positions(sample):
    positions = {}
    cameras = get_cameras(sample=sample)
    for cam in cameras:
        x, y, z, pan, van = get_view(seq_id=sample, view_id=cam, x_pos=0, y_pos=0)
        positions[cam] = (x, y, z)
    return positions


def get_cameras(sample):
    sample_dir = os.path.join(data_root_dir, sample, 'samples')
    cameras = os.listdir(sample_dir)
    return cameras


def get_view(seq_id, view_id, x_pos, y_pos):
    """
    Function to get the view of the camera.
    :param seq_id: (str) The sample name.
    :param view_id: (int) The camera ID.
    :param x_pos: (int) The x coordinate of the pixel to crop at.
    :param y_pos: (int) The y coordinate of the pixel to crop at.
    :return: The x, y, and z coordinates of the camera and the pan and van values
    """
    # seq = os.path.split(seq_id)[0]
    cal_file = os.path.join(data_root_dir, seq_id, 'calibration_' + seq_id + '.pkl')
    # print cal_file

    # load the calibration file
    with open(cal_file, 'rb') as fp:
        cal = pickle.load(fp)

    # get the camera calibration values
    # print(seq_id)
    # print(cal)
    try:
        c_data = cal[view_id[4:]]
    except:
        print
        seq_id, cal_file
    # c_data = cal[view_id]
    R = c_data["R"]
    t = c_data["t"]
    # print seq_id

    # camera intrinsic k?
    dc = c_data["distCoef"]
    K = c_data["K"]
    c_x, c_y = K[0][2] / 480., K[1][2] / 640.
    f_x, f_y = K[0][0] / 480., K[1][1] / 640.

    # compute the viewpoint here
    x, y, z = -np.dot(np.linalg.inv(R), t) / 100.
    x, y, z = x[0], y[0], z[0]

    # new run with c, f, and dc additional

    pan = 1. * x_pos / (width - resize_width)
    van = 1. * y_pos / (height - resize_height)

    # return np.array([x, y, z, c_x, c_y, f_x, f_y, dc[0], dc[1], dc[2] * 100, dc[3] * 100, dc[4], pan, van])
    # print('x{}y{}z{}pan{}van{}'.format(x, y, z, pan, van))
    return np.array([x, y, z, pan, van])


def plot_cameras(cam_positions):
    labels = []
    xs = []
    ys = []
    zs = []
    for cam in cam_positions:
        x, y, z = cam_positions[cam]
        labels.append(cam)
        xs.append(x)
        ys.append(y)
        zs.append(z)

    # print(labels)
    # print(xs)
    # print(ys)
    # print(zs)

    pos1 = None
    pos2 = None

    fig = plt.figure(1, (16, 16))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(xs, ys, zs)
    for cam in cam_positions:
        x, y, z = cam_positions[cam]
        id = cam[4:]
        if id == '10_06':
            pos1 = [x, y, z]
        if id == '11_18':
            pos2 = [x, y, z]
        ax.text(x, y, z, id, size=10, zorder=1, color='k')

    print(euclidean_distance(np.array(pos1), np.array(pos2)))

    plt.show()


def euclidean_distance(x, y):
    diff = x - y
    square = np.multiply(diff, diff)
    sum = np.sum(square)
    dist = np.sqrt(sum)
    return dist


if __name__ == '__main__':
    sample = input('What sample? ')
    cam_positions = get_camera_positions(sample)
    print(cam_positions)
    plot_cameras(cam_positions=cam_positions)
