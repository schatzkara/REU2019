# makes a .txt file with each camera and its 5 closest cameras on different panels

import os
import numpy as np
import _pickle as pickle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

data_root_dir = '/home/c2-2/yogesh/datasets/panoptic/rgb_data/'
# data_root_dir = 'C:/Users/Owner/Documents/UCF/panoptic/rgb_data/'
sample = '150303_celloScene1'  # '150821_dance4'
cal_file_path = os.path.join(data_root_dir, sample, 'calibration_' + sample + '.pkl')

output_file = 'closecams.list'


panels = range(1, 21)  # VGA panels range from 1 to 20
nodes = range(1, 25)  # VGA nodes range from 1 to 24

width = height = 128
resize_width = resize_height = 112


def get_vga_dict(panels, nodes):
    vga_dict = {}
    for panel in panels:
        vga_dict[panel] = []
        for node in nodes:
            vga_dict[panel].append(format_node(panel, node))

    return vga_dict  # has key: panel and value: list of nodes


def get_vga_list(panels, nodes):
    vga_list = []
    for panel in panels:
        for node in nodes:
            vga_list.append(format_node(panel, node))

    return vga_list


def get_existing_vga_list(root_dir):
    vga_list = []
    samples = os.listdir(root_dir)
    samples = [os.path.join(s, 'samples') for s in samples]
    for sample in samples:
        cams = os.listdir(os.path.join(root_dir, sample))
        for cam in cams:
            if cam not in vga_list:
                vga_list.append(cam)

    return vga_list


def format_node(panel, node):
    return 'vga_' + str(panel).zfill(2) + '_' + str(node).zfill(2)


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
        print(seq_id, cal_file)
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


def euclidean_distance(x, y):
    x, y = np.array(x), np.array(y)
    diff = x - y
    square = np.multiply(diff, diff)
    sum = np.sum(square)
    dist = np.sqrt(sum)
    return dist


def get_samples(root_dir):
    return os.listdir(root_dir)


def get_camera_positions(cameras, sample):
    positions = {}
    for cam in cameras:
        x, y, z, pan, van = get_view(seq_id=sample, view_id=cam, x_pos=0, y_pos=0)
        positions[cam] = (x, y, z)
    return positions


def get_all_distances(vga_list, cam_positions, allow_same_panel):
    dist_dict = {}  # key: cam, value: list of tuples (cam, distance)
    for vga1 in vga_list:
        type1, panel1, node1 = vga1.split('_')
        dist_dict[vga1] = []
        pos1 = cam_positions[vga1]
        for vga2 in vga_list:
            if vga1 == vga2:
                continue
            type2, panel2, node2 = vga2.split('_')
            if not allow_same_panel and panel1 == panel2:
                continue
            pos2 = cam_positions[vga2]
            dist = euclidean_distance(pos1, pos2)
            dist_dict[vga1].append((vga2, dist))

    return dist_dict


def sort_all_distances(dist_dict):
    for cam, dists in dist_dict.items():
        dists.sort(key=lambda x: x[1])
        dist_dict[cam] = dists

    return dist_dict


def get_k_closest_cams(dist_dict, k):
    sorted_dist_dict = sort_all_distances(dist_dict)
    k_closest_cams = {}
    for cam, dists in sorted_dist_dict.items():
        k_closest_cams[cam] = []
        close_cams = []
        used_panels = []
        i = 0
        while len(used_panels) < k:
            c, dist = sorted_dist_dict[cam][i]
            type, panel, node = c.split('_')
            if dist == 0.0 or panel in used_panels:
                i += 1
                continue
            close_cams.append(c)
            used_panels.append(panel)
            i += 1
        k_closest_cams[cam] = close_cams

    return k_closest_cams


if __name__ == '__main__':
    # vga_list = get_vga_list(panels, nodes)
    vga_list = get_existing_vga_list(data_root_dir)
    cam_positions = get_camera_positions(vga_list, sample)
    # print(cam_positions)
    cam_distances = get_all_distances(vga_list, cam_positions, allow_same_panel=False)
    closest_5_cams = get_k_closest_cams(cam_distances, k=5)

    with open(output_file, 'w') as f:
        for cam, close_cams in closest_5_cams.items():
            close_cams = ' '.join(close_cams)
            f.write(cam + ' ' + close_cams + '\n')
