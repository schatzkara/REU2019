import numpy as np
import os
import _pickle as pickle

viewidx = None
view = None
random_all = True
root_dir = '/home/c2-2/yogesh/datasets/panoptic/rgb_data/'
train_split = '/home/yogesh/kara/data/panoptic/mod_train.list'
close_cams_file = '/home/yogesh/kara/data/panoptic/closecams.list'


def make_close_cams_dict(cam_file):
    close_cams_dict = {}
    with open(cam_file, 'r') as f:
        for line in f:
            cams = line.strip().split(' ')
            close_cams_dict[cams[0]] = cams[1:]

    return close_cams_dict


def is_legal_view(camera, sample_name, sample_path_head, sample_path_tail):
    # print(camera_list)
    # num_cams = len(camera_list)
    # print(num_cams)

    path_exists = False
    cal_info_avail = False
    viewpath = get_vid_path(path_head=sample_path_head, path_tail=sample_path_tail, camera=camera)
    # print('got view path')
    path_exists = os.path.exists(viewpath)
    # print(path_exists)
    # print(viewpath)

    cal_info_avail = check_cal_info(seq_id=sample_name, view_id=camera)

    return path_exists and cal_info_avail


def check_cal_info(seq_id, view_id):
    cal_file = os.path.join(root_dir, seq_id, 'calibration_' + seq_id + '.pkl')
    if not os.path.exists(cal_file):
        print('{} DNE'.format(cal_file))

    # load the calibration file
    with open(cal_file, 'rb') as fp:
        cal = pickle.load(fp)

    try:
        c_data = cal[view_id[4:]]
    except:
        print(seq_id, view_id, cal_file)
        return False
    return True


def get_vid_path(path_head, path_tail, camera):
    """
    Function to get the paths at which the two sample views are located.
    :param path_head: (str) The first part of the vid path that contains the sample name and dir
    :param path_tail: (str) The last part of the vid path that contains the frame indices
    :return: 2 strings representing the paths for the sample views.
    """
    view_path = os.path.join(root_dir, path_head, str(camera), path_tail)
    return view_path


def process_index(index):
    """
    Function to process the information that the data file contains about the sample.
    The line of information contains the sample name as well as the frames to sample from.
    :param index: (int) The index of the sample.
    :return: the sample name, sample path, and frame indices
    """
    # ex. 170915_toddler4/samples 0_125
    sample_path, frame_nums = data_list[index].split(' ')
    sample_name = sample_path[:sample_path.index('/')]

    return sample_name, sample_path, frame_nums


if __name__ == '__main__':
    with open(train_split, 'r') as f:
        data_file = f.readlines()
    data_list = [line.strip() for line in data_file]
    close_cams_dict = make_close_cams_dict(close_cams_file)
    # sample_list = os.listdir(root_dir)
    # sample_paths = [os.path.join(root_dir, sample, 'samples') for sample in sample_list]
    for i in range(len(data_list)):
        print(i)
        sample_name, sample_path_head, sample_path_tail = process_index(i)
        legal_count = 0
        for cam, close_cams in close_cams_dict.items():
            legal = is_legal_view(cam, sample_name, sample_path_head, sample_path_tail)
            if legal:
                for cc in close_cams:
                    legal = is_legal_view(cam, sample_name, sample_path_head, sample_path_tail)
                    if legal:
                        legal_count += 1
        if legal_count < 1:
            print(sample_name, sample_path_tail)

