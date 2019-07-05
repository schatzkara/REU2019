import os

root_dir = '/home/c2-2/yogesh/datasets/panoptic/rgb_data/'
data_file = '/home/c2-2/yogesh/datasets/panoptic/train.list'


def get_actual_paths():
    actual_paths = []
    for item in data_list:
        # print(item)
        # ex. 170915_toddler4/samples 0_125
        sample, frames = item.split(' ')
        # print(sample)
        # print(frames)
        # sample_name = sample_path[:sample_path.index('/')]
        # start_frame, end_frame = frame_nums.split('_')
        sample_path = os.path.join(root_dir, sample)
        if os.path.exists(sample_path):
            cameras = os.listdir(sample_path)
            # print(cameras)
            full_paths = [os.path.join(root_dir, sample, cam, frames) for cam in cameras]
            # print(full_paths)
            exists = [os.path.exists(path) for path in full_paths]
            # print(exists)
            true_count = 0
            for _bool in exists:
                if _bool:
                    true_count += 1
            # print(true_count)
            if true_count >= 2:
                actual_paths.append(item)

    return actual_paths


def get_cam_dict():
    cam_dict = {}
    for item in data_list:
        good_cams = []
        # ex. 170915_toddler4/samples 0_125
        sample, frames = item.split(' ')
        # sample_name = sample_path[:sample_path.index('/')]
        # start_frame, end_frame = frame_nums.split('_')
        sample_path = os.path.join(root_dir, sample)
        if os.path.exists(sample_path):
            cameras = os.listdir(sample_path)
            full_paths = {cam: os.path.join(root_dir, sample, cam, frames) for cam in cameras}
            for cam, path in full_paths.items():
                if os.path.exists(path):
                    size = len(os.listdir(path=path))
                    if size == 125:
                        good_cams.append(cam)
        cam_dict[item] = good_cams

    return cam_dict


if __name__ == '__main__':
    with open(data_file) as f:
        data_file = f.readlines()
    data_list = [line.strip() for line in data_file]
    # print(data_list)

    data_list = get_actual_paths()
    print(data_list)

    cam_dict = get_cam_dict()
    print(cam_dict)
