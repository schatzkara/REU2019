import os

data_root_dir = 'C:/Users/Owner/Documents/UCF/Project/panoptic'
sample_file = 'seq.list'


if __name__ == '__main__':
    full_path = os.path.join(data_root_dir, sample_file)
    with open(full_path, 'r') as f:
        for line in f:
            sample, num = line.split(' ')
            print(sample)
            path = os.path.join(data_root_dir, 'rgb_data', sample)
            # print(path)
            if not os.path.exists(path):
                os.mkdir(path)
