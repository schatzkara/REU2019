import os

data_root_dir = '/home/c2-2/yogesh/datasets/panoptic/rgb_data/'
# data_root_dir = 'C:/Users/Owner/Documents/UCF/Project/panoptic/rgb_data/'
train_splits = '/home/c2-2/yogesh/datasets/panoptic/train.list'
# train_splits = 'C:/Users/Owner/Documents/UCF/Project/REU2019/data/panoptic/train.list'
test_splits = '/home/c2-2/yogesh/datasets/panoptic/test.list'
# test_splits = 'C:/Users/Owner/Documents/UCF/Project/REU2019/data/panoptic/train.list'

if __name__ == '__main__':
    files_to_filter = [train_splits, test_splits]
    new_file = 'train2.list'
    for file in files_to_filter:
        print(file)
        with open(new_file, 'w') as g:
            with open(file, 'r') as f:
                for line in f:
                    sample, frames = line.strip().split(' ')
                    sample_path = os.path.join(data_root_dir, sample)
                    if os.path.exists(sample_path):
                        cameras = os.listdir(sample_path)
                        full_paths = [os.path.join(data_root_dir, sample, cam, frames) for cam in cameras]
                        exists = [os.path.exists(path) for path in full_paths]
                        true_count = 0
                        for _bool in exists:
                            if _bool:
                                true_count += 1
                        if true_count >= 2:
                            g.write(line)
                        else:
                            print(full_paths)
                            print(exists)
                            print(line)
                        # for cam in cameras:
                        #     full_path = os.path.join(data_root_dir, sample, cam, frames)
                        #     if os.path.exists(full_path):
                        #         g.write(line)
                        #     else:
                        #         print(line)
        new_file = 'test2.list'
