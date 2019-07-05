import os

data_root_dir = 'C:/Users/Owner/Documents/UCF/Project/panoptic/rgb_data'
sample = '150303_celloScene1'
view_file = 'view.list'


if __name__ == '__main__':
    full_path = os.path.join(data_root_dir, sample, view_file)
    with open(full_path, 'r') as f:
        for line in f:
            view, num = line.split(' ')
            print(view)
            path = os.path.join(data_root_dir, sample, 'samples', view)
            # print(path)
            if not os.path.exists(path):
                os.mkdir(path)
