import os

data_root_dir = '/home/c2-2/yogesh/datasets/panoptic/rgb_data'

if __name__ == '__main__':
    with open('dir_sizes.txt', 'a') as file:
        samples = os.listdir(data_root_dir)
        for s in samples:
            sample_dir = os.path.join(data_root_dir, s, 'samples')
            cameras = os.listdir(sample_dir)
            for c in cameras:
                camera_dir = os.path.join(sample_dir, c)
                frame_sets = os.listdir(camera_dir)
                for f in frame_sets:
                    frame_dir = os.path.join(camera_dir, f)
                    frames = os.listdir(frame_dir)
                    if len(frames) != 125:
                        print(frame_dir)
                        print(len(frames))
                        file.write(frame_dir + '\n')
                        file.write(str(len(frames)) + '\n')
