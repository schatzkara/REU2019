# phase 2

import os
import cv2

if __name__ == '__main__':

    data_root_dir = '/home/c2-2/yogesh/datasets/panoptic/rgb_data/'
    sample = '160401_ian1'
    sample_dir = os.path.join(data_root_dir, sample, 'samples')

    output_dir = './cropped_pan_input/'
    os.mkdir(output_dir)

    cameras = os.listdir(sample_dir)
    for cam in cameras:
        cam_dir = os.path.join(sample_dir, cam)
        frame_path = os.path.join(cam_dir, '0_125', '001.png')
        out_path = os.path.join(output_dir, cam, '001.png')

        img = cv2.imread(frame_path)
        crop_img = img[8:121, 8:121, :]
        cv2.imwrite(out_path, crop_img)

        assert os.path.exists(out_path), 'image DNE'
