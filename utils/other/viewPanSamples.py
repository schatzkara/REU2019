# phase 2

import os
import torch
from PanopticDataLoader import PanopticDataset
from modelIOFuncs import from_tensor
import numpy as np
import cv2

DATASET = 'panoptic'  # 'NTU' or 'panoptic'

# data parameters
BATCH_SIZE = 20
CHANNELS = 3
FRAMES = 16
SKIP_LEN = 2
HEIGHT = 112
WIDTH = 112

# training parameters
NUM_EPOCHS = 1000
LR = 1e-4


def panoptic_config():
    # panoptic directory information
    data_root_dir = '/home/c2-2/yogesh/datasets/panoptic/rgb_data/'
    train_split = '/home/yogesh/kara/data/panoptic/mod_train.list'
    test_split = '/home/yogesh/kara/data/panoptic/mod_test.list'
    close_cams_file = 'samepanel.list'
    if not os.path.exists('./weights'):
        os.mkdir('./weights')
    weight_file = './weights/net_pan_{}_{}_{}_{}_{}_{}.pt'.format(BATCH_SIZE, FRAMES, SKIP_LEN,
                                                                       PRECROP, NUM_EPOCHS, LR)
    return data_root_dir, train_split, test_split, close_cams_file, weight_file


if __name__ == '__main__':
    """
    Main function to carry out the training loop. 
    This function creates the generator and data loaders. Then, it trains the generator.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    RANDOM_ALL = True
    PRECROP = True if DATASET.lower() == 'ntu' else False
    VP_VALUE_COUNT = 1 if DATASET.lower() == 'ntu' else 3
    CLOSE_VIEWS = True if DATASET.lower() == 'panoptic' else False

    if DATASET.lower() == 'panoptic':
        data_root_dir, train_split, test_split, close_cams_file, weight_file = panoptic_config()

        # data
        trainset = PanopticDataset(root_dir=data_root_dir, data_file=train_split,
                                   resize_height=HEIGHT, resize_width=WIDTH,
                                   clip_len=FRAMES, skip_len=SKIP_LEN,
                                   random_all=RANDOM_ALL, close_views=CLOSE_VIEWS,
                                   close_cams_file=close_cams_file, precrop=PRECROP)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

        testset = PanopticDataset(root_dir=data_root_dir, data_file=test_split,
                                  resize_height=HEIGHT, resize_width=WIDTH,
                                  clip_len=FRAMES, skip_len=SKIP_LEN,
                                  random_all=RANDOM_ALL, close_views=CLOSE_VIEWS,
                                  close_cams_file=close_cams_file, precrop=PRECROP)
        testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    for batch_idx, (vp2, vp1, vp_diff, dist, vid1, vid2) in enumerate(trainloader):
        print('{}\n{}\n{}\n{}\n{}'.format(batch_idx, vp2, vp1, vp_diff, dist))
        # print(vid1.size())

        for frame in range(FRAMES):
            frame1, frame2 = vid1[0, :, frame, :, :].squeeze().numpy(), vid2[0, :, frame, :, :].squeeze().numpy()
            # frame1, frame2 = denormalize_frame(frame1), denormalize_frame(frame2)
            frame1, frame2 = from_tensor(frame1), from_tensor(frame2)
            # print(frame1, frame2)
            # print(frame1.shape, frame2.shape)
            display = np.hstack((frame1, frame2))
            cv2.namedWindow('image', cv2.WINDOW_NORMAL)
            # cv2.resizeWindow('image', 600, 300)
            cv2.imshow('image', display)
            cv2.waitKey(150)

