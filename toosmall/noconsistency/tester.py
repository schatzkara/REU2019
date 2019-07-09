# phase 2

import os
import time
import torch
import torch.nn as nn
from networks.model import FullNetwork
from data.NTUDataLoader import NTUDataset
from data.PanopticDataLoader import PanopticDataset
from data.outputConversion import convert_to_vid
import torch.backends.cudnn as cudnn
from utils.trainingFuncs import test_model

DATASET = 'NTU'  # 'NTU' or 'panoptic'

# data parameters
BATCH_SIZE = 20
CHANNELS = 3
FRAMES = 16
SKIP_LEN = 2
HEIGHT = 112
WIDTH = 112


def ntu_config():
    # NTU directory information
    data_root_dir = '/home/c2-2/yogesh/datasets/ntu-ard/frames-240x135/'
    if FRAMES * SKIP_LEN >= 32:
        test_split = '/home/yogesh/kara/data/val16.list'
    else:
        test_split = '/home/yogesh/kara/data/val.list'
    param_file = '/home/yogesh/kara/data/view.params'
    weights_path = './weights/netnocon_ntu_{}_{}_{}_{}_{}_{}.pt'.format(BATCH_SIZE, FRAMES, SKIP_LEN,
                                                                        PRECROP, 1000, 0.0001)
    output_video_dir = './videos/ntu_net3_5epochs'

    return data_root_dir, test_split, param_file, weights_path, output_video_dir


def panoptic_config():
    # panoptic directory information
    data_root_dir = '/home/c2-2/yogesh/datasets/panoptic/rgb_data/'
    test_split = '/home/yogesh/kara/data/panoptic/mod_test.list'
    close_cams_file = '/home/yogesh/kara/data/panoptic/closecams.list'
    if not os.path.exists('./weights'):
        os.mkdir('./weights')
    weights_path = './weights/netnocon_pan_{}_{}_{}_{}_{}_{}.pt'.format(BATCH_SIZE, FRAMES, SKIP_LEN,
                                                                        PRECROP, 1000, 0.0001)
    output_video_dir = './videos/pan_net3_'

    return data_root_dir, test_split, close_cams_file, weights_path, output_video_dir


def print_params():
    """
    Function to print out all the custom parameter information for the experiment.
    :return: None
    """
    print('Parameters for testing on {}:'.format(DATASET))
    print('Batch size: {}'.format(BATCH_SIZE))
    print('Tensor size: ({},{},{},{})'.format(CHANNELS, FRAMES, HEIGHT, WIDTH))
    print('Skip Length: {}'.format(SKIP_LEN))
    print('Precrop: {}'.format(PRECROP))
    print('Close Views: {}'.format(CLOSE_VIEWS))


if __name__ == '__main__':
    """
    Main function to carry out the training loop. 
    This function creates the model and data loaders. Then, it trains the model.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    RANDOM_ALL = True
    PRECROP = True if DATASET.lower() == 'ntu' else False
    VP_VALUE_COUNT = 1 if DATASET.lower() == 'ntu' else 3
    CLOSE_VIEWS = True if DATASET.lower() == 'panoptic' else False

    if DATASET.lower() == 'ntu':
        data_root_dir, test_split, param_file, weights_path, output_video_dir = ntu_config()

        # model
        model = FullNetwork(vp_value_count=VP_VALUE_COUNT, output_shape=(BATCH_SIZE, CHANNELS, FRAMES, HEIGHT, WIDTH))
        model.load_state_dict(torch.load(weights_path))
        model = model.to(device)

        if device == 'cuda':
            net = torch.nn.DataParallel(model)
            cudnn.benchmark = True

        criterion = nn.MSELoss()

        # data
        testset = NTUDataset(root_dir=data_root_dir, data_file=test_split, param_file=param_file,
                             resize_height=HEIGHT, resize_width=WIDTH,
                             clip_len=FRAMES, skip_len=SKIP_LEN,
                             random_all=RANDOM_ALL, precrop=PRECROP)
        testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    elif DATASET.lower() == 'panoptic':
        data_root_dir, test_split, close_cams_file, weights_path, output_video_dir = panoptic_config()

        # model
        model = FullNetwork(vp_value_count=VP_VALUE_COUNT, output_shape=(BATCH_SIZE, CHANNELS, FRAMES, HEIGHT, WIDTH))
        model.load_state_dict(torch.load(weights_path))
        model = model.to(device)

        if device == 'cuda':
            net = torch.nn.DataParallel(model)
            cudnn.benchmark = True

        criterion = nn.MSELoss()

        testset = PanopticDataset(root_dir=data_root_dir, data_file=test_split,
                                  resize_height=HEIGHT, resize_width=WIDTH,
                                  clip_len=FRAMES, skip_len=SKIP_LEN,
                                  random_all=RANDOM_ALL, close_views=CLOSE_VIEWS,
                                  close_cams_file=close_cams_file, precrop=PRECROP)
        testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    else:
        print('This network has only been set up to run on the NTU and panoptic datasets.')

    print_params()
    print(model)
    test_model(model, criterion, device, testloader, loss_weights, output_video_dir)
