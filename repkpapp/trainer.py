# phase 2

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from networks.model import FullNetwork
from data.NTUDataLoader import NTUDataset
from data.PanopticDataLoader import PanopticDataset
from utils.trainingFuncs import train_model
import torch.backends.cudnn as cudnn

DATASET = 'NTU'  # 'NTU' or 'Panoptic'

# data parameters
BATCH_SIZE = 16
CHANNELS = 3
FRAMES = 16
SKIP_LEN = 2
HEIGHT = 112
WIDTH = 112

# training parameters
NUM_EPOCHS = 1000
LR = 1e-4
STDEV = 0.05
CON_LOSS_W = 0.1


PRETRAINED = True
pretrained_weights_file = './weights/net_ntu_16_16_2_True_1000_0.0001_0.05.pt'
pretrained_epochs = 14


def ntu_config():
    # NTU directory information
    data_root_dir = '/home/c2-2/yogesh/datasets/ntu-ard/frames-240x135/'
    if FRAMES * SKIP_LEN >= 32:
        train_split = '/home/yogesh/kara/data/train16.list'
        test_split = '/home/yogesh/kara/data/val16.list'
    else:
        train_split = '/home/yogesh/kara/data/train.list'
        test_split = '/home/yogesh/kara/data/val.list'
    param_file = '/home/yogesh/kara/data/view.params'
    if not os.path.exists('./weights'):
        os.mkdir('./weights')
    weight_file = './weights/net_ntu_{}_{}_{}_{}_{}_{}_{}.pt'.format(BATCH_SIZE, FRAMES, SKIP_LEN,
                                                                     PRECROP, NUM_EPOCHS, LR, STDEV)
    return data_root_dir, train_split, test_split, param_file, weight_file


def panoptic_config():
    # panoptic directory information
    data_root_dir = '/home/c2-2/yogesh/datasets/panoptic/rgb_data/'
    train_split = '/home/yogesh/kara/data/panoptic/mod_train.list'
    test_split = '/home/yogesh/kara/data/panoptic/mod_test.list'
    close_cams_file = '/home/yogesh/kara/data/panoptic/closecams.list'
    if not os.path.exists('./weights'):
        os.mkdir('./weights')
    weight_file = './weights/net_pan_{}_{}_{}_{}_{}_{}_{}.pt'.format(BATCH_SIZE, FRAMES, SKIP_LEN,
                                                                     PRECROP, NUM_EPOCHS, LR, STDEV)
    return data_root_dir, train_split, test_split, close_cams_file, weight_file


def print_params():
    """
    Function to print out all the custom parameter information for the experiment.
    :return: None
    """
    print('Parameters for training on {}'.format(DATASET))
    print('Batch Size: {}'.format(BATCH_SIZE))
    print('Tensor Size: ({},{},{},{})'.format(CHANNELS, FRAMES, HEIGHT, WIDTH))
    print('Skip Length: {}'.format(SKIP_LEN))
    print('Precrop: {}'.format(PRECROP))
    print('Close Views: {}'.format(CLOSE_VIEWS))
    print('Total Epochs: {}'.format(NUM_EPOCHS))
    print('Learning Rate: {}'.format(LR))
    print('Standard Deviation: {}'.format(STDEV))
    print('Con Loss Weight: {}'.format(CON_LOSS_W))


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

    # model
    model = FullNetwork(vp_value_count=VP_VALUE_COUNT,
                        output_shape=(BATCH_SIZE, CHANNELS, FRAMES, HEIGHT, WIDTH),
                        stdev=STDEV)
    if PRETRAINED:
        model.load_state_dict(torch.load(pretrained_weights_file))
    model = model.to(device)

    if device == 'cuda':
        net = torch.nn.DataParallel(model)
        cudnn.benchmark = True

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    if DATASET.lower() == 'ntu':
        data_root_dir, train_split, test_split, param_file, weight_file = ntu_config()

        # data
        trainset = NTUDataset(root_dir=data_root_dir, data_file=train_split, param_file=param_file,
                              resize_height=HEIGHT, resize_width=WIDTH,
                              clip_len=FRAMES, skip_len=SKIP_LEN,
                              random_all=RANDOM_ALL, precrop=PRECROP)

        testset = NTUDataset(root_dir=data_root_dir, data_file=test_split, param_file=param_file,
                             resize_height=HEIGHT, resize_width=WIDTH,
                             clip_len=FRAMES, skip_len=SKIP_LEN,
                             random_all=RANDOM_ALL, precrop=PRECROP)

    elif DATASET.lower() == 'panoptic':
        data_root_dir, train_split, test_split, close_cams_file, weight_file = panoptic_config()

        # data
        trainset = PanopticDataset(root_dir=data_root_dir, data_file=train_split,
                                   resize_height=HEIGHT, resize_width=WIDTH,
                                   clip_len=FRAMES, skip_len=SKIP_LEN,
                                   random_all=RANDOM_ALL, close_views=CLOSE_VIEWS,
                                   close_cams_file=close_cams_file, precrop=PRECROP)

        testset = PanopticDataset(root_dir=data_root_dir, data_file=test_split,
                                  resize_height=HEIGHT, resize_width=WIDTH,
                                  clip_len=FRAMES, skip_len=SKIP_LEN,
                                  random_all=RANDOM_ALL, close_views=CLOSE_VIEWS,
                                  close_cams_file=close_cams_file, precrop=PRECROP)

    else:
        print('This network has only been set up to train on the NTU and Panoptic datasets.')

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    print_params()
    print(model)
    train_model(starting_epoch=pretrained_epochs+1, num_epochs=NUM_EPOCHS, model=model, optimizer=optimizer,
                criterion=criterion, trainloader=trainloader, testloader=testloader, device=device,
                weight_file=weight_file, loss_weights={'con1': CON_LOSS_W, 'con2': CON_LOSS_W,
                                                       'con3': CON_LOSS_W, 'con4': CON_LOSS_W})
