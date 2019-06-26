# phase 2

import os
import time
import torch
import torch.nn as nn
from network import FullNetwork
from NTUDataLoader import NTUDataset
from PanopticDataLoader import PanopticDataset
from fullOutputConversion import convert_to_vid
import torch.backends.cudnn as cudnn

DATASET = 'Panoptic'  # 'NTU' or 'Panoptic'

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
    weights_path = '/home/yogesh/kara/REU2019/phase2/weights/net2_ntu2_20_16_2_True_1000_0.0001.pt'
    output_video_dir = './videos/ntu_104epochs_full'

    return data_root_dir, test_split, param_file, weights_path, output_video_dir


def panoptic_config():
    # Panoptic directory information
    data_root_dir = '/home/c2-2/yogesh/datasets/panoptic/rgb_data/'
    test_split = '/home/yogesh/kara/data/panoptic/mod_test.list'
    if not os.path.exists('./weights'):
        os.mkdir('./weights')
    weights_path = '/home/yogesh/kara/REU2019/phase2/weights/net2_pan2_closeviews_20_16_2_False_1000_0.0001.pt'
    output_video_dir = './videos/pan_closeviews_27epochs/'

    return data_root_dir, test_split, weights_path, output_video_dir


def test():
    """
    Function to carry out the testing/validation loop for the Full Network for a single epoch.
    :return: None
    """
    running_total_loss = 0.0
    running_con1_loss = 0.0
    running_con2_loss = 0.0
    running_recon1_loss = 0.0
    running_recon2_loss = 0.0

    model.eval()

    for batch_idx, (vp_diff, vid1, vid2) in enumerate(testloader):
        vp_diff = vp_diff.to(device)
        vid1, vid2 = vid1.to(device), vid2.to(device)
        img1, img2 = get_first_frame(vid1), get_first_frame(vid2)
        img1, img2 = img1.to(device), img2.to(device)

        with torch.no_grad():
            output_vid1, output_vid2, kp_v1, kp_v2, kp_v1_est, kp_v2_est = model(vp_diff=vp_diff,
                                                                                 vid1=vid1, vid2=vid2,
                                                                                 img1=img1, img2=img2)

            # save videos
            convert_to_vid(vid1, output_video_dir, batch_idx + 1, 1, 'input')
            convert_to_vid(vid2, output_video_dir, batch_idx + 1, 2, 'input')
            convert_to_vid(output_vid1, output_video_dir, batch_idx + 1, 1, 'output')
            convert_to_vid(output_vid2, output_video_dir, batch_idx + 1, 2, 'output')

            # loss
            con1_loss = criterion(kp_v1, kp_v1_est)
            con2_loss = criterion(kp_v2, kp_v2_est)
            recon1_loss = criterion(output_vid1, vid1)
            recon2_loss = criterion(output_vid2, vid2)
            loss = con1_loss + con2_loss + recon1_loss + recon2_loss

        running_total_loss += loss.item()
        running_con1_loss += con1_loss.item()
        running_con2_loss += con2_loss.item()
        running_recon1_loss += recon1_loss.item()
        running_recon2_loss += recon2_loss.item()
        if (batch_idx + 1) % 10 == 0:
            print('\tBatch {}/{} Loss:{} con1:{} con2: {} recon1:{} recon2:{}'.format(batch_idx + 1, len(testloader),
                                                                                      "{0:.5f}".format(loss),
                                                                                      "{0:.5f}".format(con1_loss),
                                                                                      "{0:.5f}".format(con2_loss),
                                                                                      "{0:.5f}".format(recon1_loss),
                                                                                      "{0:.5f}".format(recon2_loss)))

    print('Testing Complete Loss:{} con1:{} con2:{} recon1:{} recon2:{}'.format(
        "{0:.5f}".format((running_total_loss / len(testloader))),
        "{0:.5f}".format((running_con1_loss / len(testloader))),
        "{0:.5f}".format((running_con2_loss / len(testloader))),
        "{0:.5f}".format((running_recon1_loss / len(testloader))),
        "{0:.5f}".format((running_recon2_loss / len(testloader)))))


def get_first_frame(vid_batch):
    """
    Function to extract the first frame from a batch of input videos.
    We extract the first frame from each of the videos input to the network so that the network can learn appearance
    conditioning from the desired views.
    :param vid_batch: (tensor) A batch of videos from which to extract only the first frame of each.
    :return: A tensor that holds all the first frames.
    """
    # get the first frame fom each vid in the batch and eliminate temporal dimension
    frames = [torch.squeeze(vid[:, :1, :, :]) for vid in vid_batch]
    # extract the batch size from the input vid_batch
    batch_size = vid_batch.size()[0]
    # create empty tensor containing batch_size images of the correct shape (matching the frames)
    imgs = torch.zeros(batch_size, *frames[0].size())
    # put all the first frames into the tensor
    for sample in range(batch_size):
        imgs[sample] = frames[sample]

    return imgs


def test_model():
    """
    Function to test the model.
    :return: None
    """
    start_time = time.time()
    test()
    end_time = time.time()
    print('Time: {}'.format(end_time - start_time))


def print_params():
    """
    Function to print out all the custom parameter information for the experiment.
    :return: None
    """
    print('Parameters for testing on {}:'.format(DATASET))
    print('With close views')
    print('Batch size: {}'.format(BATCH_SIZE))
    print('Tensor size: ({},{},{},{})'.format(CHANNELS, FRAMES, HEIGHT, WIDTH))
    print('Skip Length: {}'.format(SKIP_LEN))
    print('Precrop: {}'.format(PRECROP))
    print('Close Views: {}'.format(VIEW_DIST_MAX))


if __name__ == '__main__':
    """
    Main function to carry out the training loop. 
    This function creates the model and data loaders. Then, it trains the model.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    RANDOM_ALL = True
    PRECROP = True if DATASET.lower() == 'ntu' else False
    CLOSE_VIEWS = False if DATASET.lower() == 'ntu' else True
    VIEW_DIST_MAX = None if DATASET.lower() == 'ntu' else 2

    if DATASET.lower() == 'ntu':
        data_root_dir, test_split, param_file, weights_path, output_video_dir = ntu_config()

        # model
        model = FullNetwork(vp_value_count=1, output_shape=(BATCH_SIZE, CHANNELS, FRAMES, HEIGHT, WIDTH))
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
        data_root_dir, test_split, weights_path, output_video_dir = panoptic_config()

        # model
        model = FullNetwork(vp_value_count=3, output_shape=(BATCH_SIZE, CHANNELS, FRAMES, HEIGHT, WIDTH))
        model.load_state_dict(torch.load(weights_path))
        model = model.to(device)

        if device == 'cuda':
            net = torch.nn.DataParallel(model)
            cudnn.benchmark = True

        criterion = nn.MSELoss()

        testset = PanopticDataset(root_dir=data_root_dir, data_file=test_split,
                                  resize_height=HEIGHT, resize_width=WIDTH,
                                  clip_len=FRAMES, skip_len=SKIP_LEN,
                                  random_all=RANDOM_ALL, close_views=CLOSE_VIEWS, view_dist_max=VIEW_DIST_MAX,
                                  precrop=PRECROP)
        testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    else:
        print('This network has only been set up to run on the NTU and Panoptic datasets.')

    print_params()
    print(model)
    test_model()
