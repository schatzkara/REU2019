# phase 1

import time
import torch
import torch.nn as nn
from .network import FullNetwork
from .NTUDataLoader import NTUDataset
import torch.backends.cudnn as cudnn
import os
import cv2
import numpy as np

# directory information
data_root_dir = '/home/c2-2/yogesh/datasets/ntu-ard/frames-240x135/'
# 'C:/Users/Owner/Documents/UCF/Project/ntu-ard/frames-240x135/'
# train_splits = '/home/yogesh/kara/data/train.list'
test_splits = '/home/yogesh/kara/data/val.list'
# 'C:/Users/Owner/Documents/UCF/Project/SSCVAS/data/shortestval.list'
weights_path = '/home/yogesh/kara/REU2019/weights/net_32_8_2_True_1000_0.0001.pt'
# 'C:/Users/Owner/Documents/UCF/Project/REU2019/weights/test_net_32_8_2_True_1000_0001.pt'

# data parameters
PRINT_PARAMS = True
# VIEW1 = 1
# VIEW2 = 2
BATCH_SIZE = 32
CHANNELS = 3
FRAMES = 8
SKIP_LEN = 2
HEIGHT = 112
WIDTH = 112
PRECROP = True

# training parameters
NUM_EPOCHS = 1
# LR = 1e-4

# weight_file_name = './weights/net_{}_{}_{}_{}'.format(BATCH_SIZE, FRAMES, NUM_EPOCHS, LR)
output_video_dir = '/home/yogesh/kara/REU2019/videos/'  # 'C:/Users/Owner/Documents/UCF/Project/REU2019/videos'


def test(epoch):
    """
    Function to carry out the testing/validation loop for the Full Network for a single epoch.
    :param epoch: (int) The current epoch in which the model is testing/validating.
    :return: None
    """
    running_total_loss = 0.0
    running_con_loss = 0.0
    running_recon1_loss = 0.0
    running_recon2_loss = 0.0

    model.eval()

    for batch_idx, (view1vid, view2vid) in enumerate(testloader):
        view1vid, view2vid = view1vid.to(device), view2vid.to(device)
        view1img, view2img = get_first_frame(view1vid), get_first_frame(view2vid)
        view1img, view2img = view1img.to(device), view2img.to(device)

        with torch.no_grad():
            output_v1, output_v2, rep_v1, rep_v2 = model(vid1=view1vid, vid2=view2vid, img1=view1img, img2=view2img)
            con_loss = criterion(rep_v1, rep_v2)
            recon1_loss = criterion(output_v1, view1vid)
            recon2_loss = criterion(output_v2, view2vid)
            loss = con_loss + recon1_loss + recon2_loss

            # save videos
            convert_to_vid(view1vid, 1, batch_idx, True), convert_to_vid(view2vid, 2, batch_idx, True)
            convert_to_vid(output_v1, 1, batch_idx, False), convert_to_vid(output_v1, 2, batch_idx, False)

        running_total_loss += loss.item()
        running_con_loss += con_loss.item()
        running_recon1_loss += recon1_loss.item()
        running_recon2_loss += recon2_loss.item()
        print('\tBatch {}/{} Loss:{} con:{} recon1:{} recon2:{}'.format(batch_idx + 1, len(testloader),
                                                                        "{0:.5f}".format(loss),
                                                                        "{0:.5f}".format(con_loss),
                                                                        "{0:.5f}".format(recon1_loss),
                                                                        "{0:.5f}".format(recon2_loss)))

    print('Validation Epoch {}/{} Loss:{} con:{} recon1:{} recon2:{}'.format(epoch + 1, NUM_EPOCHS,
                                                                             "{0:.5f}".format((
                                                                                     running_total_loss / len(
                                                                                 testloader))),
                                                                             "{0:.5f}".format((running_con_loss / len(
                                                                                 testloader))),
                                                                             "{0:.5f}".format((
                                                                                     running_recon1_loss / len(
                                                                                 testloader))),
                                                                             "{0:.5f}".format((
                                                                                     running_recon2_loss / len(
                                                                                 testloader)))))


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


def convert_to_vid(tensor, view, batch_num, input):  # whether it was an input or output
    bsz, channels, frames, height, width = tensor.size()
    for i in range(bsz):
        vid = tensor[i]
        if input:
            vid_path = os.path.join(output_video_dir, 'input')
        else:
            vid_path = os.path.join(output_video_dir, 'output')
        assert os.path.exists(vid_path), 'Vid path DNE.'
        save_frames(vid_path, vid, batch_num, i, view)


def save_frames(vid_path, vid, batch_num, vid_num, view):
    channels, frames, height, width = vid.size()
    vid_path = os.path.join(vid_path, str(batch_num), str(vid_num), str(view))
    if not os.path.exists(vid_path):
        os.mkdir(vid_path)
    for i in range(frames):
        frame_name = make_frame_name(i + 1)
        frame_path = os.path.join(vid_path, frame_name)
        # extract one frame as np array
        frame = vid[:, i, :, :].squeeze().cpu()
        # if device == 'cuda':
        #     frame.cpu()
        frame = frame.numpy()
        frame = denormalize_frame(frame)
        # pytorch tensor is (channels, height, width)
        # np is (height, width, channels)
        frame = np.transpose(frame, (1, 2, 0))

        try:
            cv2.imwrite(frame_path, frame)
        except:
            print('The image did not successfully save.')

        assert os.path.exists(frame_path), 'The image does not exist.'


def make_frame_name(frame_num):
    """
    Function to correctly generate the correctly formatted .jpg file name for the frame.
    :param frame_num: The frame number captured in the file.
    :return: str representing the file name.
    """
    return str(frame_num).zfill(3) + '.jpg'


def denormalize_frame(frame):
    frame = np.array(frame).astype(np.float32)
    return np.multiply(frame, 255.0)


def test_model():
    start_time = time.time()
    for epoch in range(NUM_EPOCHS):
        test(epoch)
    end_time = time.time()
    print('Time: {}'.format(end_time - start_time))


def print_params():
    """
    Function to print out all the custom parameter information for the experiment.
    :return: None
    """
    print('Parameters:')
    print('Batch size: {}'.format(BATCH_SIZE))
    print('Tensor size: ({},{},{},{})'.format(CHANNELS, FRAMES, HEIGHT, WIDTH))
    print('Skip Length: {}'.format(SKIP_LEN))
    print('Precrop: {}'.format(PRECROP))
    print('Total Epochs: {}'.format(NUM_EPOCHS))
    # print('Learning Rate: {}'.format(LR))


if __name__ == '__main__':
    """
    Main function to carry out the training loop. 
    This function creates the model and data loaders. Then, it trains the model.
    """
    if PRINT_PARAMS:
        print_params()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # model
    model = FullNetwork(output_shape=(BATCH_SIZE, CHANNELS, FRAMES, HEIGHT, WIDTH))
    model.load_state_dict(torch.load(weights_path))
    # model = torch.load(weights_path)
    # print('Model Built.')
    model = model.to(device)

    print(model)

    if device == 'cuda':
        net = torch.nn.DataParallel(model)
        cudnn.benchmark = True

    criterion = nn.MSELoss()
    # optimizer = optim.Adam(model.parameters(), lr=LR)  # other parameters???

    # data
    testset = NTUDataset(root_dir=data_root_dir, data_file=test_splits,
                         resize_height=HEIGHT, resize_width=WIDTH,
                         clip_len=FRAMES, skip_len=SKIP_LEN,
                         random_all=True, precrop=PRECROP)
    testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    test_model()
