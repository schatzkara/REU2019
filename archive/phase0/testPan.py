import time
import torch
import torch.nn as nn
from network import FullNetwork
from PanopticDataLoader import PanopticDataset
import torch.backends.cudnn as cudnn
import os
import cv2
import numpy as np

# directory information
data_root_dir = '/home/c2-2/yogesh/datasets/panoptic/rgb_data/'
test_splits = '/home/yogesh/kara/data/panoptic/mod_test.list'
weights_path = '/home/yogesh/kara/REU2019/phase0/weights/net_panoptic_20_16_2_False_1000_0.0001.pt'
output_video_dir = 'home/yogesh/kara/REU2019/phase0/videos/pan_66epochs/'
# 'C:/Users/Owner/Documents/UCF/Project/REU2019/videos'


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
PRECROP = False

# training parameters
NUM_EPOCHS = 1


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
            convert_to_vid(output_v1, 1, batch_idx, False), convert_to_vid(output_v2, 2, batch_idx, False)

        running_total_loss += loss.item()
        running_con_loss += con_loss.item()
        running_recon1_loss += recon1_loss.item()
        running_recon2_loss += recon2_loss.item()
        print('\tBatch {}/{} Loss:{} con:{} recon1:{} recon2:{}'.format(batch_idx + 1, len(testloader),
                                                                        "{0:.5f}".format(loss),
                                                                        "{0:.5f}".format(con_loss),
                                                                        "{0:.5f}".format(recon1_loss),
                                                                        "{0:.5f}".format(recon2_loss)))

    print('Testing Epoch {}/{} Loss:{} con:{} recon1:{} recon2:{}'.format(epoch + 1, NUM_EPOCHS,
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
    """
    Function to convert a tensor to a series of .jpg video frames
    :param tensor: (tensor) The tensor to be converted.
    :param view: (int) The view that the video is from: 1 or 2.
    :param batch_num: (int) The batch that the tensor is from.
    :param input: (bool) True if the tensor was a network input; False if it was a network output.
    :return: None
    """
    bsz, channels, frames, height, width = tensor.size()
    # loop through each video in the batch
    for i in range(bsz):
        vid = tensor[i]
        # if input:
        #     vid_path = os.path.join(output_video_dir, 'input')
        #     if not os.path.exists(vid_path):
        #         os.mkdir(vid_path)
        # else:
        #     vid_path = os.path.join(output_video_dir, 'output')
        #     if not os.path.exists(vid_path):
        #         os.mkdir(vid_path)
        # assert os.path.exists(vid_path), 'Vid path DNE.'
        save_frames(output_video_dir, vid, batch_num, i, view, input)


def save_frames(vid_path, vid, batch_num, vid_num, view, input):
    """
    Function to save the frames of a video to .jpgs.
    :param vid_path: (str) The path at which to save the video frames.
    :param vid: (tensor) The video to be saved.
    :param batch_num: (int) The batch that the video is from.
    :param vid_num: (int) The position of the video in the batch.
    :param view: (int) The view that the video is from: 1 or 2.
    :return: None
    """
    channels, frames, height, width = vid.size()
    vid_path = make_vid_path(vid_path, batch_num + 1, vid_num + 1, view, input)
    for i in range(frames):
        frame_name = make_frame_name(i + 1)
        frame_path = os.path.join(vid_path, frame_name)
        # extract one frame as np array
        frame = vid[:, i, :, :].squeeze().cpu().numpy()
        frame = denormalize_frame(frame)
        frame = from_tensor(frame)

        try:
            cv2.imwrite(frame_path, frame)
        except:
            print('The image did not successfully save.')

        assert os.path.exists(frame_path), 'The image does not exist.'


def make_vid_path(vid_path, batch_num, vid_num, view, input):
    """
    Function to make the path to save the video. Makes sure that the necessary paths exist.
    :param vid_path: (str) The path that holds all the video frames.
    :param batch_num: (int) The batch that the video is from.
    :param vid_num: (int) The position of the video in the batch.
    :param view: (int) The view that the video is from: 1 or 2.
    :return:
    """
    batch_num, vid_num, view = str(batch_num), str(vid_num), str(view)
    batch_path = os.path.join(vid_path, batch_num)
    vid_num_path = os.path.join(vid_path, batch_num, vid_num)
    view_path = os.path.join(vid_path, batch_num, vid_num, view)
    if input:
        full_path = os.path.join(vid_path, batch_path, vid_num, view, 'input')
    else:
        full_path = os.path.join(vid_path, batch_path, vid_num, view, 'output')
    if not os.path.exists(vid_path):
        os.mkdir(vid_path)
    if not os.path.exists(batch_path):
        os.mkdir(batch_path)
    if not os.path.exists(vid_num_path):
        os.mkdir(vid_num_path)
    if not os.path.exists(view_path):
        os.mkdir(view_path)
    if not os.path.exists(full_path):
        os.mkdir(full_path)
    return full_path


def make_frame_name(frame_num):
    """
    Function to correctly generate the correctly formatted .jpg file name for the frame.
    :param frame_num: The frame number captured in the file.
    :return: str representing the file name.
    """
    return str(frame_num).zfill(3) + '.jpg'


def denormalize_frame(frame):
    """
    Function to denormalize the pixel values in the frame to be between 0 and 255.
    :param frame: (array-like) The frame to be denormalized.
    :return: (np array) The denormalized frame.
    """
    frame = np.array(frame).astype(np.float32)
    return np.multiply(frame, 255.0)


def from_tensor(sample):
    """
    Function to convert the sample clip from a tensor to a numpy array.
    :param sample: (tensor) The sample to convert.
    :return: a numpy array representing the sample clip.
    """
    # pytorch tensor is (channels, height, width)
    # np is (height, width, channels)
    sample = np.transpose(sample, (1, 2, 0))

    return sample


def test_model():
    """
    Function to carry out the model's testing over all epochs.
    :return: None
    """
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
    print('panoptic')
    print('Parameters for testing:')
    print('Batch size: {}'.format(BATCH_SIZE))
    print('Tensor size: ({},{},{},{})'.format(CHANNELS, FRAMES, HEIGHT, WIDTH))
    print('Skip Length: {}'.format(SKIP_LEN))
    print('Precrop: {}'.format(PRECROP))
    print('Total Epochs: {}'.format(NUM_EPOCHS))


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
    # print('Model Built.')
    model = model.to(device)

    print(model)

    if device == 'cuda':
        net = torch.nn.DataParallel(model)
        cudnn.benchmark = True

    criterion = nn.MSELoss()

    if not os.path.exists(output_video_dir):
        os.mkdir(output_video_dir)

    # data
    testset = PanopticDataset(root_dir=data_root_dir, data_file=test_splits,
                              resize_height=HEIGHT, resize_width=WIDTH,
                              clip_len=FRAMES, skip_len=SKIP_LEN,
                              random_all=True, precrop=PRECROP)
    testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    test_model()
