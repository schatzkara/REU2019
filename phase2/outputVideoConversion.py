import os
import cv2
import numpy as np


def convert_to_vid(tensor, output_dir, batch_num, view, input):  # whether it was an input or output
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
        save_frames(vid, output_dir, batch_num, i + 1, view, input)


def save_frames(vid, output_dir, batch_num, vid_num, view, input):
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
    vid_path = make_vid_path(output_dir, batch_num, vid_num, view, input)
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


def make_vid_path(output_dir, batch_num, vid_num, view, input):
    """
    Function to make the path to save the video. Makes sure that the necessary paths exist.
    :param vid_path: (str) The path that holds all the video frames.
    :param batch_num: (int) The batch that the video is from.
    :param vid_num: (int) The position of the video in the batch.
    :param view: (int) The view that the video is from: 1 or 2.
    :return:
    """
    batch_num, vid_num, view = str(batch_num), str(vid_num), str(view)
    batch_path = os.path.join(output_dir, batch_num)
    vid_path = os.path.join(batch_path, vid_num)
    view_path = os.path.join(vid_path, view)
    if input:
        full_path = os.path.join(view_path, 'input')
    else:
        full_path = os.path.join(view_path, 'output')

    dir_list = [output_dir, batch_path, vid_path, view_path, full_path]
    for dir in dir_list:
        if not os.path.exists(dir):
            os.mkdir(dir)
    # if not os.path.exists(output_dir):
    #     os.mkdir(output_dir)
    # if not os.path.exists(batch_path):
    #     os.mkdir(batch_path)
    # if not os.path.exists(vid_path):
    #     os.mkdir(vid_path)
    # if not os.path.exists(view_path):
    #     os.mkdir(view_path)
    # if not os.path.exists(full_path):
    #     os.mkdir(full_path)
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