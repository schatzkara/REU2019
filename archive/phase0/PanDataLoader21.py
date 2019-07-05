# phase 1

from torch.utils.data.dataset import Dataset
import os
import numpy as np
import cv2
import _pickle as pickle


class PanopticDataset(Dataset):
    """panoptic Dataset"""

    def __init__(self, root_dir, data_file, resize_height, resize_width, clip_len,
                 height=128, width=128, frame_count=125,
                 skip_len=1, num_views=None, random_views=False, random_all=False,
                 precrop=False):
        """
        Initializes the panoptic Dataset object used for extracting samples to run through the network.
        :param root_dir: (str) Directory with all the frames extracted from the videos.
        :param data_file: (str) Path to file containing the sample IDs.
        :param resize_height: (int) The desired frame height.
        :param resize_width: (int) The desired frame width.
        :param clip_len: (int) The number of frames desired in the sample clip.
        :param height: (int, optional) The height of the frames in the dataset (default 128 for panoptic Dataset).
        :param width: (int, optional) The width of the frames in the dataset (default 128 for panoptic Dataset).
        :param frame_count: (int) The number of total frames in each sample (default 125 for panoptic Dataset).
        :param skip_len: (int, optional) The number of frames to skip between each when creating the clip (default 1).
        :param num_views: (int, optional) The number of views available to choose from for all samples (default None).
        :param random_views: (boolean, optional) True to use 2 constant randomly generated views (default False).
        :param random_all: (boolean, optional) True to use 2 randomly generated views for each sample (default False).
        :param precrop: (boolean, optional) True to crop 50 pixels off left and right of frame before randomly cropping
                        (default False).
        """
        self.root_dir = root_dir
        with open(data_file) as f:
            self.data_file = f.readlines()
        self.data_list = [line.strip() for line in self.data_file]  # list of all sample IDs
        self.data_list = self.get_actual_ids()  # list of sample IDs with at least 2 viewpoints
        self.cam_dict = self.get_viewpoints()  # dict with key: sample ID and value: available viewpoints
        self.clip_len = clip_len
        self.skip_len = skip_len
        self.height = height
        self.width = width
        self.frame_count = frame_count
        self.resize_height = resize_height
        self.resize_width = resize_width
        self.channels = 3
        self.num_views = num_views
        self.view1 = None
        self.view2 = None
        if random_views:
            self.get_random_views(self.num_views)
        self.random_all = random_all
        self.precrop = precrop

    def get_actual_ids(self):
        """
        Function to get only the sample IDs for which the sample contains at least 2 available viewpoints.
        :return: (list) A list of the sample IDs satisfying the criterion.
        """
        actual_ids = []
        for item in self.data_list:
            sample, frames = item.split(' ')
            sample_path = os.path.join(self.root_dir, sample)
            if os.path.exists(sample_path):
                # list of all the sample viewpoints
                cameras = os.listdir(sample_path)
                full_paths = [os.path.join(self.root_dir, sample, cam, frames) for cam in cameras]
                # determines if the viewpoints have the frames necessary
                exists = [os.path.exists(path) for path in full_paths]
                true_count = 0
                for _bool in exists:
                    if _bool:
                        true_count += 1
                # if there are at least 2 viewpoints with the necessary frames, then the sample can be used
                if true_count >= 2:
                    actual_ids.append(item)

        return actual_ids

    def get_viewpoints(self):
        """
        Function to get the possible viewpoints/cameras for each sample.
        :return: (dict) A dictionary with keys: sample IDs and values: list of viewpoints
        """
        cam_dict = {}
        for item in self.data_list:
            good_cams = []
            sample, frames = item.split(' ')
            sample_path = os.path.join(self.root_dir, sample)
            if os.path.exists(sample_path):
                cameras = os.listdir(sample_path)
                full_paths = {cam: os.path.join(self.root_dir, sample, cam, frames) for cam in cameras}
                for cam, path in full_paths.items():
                    # makes sure that the camera has the correct frames and has frame_count of them
                    if os.path.exists(path):
                        size = len(os.listdir(path=path))
                        if size == self.frame_count:
                            good_cams.append(cam)
            cam_dict[item] = good_cams

        return cam_dict

    def __len__(self):
        """
        Function to return the number of samples in the dataset.
        :return: int representing the number of samples in the dataset.
        """
        return len(self.data_list)

    def __getitem__(self, idx):
        """
        Function to get a single sample from the dataset. All modifications to the sample are done here.
        :param idx: (int) The index of the sample to get.
        :return: 2 tensors representing the sample video from 2 different viewpoints
        """
        sample_name, sample_path_head, sample_path_tail, cameras = self.process_index(index=idx)
        num_cameras = len(cameras)

        if self.random_all:
            self.get_random_views(num_views=num_cameras)

        self.view1 = cameras[self.view1]
        self.view2 = cameras[self.view2]

        frame_index = self.rand_frame_index()
        pixel_index = self.rand_pixel_index()

        view1path, view2path = self.get_vid_paths(path_head=sample_path_head, path_tail=sample_path_tail)

        vid1 = self.load_frames(vid_path=view1path,
                                frame_index=frame_index, pixel_index=pixel_index)
        vid2 = self.load_frames(vid_path=view2path,
                                frame_index=frame_index, pixel_index=pixel_index)
        vid1, vid2 = PanopticDataset.to_tensor(sample=vid1), PanopticDataset.to_tensor(sample=vid2)

        return vid1, vid2

    def process_index(self, index):
        """
        Function to process the information that the data file contains about the sample.
        The line of information contains the sample name as well as the frames to sample from.
        :param index: (int) The index of the sample.
        :return: the sample name, sample path, frame indices, and available viewpoints/cameras
        """
        # ex. 170915_toddler4/samples 0_125
        item = self.data_list[index]
        sample_path, frame_nums = item.split(' ')
        sample_name = sample_path[:sample_path.index('/')]
        # start_frame, end_frame = frame_nums.split('_')
        cameras = self.cam_dict[item]

        return sample_name, sample_path, frame_nums, cameras

    def get_random_views(self, num_views):
        """
        Function to generate 2 randomStuff viewpoints for the sample.
        :param num_views: (int) The number of available view to choose from.
        :return: 2 ints representing the viewpoints for the sample.
        """
        self.view1, self.view2 = np.random.randint(1, num_views), np.random.randint(1, num_views)
        while self.view2 == self.view1:
            self.view2 = np.random.randint(1, num_views)

    def get_vid_paths(self, path_head, path_tail):
        """
        Function to get the paths at which the two sample views are located.
        :param path_head: (str) The first part of the vid path that contains the sample name and dir
        :param path_tail: (str) The last part of the vid path that contains the frame indices
        :return: 2 strings representing the paths for the sample views.
        """
        view1_path = os.path.join(self.root_dir, path_head, str(self.view1), path_tail)
        view2_path = os.path.join(self.root_dir, path_head, str(self.view2), path_tail)

        return view1_path, view2_path

    def rand_frame_index(self):
        """
        Function to generate a randomStuff starting frame index for cropping the temporal dimension of the video.
        :return: The starting frame index for the sample.
        """
        max_frame = self.frame_count - (self.skip_len * self.clip_len)
        assert max_frame >= 1, 'Not enough frames to sample from.'
        frame_index = np.random.randint(0, max_frame)

        return frame_index

    def rand_pixel_index(self):
        """
        Function to generate a randomStuff starting pixel for cropping the height and width of the frames.
        :return: 2 ints representing the starting pixel's x and y coordinates.
        """
        # if the sample is being precropped, then there are 50 pixels removed from the left and right
        if self.precrop:
            width = self.width - 100
        else:
            width = self.width
        height_index = np.random.randint(0, self.height - self.resize_height)
        width_index = np.random.randint(0, width - self.resize_width)

        return height_index, width_index

    def load_frames(self, vid_path, frame_index, pixel_index):
        """
        Function to load the video frames for the sample.
        :param vid_path: (str) The path at which the sample video is located.
        :param frame_index: (int) The starting frame index for the sample.
        :param pixel_index: (tuple: int, int) The height and width indices of the pixel at which to crop.
        :return: np array representing the sample video clip.
        """
        buffer = np.empty((self.clip_len, self.resize_height, self.resize_width, self.channels), np.dtype('float32'))

        # retrieve and crop each frame for the clip
        for i in range(self.clip_len):
            # retrieve the frame (next 9 lines)
            frame_num = frame_index + (i * self.skip_len)
            frame_name = PanopticDataset.make_frame_name(frame_num=frame_num)
            frame_path = os.path.join(vid_path, frame_name)
            assert os.path.exists(frame_path), 'Frame path {} DNE.'.format(frame_path)
            try:
                frame = cv2.imread(frame_path)
            except:
                print('The image did not successfully load.')
            frame = np.array(frame).astype(np.float32)

            # crop the frame (next 3 lines)
            height_index, width_index = pixel_index
            frame = self.crop_frame(frame=frame,
                                    h_index=height_index, w_index=width_index)
            # normalize
            frame = PanopticDataset.normalize_frame(frame)

            # add the frame to the buffer (clip)
            buffer[i] = frame

        return buffer

    @staticmethod
    def make_frame_name(frame_num):
        """
        Function to correctly generate the correctly formatted .png file name for the frame.
        :param frame_num: The frame number captured in the file.
        :return: str representing the file name.
        """
        return str(frame_num).zfill(3) + '.png'

    def crop_frame(self, frame, h_index, w_index):
        """
        Function that crops a frame.
        :param frame: (array-like) The frame to crop.
        :param h_index: (int) The height index to start the crop.
        :param w_index: (int) The width index to start the crop.
        :return: np array representing the cropped frame
        """
        frame = np.array(frame).astype(np.float32)
        if self.precrop:
            frame = frame[:, 50:-50]
        cropped_frame = frame[h_index:h_index + self.resize_height, w_index:w_index + self.resize_width]

        return cropped_frame

    @staticmethod
    def normalize_frame(frame):
        """
        Function to normalize the pixel values in the frame to be between 0 and 1.
        :param frame: (array-like) The frame to be normalized.
        :return: (np array) The normalized frame.
        """
        frame = np.array(frame).astype(np.float32)
        return np.divide(frame, 255.0)

    @staticmethod
    def to_tensor(sample):
        """
        Function to convert the sample clip to a tensor.
        :param sample: (np array) The sample to convert.
        :return: a tensor representing the sample clip.
        """
        # np is (temporal, height, width, channels)
        # pytorch tensor is (channels, temporal, height, width)
        sample = np.transpose(sample, (3, 0, 1, 2))

        return sample
