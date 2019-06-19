# phase 2

from torch.utils.data.dataset import Dataset
import os
import numpy as np
import cv2
import _pickle as pickle


class PanopticDataset(Dataset):
    """Panoptic Dataset"""

    def __init__(self, root_dir, data_file, resize_height, resize_width, clip_len,
                 height=128, width=128, frame_count=125,
                 skip_len=1, random_views=False, random_all=False,
                 precrop=False):

        with open(data_file) as f:
            self.data_file = f.readlines()
        self.data_file = [line.strip() for line in self.data_file]

        self.root_dir = root_dir
        self.clip_len = clip_len
        self.skip_len = skip_len
        self.height = height
        self.width = width
        self.frame_count = frame_count
        self.resize_height = resize_height
        self.resize_width = resize_width
        self.channels = 3
        self.view1 = 1
        self.view2 = 2
        if random_views:
            self.get_random_views()
        self.random_all = random_all
        self.precrop = precrop

    def __len__(self):
        """
        Function to return the number of samples in the dataset.
        :return: int representing the number of samples in the dataset.
        """
        return len(self.data_file)

    def __getitem__(self, idx):
        """
        Function to get a single sample from the dataset. All modifications to the sample are done here.
        :param idx: (int) The index of the sample to get.
        :return: 2 tensors representing the sample video from 2 different viewpoints
        """
        sample_name, sample_path_head, sample_path_tail = self.process_index(index=idx)
        cameras = os.listdir(os.path.join(self.root_dir, sample_path_head))
        num_cameras = len(cameras)

        if self.random_all:
            self.get_random_views(num_views=num_cameras)

        self.view1 = cameras[self.view1]
        self.view2 = cameras[self.view2]

        frame_index = self.rand_frame_index()
        pixel_index = self.rand_pixel_index()

        vp1 = self.get_view(seq_id=sample_name, view_id=self.view1, x_pos=pixel_index[0], y_pos=pixel_index[1])
        vp2 = self.get_view(seq_id=sample_name, view_id=self.view2, x_pos=pixel_index[0], y_pos=pixel_index[1])

        view1path, view2path = self.get_vid_paths(path_head=sample_path_head, path_tail=sample_path_tail)

        vid1 = self.load_frames(vid_path=view1path,
                                frame_index=frame_index, pixel_index=pixel_index)
        vid2 = self.load_frames(vid_path=view2path,
                                frame_index=frame_index, pixel_index=pixel_index)
        vid1, vid2 = PanopticDataset.to_tensor(sample=vid1), PanopticDataset.to_tensor(sample=vid2)

        # print(vp1)
        # print(vp2)
        return vp1, vp2, vid1, vid2

    # MAKE SURE TO NOW GO CHANGE TRAINING AND TESTING LOOPS #

    # CHANGED
    def process_index(self, index):
        """
        Function to process the information that the data file contains about the sample.
        The line of information contains the sample name as well as the number of available frames from each view
        :param index: (int) The index of the sample.
        :return: the action class, sample_id, and two ints representing the number of frames in view1 and view2
        """
        # ex. 170915_toddler4/samples 0_125
        sample_path, frame_nums = self.data_file[index].split(' ')
        sample_name = sample_path[:sample_path.index('/')]
        # start_frame, end_frame = frame_nums.split('_')

        info = (sample_name, sample_path, frame_nums)

        return info

    def get_random_views(self, num_views):
        """
        Function to generate 2 random viewpoints for the sample.
        :return: 2 ints representing the viewpoints for the sample.
        """
        self.view1, self.view2 = np.random.randint(1, num_views), np.random.randint(1, num_views)
        while self.view2 == self.view1:
            self.view2 = np.random.randint(1, num_views)

    def get_vid_paths(self, path_head, path_tail):
        """
        Function to get the paths at which the two sample views are located.
        :param action: (int) The action class that the sample captures.
        :param sample_id: (str) The id for the sample from the data file.
        :return: 2 strings representing the paths for the sample views.
        """
        view1_path = os.path.join(self.root_dir, path_head, str(self.view1), path_tail)
        view2_path = os.path.join(self.root_dir, path_head, str(self.view2), path_tail)

        return view1_path, view2_path

    def rand_frame_index(self):
        """
        Function to generate a random starting frame index for cropping the temporal dimension of the video.
        :param frame_count: (int) The number of available frames in the sample video.
        :param clip_len: (int) The number of frames desired in the sample clip.
        :param skip_len: (int) The number of frames to skip between each when creating the clip.
        :return: The starting frame index for the sample.
        """
        max_frame = self.frame_count - (self.skip_len * self.clip_len)
        assert max_frame >= 1, 'Not enough frames to sample from.'
        frame_index = np.random.randint(0, max_frame)

        return frame_index

    def rand_pixel_index(self):
        """
        Function to generate a random starting pixel for cropping the height and width of the frames.
        :param height: (int) The height of the video.
        :param width: (int) The width of the video.
        :param desired_height: (int) The desired height of the video.
        :param desired_width: (int) The desired width of the video.
        :return: 2 ints representing the starting pixel's x and y coordinates.
        """
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
        :param clip_len: (int) The number of frames desired in the sample clip.
        :param skip_len: (int) The number of frames to skip between each when creating the clip.
        :param frame_index: (int) The starting frame index for the sample.
        :param pixel_index: (tuple: int, int) The height and width indices of the pixel at which to crop.
        :param resize_height: (int) The desired frame height.
        :param resize_width: (int) The desired frame width.
        :param channels: (int) The number of channels in each frame.
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
                                    height_index=height_index, width_index=width_index)
            # normalize
            frame = PanopticDataset.normalize_frame(frame)

            # add the frame to the buffer (clip)
            buffer[i] = frame

        return buffer

    @staticmethod
    def make_frame_name(frame_num):
        """
        Function to correctly generate the correctly formatted .jpg file name for the frame.
        :param frame_num: The frame number captured in the file.
        :return: str representing the file name.
        """
        return str(frame_num).zfill(3) + '.png'

    def crop_frame(self, frame, height_index, width_index):
        """
        Function that crops a frame.
        :param frame: (array-like) The frame to crop.
        :param height_index: (int) The height index to start the crop.
        :param width_index: (int) The width index to start the crop.
        :param desired_height: (int) The desired height of the cropped frame.
        :param desired_width: (int) The desired width of the cropped frame.
        :return: np array representing the cropped frame
        """
        frame = np.array(frame).astype(np.float32)
        if self.precrop:
            frame = frame[:, 50:-50]
        cropped_frame = frame[height_index:height_index + self.resize_height,
                        width_index:width_index + self.resize_width]

        return cropped_frame

    @staticmethod
    def normalize_frame(frame):
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

    def get_view(self, seq_id, view_id, x_pos, y_pos):
        # seq = os.path.split(seq_id)[0]
        cal_file = os.path.join(self.root_dir, seq_id, 'calibration_' + seq_id + '.pkl')
        # print cal_file

        # load the calibration file
        with open(cal_file, 'rb') as fp:
            cal = pickle.load(fp)

        # get the camera calibration values
        # print(seq_id)
        # print(cal)
        try:
            c_data = cal[view_id[4:]]
        except:
            print
            seq_id, cal_file
        # c_data = cal[view_id]
        R = c_data["R"]
        t = c_data["t"]
        # print seq_id

        # camera intrinsic k?
        dc = c_data["distCoef"]
        K = c_data["K"]
        c_x, c_y = K[0][2] / 480., K[1][2] / 640.
        f_x, f_y = K[0][0] / 480., K[1][1] / 640.

        # compute the viewpoint here
        x, y, z = -np.dot(np.linalg.inv(R), t) / 100.
        x, y, z = x[0], y[0], z[0]

        # new run with c, f, and dc additional

        pan = 1. * x_pos / (self.width - self.resize_width)
        van = 1. * y_pos / (self.height - self.resize_height)

        # return np.array([x, y, z, c_x, c_y, f_x, f_y, dc[0], dc[1], dc[2] * 100, dc[3] * 100, dc[4], pan, van])
        # print('x{}y{}z{}pan{}van{}'.format(x, y, z, pan, van))
        return np.array([x, y, z, pan, van])
