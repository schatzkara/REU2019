# phase 2

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
                 close_views=False, close_cams_file=None,
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
        with open(data_file, 'r') as f:
            data_file = f.readlines()
        self.data_list = [line.strip() for line in data_file]
        self.root_dir = root_dir
        self.clip_len = clip_len
        self.skip_len = skip_len
        self.height = height
        self.width = width
        self.frame_count = frame_count
        self.resize_height = resize_height
        self.resize_width = resize_width
        self.channels = 3
        self.num_views = num_views
        self.view1idx = None
        self.view2idx = None
        self.view1 = None
        self.view2 = None
        # if random_views:
        #     self.get_random_views(self.num_views)
        self.random_all = random_all
        self.close_views = close_views
        if self.close_views:
            self.close_cams = self.make_close_cams_dict(close_cams_file)
        self.precrop = precrop

    @staticmethod
    def make_close_cams_dict(cam_file):
        close_cams_dict = {}
        with open(cam_file, 'r') as f:
            f = f.readlines()
            f = [line.strip() for line in f]
            current_sample = ''
            for i in range(len(f)):
                line = f[i]
                if not line.startswith('vga'):
                    current_sample = line
                    close_cams_dict[current_sample] = {}
                else:
                    cams = line.strip().split(' ')
                    close_cams_dict[current_sample][cams[0]] = cams[1:]

        return close_cams_dict

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
        :return: A floats representing the difference between the viewpoints,
                 and a tensor representing the viewpoint difference i.e. the viewpoint change to get from v1 to v2.
        """
        sample_name, sample_path_head, sample_path_tail = self.process_index(index=idx)
        # print('processed index')
        self.get_viewpoints(sample_name=sample_name,
                            sample_path_head=sample_path_head,
                            sample_path_tail=sample_path_tail)

        frame_index = self.rand_frame_index()
        pixel_index = self.rand_pixel_index()

        vp1 = self.get_view(seq_id=sample_name, view_id=self.view1, x_pos=pixel_index[0], y_pos=pixel_index[1])[:3]
        vp2 = self.get_view(seq_id=sample_name, view_id=self.view2, x_pos=pixel_index[0], y_pos=pixel_index[1])[:3]

        vp_diff = vp2 - vp1

        view1path = self.get_vid_path(path_head=sample_path_head, path_tail=sample_path_tail, view_num=1)
        view2path = self.get_vid_path(path_head=sample_path_head, path_tail=sample_path_tail, view_num=2)

        vid1 = self.load_frames(vid_path=view1path, frame_index=frame_index, pixel_index=pixel_index)
        vid2 = self.load_frames(vid_path=view2path, frame_index=frame_index, pixel_index=pixel_index)

        # print('got sample')
        return vp_diff, vid1, vid2

    def process_index(self, index):
        """
        Function to process the information that the data file contains about the sample.
        The line of information contains the sample name as well as the frames to sample from.
        :param index: (int) The index of the sample.
        :return: the sample name, sample path, and frame indices
        """
        # ex. 170915_toddler4/samples 0_125
        sample_path, frame_nums = self.data_list[index].split(' ')
        sample_name = sample_path[:sample_path.index('/')]

        return sample_name, sample_path, frame_nums

    def get_viewpoints(self, sample_name, sample_path_head, sample_path_tail):
        cameras = os.listdir(os.path.join(self.root_dir, sample_path_head))
        if self.close_views:
            cameras = list(self.close_cams.keys())
        num_cameras = len(cameras)
        # print(num_cameras)

        if self.random_all:
            self.get_legal_view(camera_list=cameras, view_num=1, sample_name=sample_name,
                                sample_path_head=sample_path_head, sample_path_tail=sample_path_tail)
            # print('got one')
            if self.close_views:
                cameras = self.close_cams[sample_name][self.view1]
            self.get_legal_view(camera_list=cameras, view_num=2, sample_name=sample_name,
                                sample_path_head=sample_path_head, sample_path_tail=sample_path_tail)
        # print('done')

    def get_legal_view(self, camera_list, view_num, sample_name, sample_path_head, sample_path_tail):
        # print(camera_list)
        # num_cams = len(camera_list)
        # print(num_cams)

        path_exists = False
        cal_info_avail = False
        while not path_exists or not cal_info_avail:
            num_cams = len(camera_list)
            assert num_cams > 0, 'There are no cameras to choose from.'
            # print(num_cams)
            if self.random_all:
                self.get_random_view(num_options=num_cams, view_num=view_num, cameras=camera_list)
            # print(self.view2idx)
            # if view_num == 1:
            #     self.view1 = camera_list[self.view1idx]
            # elif view_num == 2:
            #     self.view2 = camera_list[self.view2idx]
            # print(self.view2)
            # print('got potential view')
            viewpath = self.get_vid_path(path_head=sample_path_head, path_tail=sample_path_tail, view_num=view_num)
            # print('got view path')
            path_exists = os.path.exists(viewpath)
            if path_exists:
                frames = os.listdir(viewpath)
                if len(frames) == self.frame_count:
                    path_exists = True
                else:
                    path_exists = False
            # print(path_exists)
            # print(viewpath)
            if not path_exists:
                if view_num == 1:
                    camera_list.remove(self.view1)
                elif view_num == 2:
                    camera_list.remove(self.view2)
                continue

            if view_num == 1:
                view_id = self.view1
            elif view_num == 2:
                view_id = self.view2
            cal_info_avail = self.check_cal_info(seq_id=sample_name, view_id=view_id)
            # print(cal_info_avail)
            if not cal_info_avail:
                if view_num == 1:
                    camera_list.remove(self.view1)
                elif view_num == 2:
                    camera_list.remove(self.view2)
                continue

    @staticmethod
    def euclidean_distance(x, y):
        x, y = np.array(x), np.array(y)
        diff = x - y
        square = np.multiply(diff, diff)
        sum = np.sum(square)
        dist = np.sqrt(sum)
        return dist

    def check_cal_info(self, seq_id, view_id):
        cal_file = os.path.join(self.root_dir, seq_id, 'calibration_' + seq_id + '.pkl')
        if not os.path.exists(cal_file):
            print('{} DNE'.format(cal_file))

        # load the calibration file
        with open(cal_file, 'rb') as fp:
            cal = pickle.load(fp)

        try:
            c_data = cal[view_id[4:]]
        except:
            print(seq_id, view_id, cal_file)
            return False
        return True

    def get_random_view(self, num_options, view_num, cameras):
        """
        Function to generate 2 randomStuff viewpoints for the sample.
        :return: 2 ints representing the viewpoints for the sample.
        """
        # print(num_options)
        if view_num == 1:
            self.view1idx = np.random.randint(0, num_options)
            self.view1 = cameras[self.view1idx]
        elif view_num == 2:
            self.view2idx = np.random.randint(0, num_options)
            self.view2 = cameras[self.view2idx]
            while self.view2 == self.view1:
                self.view2idx = np.random.randint(0, num_options)
                self.view2 = cameras[self.view2idx]

    def get_vid_path(self, path_head, path_tail, view_num):
        """
        Function to get the paths at which the two sample views are located.
        :param path_head: (str) The first part of the vid path that contains the sample name and dir
        :param path_tail: (str) The last part of the vid path that contains the frame indices
        :return: 2 strings representing the paths for the sample views.
        """
        # print(self.view1)
        if view_num == 1:
            view_path = os.path.join(self.root_dir, path_head, str(self.view1), path_tail)
            # print(view_path)
        elif view_num == 2:
            view_path = os.path.join(self.root_dir, path_head, str(self.view2), path_tail)
        # print(view_path)
        return view_path

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
                                    height_index=height_index, width_index=width_index)
            # normalize
            frame = PanopticDataset.normalize_frame(frame)

            # add the frame to the buffer (clip)
            buffer[i] = frame
        buffer = PanopticDataset.to_tensor(buffer)

        return buffer

    @staticmethod
    def make_frame_name(frame_num):
        """
        Function to correctly generate the correctly formatted .png file name for the frame.
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

    def get_view(self, seq_id, view_id, x_pos, y_pos):
        """
        Function to get the view of the camera.
        :param seq_id: (str) The sample name.
        :param view_id: (int) The camera ID.
        :param x_pos: (int) The x coordinate of the pixel to crop at.
        :param y_pos: (int) The y coordinate of the pixel to crop at.
        :return: The x, y, and z coordinates of the camera and the pan and van values
        """
        # seq = os.path.split(seq_id)[0]
        cal_file = os.path.join(self.root_dir, seq_id, 'calibration_' + seq_id + '.pkl')
        # print cal_file
        if not os.path.exists(cal_file):
            print('{} DNE'.format(cal_file))

        # load the calibration file
        with open(cal_file, 'rb') as fp:
            cal = pickle.load(fp)

        # get the camera calibration values
        # print(seq_id)
        # print(view_id)
        # print(cal)
        try:
            c_data = cal[view_id[4:]]
        except:
            print(seq_id, view_id, cal_file)
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
