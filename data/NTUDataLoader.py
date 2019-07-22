
from torch.utils.data.dataset import Dataset
import os
import numpy as np
import cv2
import torch


class NTUDataset(Dataset):
    """NTU Dataset"""

    def __init__(self, root_dir, data_file, param_file,
                 resize_height, resize_width, clip_len, height=135, width=240, precrop=False, skip_len=1,
                 view1=1, view2=2, random_views=False, random_all=False,
                 diff_actors=False, diff_scenes=False
                 ):
        """
        Initializes the NTU Dataset object used for extracting samples to run through the network.
        :param root_dir: (str) Directory with all the frames extracted from the videos.
        :param data_file: (str) Path to .txt file containing the sample IDs.
        :param resize_height: (int) The desired frame height.
        :param resize_width: (int) The desired frame width.
        :param clip_len: (int) The number of frames desired in the sample clip.
        :param height: (int, optional) The height of the frames in the dataset (default 135 for NTU Dataset).
        :param width: (int, optional) The width of the frames in the dataset (default 240 for NTU Dataset).
        :param precrop: (boolean, optional) True to crop 50 pixels off left and right of frame before randomly cropping
                        (default False).
        :param skip_len: (int, optional) The number of frames to skip between each when creating the clip (default 1).
        :param view1: (int, optional) The desired viewpoint to use as the first view; can be 1, 2, or 3 (default 1).
        :param view2: (int, optional) The desired viewpoint to use as the seconds view; can be 1, 2, or 3 (default 2).
        :param random_views: (boolean, optional) True to use 2 constant randomly generated views (default False).
        :param random_all: (boolean, optional) True to use 2 randomly generated views for each sample (default False).
        :param diff_actors: (boolean, optional) True to use different actors in view 1 and view 2 (default False).
        :param diff_scenes: (boolean, optional) True to use different scenes in view 1 and view 2 (default False).
        """
        with open(data_file) as f:
            self.data_file = f.readlines()
        self.data_file = [line.strip() for line in self.data_file]
        self.root_dir = root_dir
        self.param_file = param_file

        self.clip_len = clip_len
        self.skip_len = skip_len
        self.height = height
        self.width = width
        self.resize_height = resize_height
        self.resize_width = resize_width
        self.precrop = precrop
        self.channels = 3

        self.view1 = view1
        self.view2 = view2
        if random_views:
            self.get_random_views()
        self.random_all = random_all
        self.view_params = self.load_view_params()

        self.diff_actors = diff_actors
        self.diff_scenes = diff_scenes

        # make dicts that list the possible view2 samples and information for a given view1 sample
        if self.diff_actors or self.diff_scenes:
            self.sample_dict = self.make_sample_dict()
        if self.diff_actors and self.diff_scenes:
            self.ra_dict = self.make_ra_dict()
        elif self.diff_actors:
            self.sra_dict = self.make_sra_dict()
        elif self.diff_scenes:
            self.pra_dict = self.make_pra_dict()

    def load_view_params(self):
        """
        Function to load the params associated with the camera views for the NTU Dataset.
        :return: An array containing all the view params.
        """
        view_params = np.loadtxt(self.param_file)
        # normalize the distances
        view_params /= view_params.max(axis=0)

        return view_params

    def make_sample_dict(self):
        """
        Function to make a dictionary that contains all the samples and the information associated with them.
        :return: A dictionary with keys: sample_id and values: the sample's information from the data_file.
        """
        sample_dict = {}

        for sample in self.data_file:
            sample_info = sample.split(' ')
            sample_id = sample_info[0][sample_info[0].index('/') + 1:]
            sample_dict[sample_id] = sample

        return sample_dict

    def make_sra_dict(self):
        """
        Function to create a dictionary of samples that share the same scene, repetition, and action. Used when
        diff_actors is true and diff_scenes is false to find a valid view2 sample given the SRA id from the view1
        sample. The SRA id indicates the scene, repetition, and action of the sample.
        :return: A dictionary with keys: SRA id (ex. S001R001A001) and values: list of samples with that SRA id.
        """
        SRA = {}  # SceneRepetitionAction

        for sample in self.data_file:
            sample_info = sample.split(' ')
            sample_id = sample_info[0][sample_info[0].index('/') + 1:]
            scene, pid, rid, action = self.decrypt_vid_name(sample_id)
            sra = self.get_sra(scene, rid, action)
            if sra not in SRA.keys():
                SRA[sra] = [sample_id]
            else:
                SRA[sra].append(sample_id)

        return SRA

    def make_pra_dict(self):
        """
        Function to create a dictionary of samples that share the same person, repetition, and action. Used when
        diff_scenes is true and diff_actors is false to find a valid view2 sample given the PRA id from the view1
        sample. The PRA id indicates the person, repetition, and action of the sample.
        :return: A dictionary with keys: PRA id (ex. P001R001A001) and values: list of samples with that PRA id.
        """
        PRA = {}  # PersonRepetitionAction

        for sample in self.data_file:
            sample_info = sample.split(' ')
            sample_id = sample_info[0][sample_info[0].index('/') + 1:]
            scene, pid, rid, action = self.decrypt_vid_name(sample_id)
            pra = self.get_pra(pid, rid, action)
            if pra not in PRA.keys():
                PRA[pra] = [sample_id]
            else:
                PRA[pra].append(sample_id)

        return PRA

    def make_ra_dict(self):
        """
        Function to create a dictionary of samples that share the same repetition, and action. Used when
        diff_actors and diff_scenes are both true to find a valid view2 sample given the RA id from the view1 sample.
        The RA id indicates the repetition and action of the sample.
        :return: A dictionary with keys: RA id (ex. R001A001) and values: list of samples with that RA id.
        """
        RA = {}  # RepetitionAction

        for sample in self.data_file:
            sample_info = sample.split(' ')
            sample_id = sample_info[0][sample_info[0].index('/') + 1:]
            scene, pid, rid, action = self.decrypt_vid_name(sample_id)
            ra = self.get_ra(rid, action)
            if ra not in RA.keys():
                RA[ra] = [sample_id]
            else:
                RA[ra].append(sample_id)

        return RA

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
        :return: a float representing the viewpoint difference from the perspective of the first view, i.e. a negative
                 value indicates that the second view is clockwise from the first and a positive values indicates that
                 the second value is counterclockwise from the first,
                 and 2 tensors representing the sample video from 2 viewpoints
        """
        if self.random_all:
            self.get_random_views()

        action, sample_id_v1, sample_id_v2, vp1, vp2, nf_v1, nf_v2 = self.process_index(index=idx)
        view1path = self.get_vid_path(action=action, sample_id=sample_id_v1, view_id=self.view1)
        view2path = self.get_vid_path(action=action, sample_id=sample_id_v2, view_id=self.view2)

        vp_diff = vp2 - vp1

        frame_count = min(nf_v1, nf_v2)
        frame_index = self.rand_frame_index(frame_count=frame_count)
        pixel_index = self.rand_pixel_index()

        vid1 = self.load_frames(vid_path=view1path,
                                frame_index=frame_index, pixel_index=pixel_index)
        vid2 = self.load_frames(vid_path=view2path,
                                frame_index=frame_index, pixel_index=pixel_index)
        vid1, vid2 = self.to_tensor(sample=vid1), self.to_tensor(sample=vid2)

        return vp_diff, vid1, vid2

    def get_random_views(self):
        """
        Function to generate 2 random viewpoints for the sample. Automatically updates view1 and view2
        :return: None
        """
        self.view1, self.view2 = np.random.randint(1, 4), np.random.randint(1, 4)
        while self.view2 == self.view1:
            self.view2 = np.random.randint(1, 4)

    def process_index(self, index):
        """
        Function to process the information that the data file contains about the sample.
        The line of information contains the sample name as well as the number of available frames from each view
        :param index: (int) The index of the sample.
        :return: the action class, the view1 and view2 sample_ids, two floats representing the viewpoint angles, and
                 two ints representing the number of frames in view1 and view2
        """
        sample_info = self.data_file[index].split(' ')
        sample_id = sample_info[0][sample_info[0].index('/') + 1:]
        scene, pid, rid, action = self.decrypt_vid_name(vid_name=sample_id)

        sample_id_v1 = sample_id
        sample_info_v1 = sample_info
        if self.diff_actors and self.diff_scenes:
            ra = self.get_ra(rid, action)
            sample_id_v2 = np.random.choice(self.ra_dict[ra])
            sample_info_v2 = self.sample_dict[sample_id_v2].split(' ')
        elif self.diff_actors:
            sra = self.get_sra(scene, rid, action)
            sample_id_v2 = np.random.choice(self.sra_dict[sra])
            sample_info_v2 = self.sample_dict[sample_id_v2].split(' ')
        elif self.diff_scenes:
            pra = self.get_pra(pid, rid, action)
            sample_id_v2 = np.random.choice(self.pra_dict[pra])
            sample_info_v2 = self.sample_dict[sample_id_v2].split(' ')
        else:
            sample_id_v2 = sample_id
            sample_info_v2 = sample_info

        angle_v1 = self.get_viewing_angle(rid=rid, cam=self.view1)
        angle_v2 = self.get_viewing_angle(rid=rid, cam=self.view2)

        nf_v1 = int(sample_info_v1[self.view1])
        nf_v2 = int(sample_info_v2[self.view2])

        return action, sample_id_v1, sample_id_v2, angle_v1, angle_v2, nf_v1, nf_v2

        # nf_v1, nf_v2, nf_v3 = sample_info_v1[1:]
        # nf_v1, nf_v2, nf_v3 = int(nf_v1), int(nf_v2), int(nf_v3)
        #
        # info = (action, sample_id_v1, sample_id_v2, angle_v1, angle_v2)
        # if self.view1 == 1:
        #     info = info + (nf_v1,)
        # elif self.view1 == 2:
        #     info = info + (nf_v2,)
        # elif self.view1 == 3:
        #     info = info + (nf_v3,)
        #
        # nf_v1, nf_v2, nf_v3 = sample_info_v2[1:]
        # nf_v1, nf_v2, nf_v3 = int(nf_v1), int(nf_v2), int(nf_v3)
        # if self.view2 == 1:
        #     info = info + (nf_v1,)
        # elif self.view2 == 2:
        #     info = info + (nf_v2,)
        # elif self.view2 == 3:
        #     info = info + (nf_v3,)
        #
        # return info

    @staticmethod
    def get_viewing_angle(rid, cam):
        """
        Function to get the camera viewing angles.
        :param rid: (int) The repetition ID of the sample video. Legal values are 1 and 2.
        :param cam: (int) The camera ID. Legal values are 1, 2, and 3.
        :return: float representing the camera viewing angle.
        """
        vpt = 0.
        pi = 22 / 7.
        # rid-1 implies face towards cam3; rid-2 implies face towards cam2; cam1 is the center camera
        if rid == 1:
            if cam == 1:
                vpt = pi / 4
            elif cam == 2:
                vpt = pi / 2
            elif cam == 3:
                vpt = 0.00
        elif rid == 2:
            if cam == 1:
                vpt = -pi / 4
            elif cam == 2:
                vpt = 0.00
            elif cam == 3:
                vpt = -pi / 2

        return vpt

    def get_vid_path(self, action, sample_id, view_id):
        """
        Function to get the paths at which the two sample views are located.
        :param action: (int) The action class that the sample captures.
        :param sample_id: (str) The id for the sample from the data file.
        :param view_id: (int) The view number of the sample.
        :return: A string representing the path for the sample view.
        """
        vid_path = self.make_sample_path(action=action, sample_id=sample_id)
        view_path = os.path.join(vid_path, str(view_id))

        return view_path

    def make_sample_path(self, action, sample_id):
        """
        Function to make the path at which the sample is located.
        :param action: (int) The action class that the sample captures.
        :param sample_id: (str) The id for the sample from the data file.
        :return: The path at which the sample is located.
        """
        vid_path = os.path.join(self.root_dir, str(action), sample_id)

        return vid_path

    @staticmethod
    def decrypt_vid_name(vid_name):
        """
        Function to break up the meaning of the video name.
        :param vid_name: (string) The name of the video.
        :return: 4 ints representing the scene, person, repetition, and action that the video captures.
        """
        scene = int(vid_name[1:4])
        pid = int(vid_name[5:8])
        rid = int(vid_name[9:12])
        action = int(vid_name[13:16])

        return scene, pid, rid, action

    @staticmethod
    def get_sra(scene, rid, action):
        """
        Function to create the SRA id for a sample, which indicates the scene, repetition, and action of the sample.
        :param scene: (int) The scene id for the sample.
        :param rid: (int) The repetition id for the sample.
        :param action: (int) The action class id for the sample.
        :return: A string representing the SRA id for the sample.
        """
        return 'S' + str(scene).zfill(3) + 'R' + str(rid).zfill(3) + 'A' + str(action).zfill(3)

    @staticmethod
    def get_pra(pid, rid, action):
        """
        Function to create the PRA id for a sample, which indicates the person, repetition, and action of the sample.
        :param pid: (int) The person/actor id for the sample.
        :param rid: (int) The repetition id for the sample.
        :param action: (int) The action class id for the sample.
        :return: A string representing the PRA id for the sample.
        """
        return 'P' + str(pid).zfill(3) + 'R' + str(rid).zfill(3) + 'A' + str(action).zfill(3)

    @staticmethod
    def get_ra(rid, action):
        """
        Function to create the RA id for a sample, which indicates the repetition and action of the sample.
        :param rid: (int) The repetition id for the sample.
        :param action: (int) The action class id for the sample.
        :return: A string representing the RA id for the sample.
        """
        return 'R' + str(rid).zfill(3) + 'A' + str(action).zfill(3)

    def rand_frame_index(self, frame_count):
        """
        Function to generate a random starting frame index for cropping the temporal dimension of the video.
        :param frame_count: (int) The number of available frames in the sample video.
        :return: The starting frame index for the sample.
        """
        max_frame = frame_count - (self.skip_len * self.clip_len)
        assert max_frame >= 1, 'Not enough frames to sample from.'
        frame_index = np.random.randint(0, max_frame)

        return frame_index

    def rand_pixel_index(self):
        """
        Function to generate a random starting pixel for cropping the height and width of the frames.
        :return: 2 ints representing the starting pixel's x and y coordinates.
        """
        # if the frame is precropped, then 50 pixels are removed from the right and left each.
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
            frame_name = self.make_frame_name(frame_num=frame_num)
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
            frame = self.normalize_frame(frame)

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
        return str(frame_num).zfill(3) + '.jpg'

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
