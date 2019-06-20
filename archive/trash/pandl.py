import numpy as np
import keras
import cPickle as pickle
import random
import params
import cv2
import os
import json

import params as params


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, list_IDs, batch_size=32, num_frames=8,
                 crop_size=112, num_channels=3, num_views=8,
                 shuffle=True, skip_rate=3, unseen_testing=False, train=True):
        'Initialization'
        self.num_frames = num_frames
        self.crop_size = crop_size
        self.num_views = num_views
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.num_channels = num_channels
        self.shuffle = shuffle
        self.train = train
        self.skip_rate = skip_rate

        self.view_dims = params.view_dims
        self.noise_dims = params.noise_dims

        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return len(self.list_IDs)

    def __getitem__(self, index):
        'Generate one sample'
        # Generate indexes of the batch. Index is stored by the parent class to indicate the batch number.
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        clips, view, target, t_view, t_noise = self.__data_generation(list_IDs_temp)

        return [clips, view, t_view, t_noise], target

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def get_scene_parameters(self, scene):
        return self.view_params[scene][1], self.view_params[scene][2]

    def _get_view(self, seq_id, view_id, x_pos, y_pos):
        seq = os.path.split(seq_id)[0]
        cal_file = os.path.join(params.rgb_data, seq, 'calibration_' + seq + '.pkl')
        # print cal_file

        # load the calibration file
        with open(cal_file) as fp:
            cal = pickle.load(fp)

        # get the camera calibration values
        # print seq_id
        # print cal
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

        # new run with c, f, and dc additional

        pan = 1. * x_pos / (params.img_size - self.crop_size)
        van = 1. * y_pos / (params.img_size - self.crop_size)

        return np.array([x, y, z, c_x, c_y, f_x, f_y, dc[0], dc[1], dc[2] * 100, dc[3] * 100, dc[4], pan, van])
        # return np.array([x, y, z, pan, van])

    def _get_target(self, ID, view_id, f_start):
        vid = ID[0]

        frames = np.empty((self.num_frames, self.crop_size, self.crop_size, self.num_channels))
        t_view = np.empty((1, 1, self.view_dims))

        s_head = ID[0]  # path
        s_tail = ID[1]  # frame numbers
        s_mid = view_id[0]
        v_path = os.path.join(params.rgb_data, s_head, s_mid, s_tail)

        crop_pos_x = np.random.randint(0, params.img_size - self.crop_size)
        crop_pos_y = np.random.randint(0, params.img_size - self.crop_size)

        # collect frames for this view
        for j in range(params.num_frames):
            f_path = os.path.join(v_path, '{:03d}.png'.format(f_start + j * self.skip_rate))
            img = cv2.imread(f_path)
            img_sample = img[crop_pos_x:crop_pos_x + self.crop_size, crop_pos_y:crop_pos_y + self.crop_size]
            frames[j,] = (img_sample - 128.) / 128.

        t_view[0, 0,] = self._get_view(ID[0], s_mid, crop_pos_x, crop_pos_y)
        # print t_view[0,0,]

        t_noise = np.random.rand(1, 1, self.noise_dims)
        # print t_noise

        return frames, t_view, t_noise

    def _get_samples(self, ID, views_list, ids, f_start):

        frames = np.empty((self.num_views, self.num_frames, self.crop_size, self.crop_size, self.num_channels))
        view = np.empty((self.num_views, self.num_frames, 1, 1, self.view_dims))

        s_head = ID[0]
        s_tail = ID[1]
        # iterate through all views and collect frames
        for i, view_id in enumerate(ids):
            s_mid = views_list[view_id][0]
            v_path = os.path.join(params.rgb_data, s_head, s_mid, s_tail)

            crop_pos_x = np.random.randint(0, params.img_size - self.crop_size)
            crop_pos_y = np.random.randint(0, params.img_size - self.crop_size)

            # collect frames for this view
            for j in range(params.num_frames):
                f_path = os.path.join(v_path, '{:03d}.png'.format(f_start + j * self.skip_rate))
                img = cv2.imread(f_path)
                img_sample = img[crop_pos_x:crop_pos_x + self.crop_size, crop_pos_y:crop_pos_y + self.crop_size]
                frames[i, j,] = (img_sample - 128.) / 128.

                view[i, j, 0, 0,] = self._get_view(ID[0], s_mid, crop_pos_x, crop_pos_y)

        # print j, 'get_samples'
        return frames, view

    def _get_data(self, ID):
        # randomly select num_views+1 clip from available views
        # and pick one for rendering and others for learning
        # first load the available views for this sample
        seq_id = os.path.split(ID[0])[0]
        f_list = params.train_list
        if not self.train and unseen_testing:
            f_list = params.test_list

        vl_path = os.path.join(params.rgb_data, seq_id, f_list)
        views_list = np.loadtxt(vl_path, dtype=str)
        # + 1 for rendering
        ids = np.random.randint(0, len(views_list), params.num_views + 1)

        num_req_frames = self.num_frames * self.skip_rate - self.skip_rate + 1
        # select random start frame
        # all views will have same starting point
        f_start = np.random.randint(0, params.fcount - num_req_frames)

        frames, views = self._get_samples(ID, views_list, ids[:-1], f_start)

        target, t_view, t_noise = self._get_target(ID, views_list[ids[-1]], f_start)

        return frames, views, target, t_view, t_noise

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
        # Initialization
        frames = np.empty(
            (self.batch_size, self.num_views, self.num_frames, self.crop_size, self.crop_size, self.num_channels))
        view = np.empty((self.batch_size, self.num_views, self.num_frames, 1, 1, self.view_dims))

        target = np.empty((self.batch_size, self.num_frames, self.crop_size, self.crop_size, self.num_channels))
        # t_view = np.empty((self.batch_size, self.num_frames, 1, 1, self.view_dims))
        # t_noise = np.empty((self.batch_size, self.num_frames, 1, 1, self.noise_dims))
        t_view = np.empty((self.batch_size, 1, 1, self.view_dims))
        t_noise = np.empty((self.batch_size, 1, 1, self.noise_dims))

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            frames[i,], view[i,], target[i,], t_view[i,], t_noise[i,] = self._get_data(ID)

        return frames, view, target, t_view, t_noise

