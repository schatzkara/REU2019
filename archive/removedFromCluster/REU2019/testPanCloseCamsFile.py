import torch
from PanopticDataLoader import PanopticDataset


if __name__ == '__main__':

    data_root_dir = '/home/c2-2/yogesh/datasets/panoptic/rgb_data/'
    train_split = '/home/yogesh/kara/data/panoptic/mod_train.list'
    test_split = '/home/yogesh/kara/data/panoptic/mod_test.list'
    close_cams_file = '/home/yogesh/kara/data/panoptic/closecams.list'

    RANDOM_ALL = True
    BATCH_SIZE = 32
    CHANNELS = 3
    FRAMES = 8
    SKIP_LEN = 2
    HEIGHT = 112
    WIDTH = 112
    PRECROP = False
    CLOSE_VIEWS = True
    VIEW_DIST_MAX = 10

    trainset = PanopticDataset(root_dir=data_root_dir, data_file=train_split,
                               resize_height=HEIGHT, resize_width=WIDTH,
                               clip_len=FRAMES, skip_len=SKIP_LEN,
                               random_all=RANDOM_ALL, close_views=CLOSE_VIEWS,
                               close_cams_file=close_cams_file, precrop=PRECROP)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    
    testset = PanopticDataset(root_dir=data_root_dir, data_file=train_split,
                               resize_height=HEIGHT, resize_width=WIDTH,
                               clip_len=FRAMES, skip_len=SKIP_LEN,
                               random_all=RANDOM_ALL, close_views=CLOSE_VIEWS,
                               close_cams_file=close_cams_file, precrop=PRECROP)
    testloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    print('TRAINING...')
    for batch_idx, (vp_diff, vid1, vid2) in enumerate(trainloader):
        print('{} {} {} {}'.format(batch_idx, vp_diff.size(), vid1.size(), vid2.size()))

    print('TESTING...')
    for batch_idx, (vp_diff, vid1, vid2) in enumerate(testloader):
        print('{} {} {} {}'.format(batch_idx, vp_diff.size(), vid1.size(), vid2.size()))