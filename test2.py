from phase2.PanopticDataLoader import PanopticDataset
import torch

data_root_dir = 'C:/Users/Owner/Documents/UCF/Project/panoptic/rgb_data/'
test_splits = 'C:/Users/Owner/Documents/UCF/Project/REU2019/data/Panoptic/one.list'

# VIEW1 = 1
# VIEW2 = 2
BATCH_SIZE = 32
CHANNELS = 3
FRAMES = 8
SKIP_LEN = 2
HEIGHT = 112
WIDTH = 112
PRECROP = False

if __name__ == '__main__':
    testset = PanopticDataset(root_dir=data_root_dir, data_file=test_splits,
                              resize_height=HEIGHT, resize_width=WIDTH,
                              clip_len=FRAMES, skip_len=SKIP_LEN,
                              random_all=True, precrop=PRECROP)
    testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=1)

    for batch_idx, (vp1, vp2, view1vid, view2vid) in enumerate(testloader):
        print(view1vid.size())
        # print(view1vid)
        print(view2vid.size())
        # print(view2vid)
        print(vp1)
        print(vp2)
        print(vp1.size())
