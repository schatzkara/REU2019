
# if __name__ == "__main__":
#     model = FullNetwork()
#     model(0,0)
#     print_summary = True
#
#     vgg = VGG('VGG16')
#     i3d = InceptionI3d(final_endpoint='Mixed_4f')
#     gen = Generator()
#
#     if print_summary:
#         summary(vgg, input_size=(20, 3, 112, 112))
#         summary(i3d, input_size=(20, 3, 8, 112, 112))
#         summary(gen, input_size=(20, 512, 3, 7, 7))
#



for value in training_data.items():
    print(value[1].size())
for value in testing_data.items():
    print(value[1].size())

for value in training_batches.values():
    print(value.size())



x = torch.zeros([3, 4, 4])
y = torch.zeros([3, 2, 4, 4])
print('x:', x)
print('y:', y)

z = torch.unsqueeze(x, 1)
print('z:', z)
print(z.size())

a = torch.cat([y, z], dim=1)
print(a.size())



rand_input = torch.randn(1, 1536, 7, 7)
model = Generator(in_channels=1536)
output = model(rand_input)
print(output.size())



def test(dataset, epoch):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = FullNetwork()

    criterion = None
    optimizer = None

    train_data = DataLoader(...)
    val_data = DataLoader(...)
    test_data = DataLoader(...)

    data = {'train': train_data, 'val': val_data, 'test': test_data}

    for epoch in range(num_epochs):  # loop over the dataset multiple times

        for phase in ['train', 'validation']:
            start_time = time.time()

            running_loss = 0.0

            if phase == 'train':
                model.train()
            else:
                model.eval()

            for inputs, targets in data[phase]:
                inputs = inputs.to(device)
                targets = targets.to(device)

                optimizer.zero_grad()

                if phase == 'train':
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    loss.backward()
                    optimizer.step()
                else:
                    with torch.no_grad():
                        outputs = model(inputs)
                    loss = criterion(outputs, targets)

                running_loss += loss.item() * inputs.size(0)

                print('Epoch {}/{}, {} Phase, Loss: {}'.format(epoch, num_epochs, phase, loss))
                end_time = time.time()
                print('Time: {}'.format(end_time - start_time))

                running_loss = 0.0



random_images_v1 = torch.randn(1, 3, 112, 112)
random_images_v2 = torch.randn(1, 3, 112, 112)
random_videos_v1 = torch.randn(1, 3, 8, 112, 112)
random_videos_v2 = torch.randn(1, 3, 8, 112, 112)

model = FullNetwork()

output1, output2 = model(random_images_v1, random_images_v2, random_videos_v1, random_videos_v2)

print(output1)
print(output2)
print(output1.size())
print(output2.size())



l = [1, 2, 3, 4, 5]
print(l)
print(*l)



with open('test.txt') as f:
    data_file = f.readlines()
    data_file = [line.strip() for line in data_file]

print(data_file)
print(len(data_file))



import glob

for filename in glob.iglob('C:/Users/Owner/Documents/*', recursive=True):
    print(filename)



import os
# import glob

# Class(1 - 60)
# Sample
# Views(1 2 3)
# Frames


# def listdir_nohidden(path):
#     return glob.glob(os.path.join(path, '*'))


# print(listdir_nohidden(path='C:/Users/Owner/Documents'))

dir = 'C:/Users/Owner/Documents'
x = os.listdir(path=dir)
classes = []
for y in x:
    if not y.startswith('~') and not y.endswith('.ini') and not y.startswith('M'):
        classes.append(os.path.join(dir, y))
print("CLASSES", classes)
samples = []
for c in classes:
    samples.append([os.path.join(c, s) for s in os.listdir(c)])
# samples = [os.listdir(os.path.join(dir, c)) for c in classes]
print("SAMPLES", samples)
views = [os.listdir(path=s) for s in samples]
print("VIEWS", views)




import torch

x = torch.zeros(10, 8, 3, 12, 12)
print(x.size()[0])

frame = torch.zeros(3, 12, 12)

imgs = torch.zeros(10, *frame.size())
print(imgs.size())




l = [0,1,2,3]

x, y, z = l[1:]
print(x, y, z)

a = () + (1,)
b = a + (2,)
print(a, b)




view1 = 3
view2 = 2

sample_info = '55/S001P001R002A055 76 77 75'.split(' ')
sample_id = sample_info[0]  # [sample_info.index('/'):]
# scene, pid, rid, action = PanopticDataset.decrypt_vid_name(sample_id)

nf_v1, nf_v2, nf_v3 = sample_info[1:]

info = (sample_id,)
if view1 == 1:
    info = info + (nf_v1,)
elif view1 == 2:
    info = info + (nf_v2,)
elif view1 == 3:
    info = info + (nf_v3,)

if view2 == 1:
    info = info + (nf_v1,)
elif view2 == 2:
    info = info + (nf_v2,)
elif view2 == 3:
    info = info + (nf_v3,)

print(info)




'''@staticmethod
    def get_data_paths(root_dir):
        # classes = os.listdir(path=dir)
        # print(classes)
        # samples = [os.listdir(path=c) for c in classes]
        # print(samples)
        # views = [os.listdir(path=s) for s in samples]
        # print(views)
        # return views
        x = os.listdir(path=root_dir)
        classes = []
        for y in x:
            if not y.startswith('~'):
                classes.append(os.path.join(root_dir, y))
        print("CLASSES", classes)
        samples = []
        for c in classes:
            samples.extend([os.path.join(c, s) for s in os.listdir(c)])
        print("SAMPLES", samples)
        views = []
        for s in samples:
            views.extend([os.path.join(v, s) for s in os.listdir(s)])
        return views'''




l = [0,1,2,3]

x, y, z = l[1:]
print(x, y, z)

a = () + (1,)
b = a + (2,)
print(a, b)

m = '2'.zfill(3)
print(m)





import torch

a = torch.tensor([[1, 2, 3], [4, 5, 6]])
b = torch.tensor([[1, 2, 6], [4, 5, 7]])

# a = np.array([[[1, 2], [1, 2]], [[1, 2], [1, 2]]])
# b = np.array([[[1, 4], [1, 4]], [[1, 4], [1, 4]]])

# print(a.size())
#
print(torch.eq(a, b))
#
# print(acc(a, b))






s = " hey i am kara"

t = s.split(" ")

print(t)

a = 52075235.239528592034

print("{0:.5f}".format(a))

b = [1, 2, 3, 4, 5]

print(b[2:-2])

import torch

x = torch.tensor([[1, 2], [1, 2]])
print(x.size()[0])

from random import sms

n = 5

sms.send("hello{}".format(n), "6304876751", "att")



import torch


def save_frames(vid_path, vid):
    channels, frames, height, width = vid.size()
    for i in range(frames):
        frame_name = make_frame_name(i+1)
        frame_path = os.path.join(vid_path, frame_name)
        print(frame_path)
        frame = vid[:, i, :, :].squeeze().numpy()
        # print(frame.size())
        # frame = frame.numpy()
        frame = np.transpose(frame, (1, 2, 0))
        print(frame.shape)

        try:
            cv2.imwrite(frame_path, frame)
        except:
            print('The image did not successfully save.')


def make_frame_name(frame_num):
    """
    Function to correctly generate the correctly formatted .jpg file name for the frame.
    :param frame_num: The frame number captured in the file.
    :return: str representing the file name.
    """
    return str(frame_num).zfill(3) + '.jpg'


if __name__ == '__main__':
    a = torch.randint(0, 255, (3, 8, 112, 112))
    # print(a)

    save_frames('C:/Users/Owner/Documents/UCF/Project/REU2019/test/', a)




import matplotlib.pyplot as plt


def single_plot(show_plot, x_data, y_data, title, x_label, y_label):
    plt.figure(1, (16, 8))  # something, plot size
    plt.subplot(111)
    for i in range(len(x_data)):
        plt.plot(x_data[i], y_data[i])
    plt.title(title)
    plt.xlabel(x_label, fontsize=12)
    plt.ylabel(y_label, fontsize=12)
    # plt.legend(['Train', 'Test'], loc='upper left')
    if show_plot:
        plt.show()

    return plt


if __name__ == '__main__':
    x_data = [range(1, 10), range(0, 10)]
    y_data = [range(1, 10), range(5, 15)]
    single_plot(True, x_data, y_data, "", "", "")





import numpy as np
import cv2

a = np.array([[156, 234, 124, ], [34, 32, 54]])

b = np.divide(a, 256)

i = cv2.imread('C:/Users/Owner/Documents/UCF/Project/ntu-ard/frames-240x135/18/S005P004R001A018/1/002.jpg')
i = np.array(i)
print(np.amin(i))
print(np.amax(i))
# print(i)

# print(b)




import cv2
import numpy as np
import os

frame_path = './cluster/REU2019/videos/418epochs/input/0_0_1_001.jpg'
    # './videos/60ish_epochs/input/0_0_1_001.jpg'

frame = cv2.imread(frame_path)
print(os.path.exists(frame_path))
# 'C:/Users/Owner/Documents/UCF/Project/REU2019/cluster/REU2019/videos/418epochs/inputs/000.jpg'
frame = np.array(frame).astype(np.float32)
print(frame)
print(frame.shape)


def normalize_frame(frame):
    frame = np.array(frame).astype(np.float32)
    return np.multiply(frame, 255.0)


frame = normalize_frame(frame)
print(frame)

cv2.imwrite('./img.jpg', frame)



from phase0.NTUDataLoader import NTUDataset
import torch

data_root_dir = 'C:/Users/Owner/Documents/UCF/Project/ntu-ard/frames-240x135/'
test_splits = 'C:/Users/Owner/Documents/UCF/Project/SSCVAS/data/shortestval.list'

# VIEW1 = 1
# VIEW2 = 2
BATCH_SIZE = 32
CHANNELS = 3
FRAMES = 8
SKIP_LEN = 2
HEIGHT = 112
WIDTH = 112
PRECROP = True


if __name__ == '__main__':
    testset = NTUDataset(root_dir=data_root_dir, data_file=test_splits,
                         resize_height=HEIGHT, resize_width=WIDTH,
                         clip_len=FRAMES, skip_len=SKIP_LEN,
                         random_all=True, precrop=PRECROP)
    testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=1)

    for batch_idx, (view1vid, view2vid) in enumerate(testloader):
        print(view1vid.size())
        print(view1vid)
        print(view2vid.size())
        print(view2vid)



# from PanopticDataLoader import PanopticDataset
from archive.phase2.data import NTUDataset
import torch

data_root_dir = 'C:/Users/Owner/Documents/UCF/Project/ntu-ard/frames-240x135'
test_splits = 'C:/Users/Owner/Documents/UCF/Project/REU2019/data/NTU/one.list'

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
    testset = NTUDataset(root_dir=data_root_dir, data_file=test_splits,
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



import numpy as np
import cv2
import os

root_dir = 'C:/Users/Owner/Documents/UCF/Project/panoptic/rgb_data/150303_celloScene1/samples/vga_01_01/1125_1250/'

frames = os.listdir(root_dir)
frames = [os.path.join(root_dir, frame) for frame in frames]
i = cv2.imread(frames[0])
i = np.array(i)
mini = np.amin(i)
maxi = np.amax(i)
for frame in frames:
    i = cv2.imread(frame)
    i = np.array(i)
    mini_ = np.amin(i)
    maxi_ = np.amax(i)
    if mini_ < mini:
        mini = mini_
    if maxi_ > maxi:
        maxi = maxi_

print(mini)
print(maxi)



import time
import torch
import torch.nn as nn
from phase0.network import FullNetwork
from phase0.NTUDataLoader import NTUDataset
import torch.backends.cudnn as cudnn
import cv2

# directory information
data_root_dir = '/home/c2-2/yogesh/datasets/ntu-ard/frames-240x135/'
# train_splits = '/home/yogesh/kara/data/train.list'
test_splits = '/home/yogesh/kara/data/val.list'
weights_path = '/home/yogesh/kara/REU2019/weights/test_net_32_8_2_True_1000_0001.pt'

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
# LR = 1e-4

# weight_file_name = './weights/net_{}_{}_{}_{}'.format(BATCH_SIZE, FRAMES, NUM_EPOCHS, LR)
output_video_dir = '/home/yogesh/kara/REU2019/videos/'


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
            convert_to_vid(view1vid, batch_idx, True), convert_to_vid(view2vid, batch_idx, True)
            convert_to_vid(output_v1, batch_idx, False), convert_to_vid(output_v1, batch_idx, False)

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


def convert_to_vid(tensor, batch_num, input):  # whether it was an input or output
    bsz, channels, frames, height, width = tensor.size()
    for i in range(bsz):
        vid = tensor[i]
        if input:
            vid_path = os.path.join(output_video_dir, 'input', str(batch_num), str(i))
        else:
            vid_path = os.path.join(output_video_dir, 'output', str(batch_num), str(i))
        save_frames(vid_path, vid)


def save_frames(vid_path, vid):
    channels, frames, height, width = vid.size()
    for i in range(frames):
        frame_name = make_frame_name(i + 1)
        frame_path = os.path.join(vid_path, frame_name)
        # extract one frame as np array
        frame = vid[:, i, :, :].squeeze().cpu().numpy()
        # pytorch tensor is (channels, height, width)
        # np is (height, width, channels)
        frame = np.transpose(frame, (1, 2, 0))

        try:
            cv2.imwrite(frame_path, frame)
        except:
            print('The image did not successfully save.')


def make_frame_name(frame_num):
    """
    Function to correctly generate the correctly formatted .jpg file name for the frame.
    :param frame_num: The frame number captured in the file.
    :return: str representing the file name.
    """
    return str(frame_num).zfill(3) + '.jpg'


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




import torch

a = torch.randn((3, 512, 7, 7))

print(a.numel())




import torch

# print(5 * torch.ones(2, 2))
#
# app = torch.ones(1, 2, 7, 7)

# bsz, channels, height, width = app.size()
# buffer = torch.zeros(bsz, channels, 8, height, width)
# print(app.size())
# print(buffer.size())
# for frame in range(8):
#     buffer[:, :, frame, :, :] = app
#
# print(buffer)
# print(buffer.size())

print(torch.tensor([5]) - torch.tensor([2]))




import torch

i3d_weights_path = 'C:/Users/Owner/Documents/UCF/Project/REU2019/weights/rgb_charades.pt'

x = torch.load(i3d_weights_path)

print(x.keys())





import torch
import torch.nn as nn

split = '/home/c2-2/yogesh/datasets/panoptic/rgb_data/160906_ian2/samples/vga_11_21/7500_7625'.split('/')

print(split)

x = torch.zeros(32, 4, 16, 28, 28)

x.cpu()

x = nn.functional.interpolate(x, size=(16, 56, 56), mode='nearest')

print(x.size())





import torch

vp = torch.ones(32, 3)

bsz = vp.size()[0]

buffer = torch.zeros(bsz, 3, 16, 28, 28)
for c in range(16):
    for d in range(28):
        for e in range(28):
            buffer[:, :, c, d, e] = vp

print(buffer)







import torch
import _pickle as pickle
import os

x = torch.ones(2, 2)
print(-x)

y = torch.tensor([[1, 0], [0, 1]], dtype=torch.float32)

print(x - y)

x = torch.ones(32)
print(x.size())

x = torch.unsqueeze(x, dim=1)
print(x.size())

cal_file = 'C:/Users/Owner/Documents/UCF/Project/panoptic/rgb_data/151125_mafia/calibration_151125_mafia.pkl'
view_id = 'vga_10_08'

with open(cal_file, 'rb') as fp:
    cal = pickle.load(fp)

print(cal.keys())

# c_data = cal[view_id[4:]]
#
# print(cal)
# print(c_data)

print(os.path.exists(''))







from archive.phase2.data.PanopticDataLoader import PanopticDataset
import torch

data_root_dir = 'C:/Users/Owner/Documents/UCF/Project/panoptic/rgb_data/'
test_splits = 'C:/Users/Owner/Documents/UCF/Project/REU2019/data/panoptic/one.list'

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


from data.panoptic.PanopticDataLoader import PanopticDataset
import torch

data_root_dir = 'C:/Users/Owner/Documents/UCF/panoptic/rgb_data/'
test_splits = 'C:/Users/Owner/Documents/UCF/REU2019/data/panoptic/one.list'
close_cams_file = 'C:/Users/Owner/Documents/UCF/REU2019/data/panoptic/closecams.list'


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
                              random_all=True, close_views=True, close_cams_file=close_cams_file,
                              precrop=PRECROP)
    testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=1)

    for batch_idx, (vp1, vp2, view1vid, view2vid) in enumerate(testloader):
        print(view1vid.size())
        # print(view1vid)
        print(view2vid.size())
        # print(view2vid)
        print(vp1)
        print(vp2)
        print(vp1.size())



import torch
import torch.nn as nn

shrink = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

x = torch.zeros(20,3,16,112,112)

y = shrink(x)

print(y.size())








import numpy as np
import torch

a = [1, 2, 3]
b = [0, 1, 2]

a = np.array(a)
b = np.array(b)

print(b - a)
print(np.sum(np.abs(b - a)))

a = torch.randn(2, 2, 2, 2)
a = torch.unsqueeze(a, dim=2)
print(a.size())




import torch

weights_path = 'C:/Users/Owner/Documents/UCF/Project/REU2019/weights/vgg16-397923af.pth'

state_dict = torch.load(weights_path)

for key, value in state_dict.items():
    print(key)
    first_per = key.index('.')
    second_per = key[first_per + 1:].index('.')
    id = key[:first_per + second_per + 1]
    print(id)

import torch
import torch.nn as nn
import torch.nn.functional as f

if __name__ == "__main__":
    x = torch.randint(1, 3, (2, 2, 2, 2, 2))
    x = torch.tensor(x, dtype=torch.float32)
    x = torch.tensor([[[[[1., 2.], [2., 2.]],
                        [[1., 2.], [2., 2.]]],
                       [[[1., 2.], [2., 2.]],
                        [[1., 2.], [2., 2.]]]],
                      [[[[1., 2.], [2., 2.]],
                        [[1., 2.], [2., 2.]]],
                       [[[1., 2.], [2., 2.]],
                        [[1., 2.], [2., 2.]]]]])
    print(x)
    x = f.interpolate(x, size=(4, 4, 4), mode='trilinear', align_corners=False)
    print(x)

    x = torch.ones(2, 3, 4, 8, 8)
    print(torch.sum(x))
    bsz, channels, frames, height, width = x.size()
    x = torch.reshape(x, (bsz, channels, frames, height * width))
    print(x.size())
    x = nn.Softmax(dim=3)(x)
    print(x.size())
    x = torch.reshape(x, (bsz, channels, frames, height, width))
    print(x.size())
    for i in range(height * width):
        print(x[:, :, :, i].size())
        print(torch.sum(torch.squeeze(x[i, i, i, :])))

    from archive.phase2.data.outputConversion import convert_to_vid

    x = torch.randn(2, 6, 8, 100, 100)

    convert_to_vid(tensor=x, output_dir='./testOutput', batch_num=0, view=0, item_type='random')



 samples = get_samples(data_root_dir)
    vga_list = get_vga_list(panels, nodes)
    for sample in samples:
        cal_file = os.path.join(data_root_dir, sample, 'calibration_' + sample + '.pkl')

        # load the calibration file
        with open(cal_file, 'rb') as fp:
            cal = pickle.load(fp)

        cameras = list(cal.keys())
        for vga in vga_list:
            # print(vga)
            cameras.remove(vga)
        print(sample)
        print(cameras)
        print(len(cameras))





my_list = [('a', 0), ('z', 25), ('h', 7), ('c', 2)]
my_list.sort(key=lambda x: x[1])

print(my_list)

print(str(my_list))

cam_file = 'C:/Users/Owner/Documents/UCF/REU2019/data/panoptic/closecams.list'

close_cams_dict = {}
with open(cam_file, 'r') as f:
    for line in f:
        cams = line.strip().split(' ')
        close_cams_dict[cams[0]] = cams[1:]

print(close_cams_dict)













