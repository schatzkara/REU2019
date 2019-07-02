# phase 2

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from network import FullNetwork
from NTUDataLoader import NTUDataset
from PanopticDataLoader import PanopticDataset
import torch.backends.cudnn as cudnn
import sms

DATASET = 'panoptic'  # 'NTU' or 'panoptic'

# data parameters
BATCH_SIZE = 20
CHANNELS = 3
FRAMES = 16
SKIP_LEN = 2
HEIGHT = 112
WIDTH = 112

# training parameters
NUM_EPOCHS = 1000
LR = 1e-4


def ntu_config():
    # NTU directory information
    data_root_dir = '/home/c2-2/yogesh/datasets/ntu-ard/frames-240x135/'
    if FRAMES * SKIP_LEN >= 32:
        train_split = '/home/yogesh/kara/data/train16.list'
        test_split = '/home/yogesh/kara/data/val16.list'
    else:
        train_split = '/home/yogesh/kara/data/train.list'
        test_split = '/home/yogesh/kara/data/val.list'
    param_file = '/home/yogesh/kara/data/view.params'
    if not os.path.exists('./weights'):
        os.mkdir('./weights')
    weight_file = './weights/net2_ntu2_{}_{}_{}_{}_{}_{}.pt'.format(BATCH_SIZE, FRAMES, SKIP_LEN,
                                                                    PRECROP, NUM_EPOCHS, LR)
    return data_root_dir, train_split, test_split, param_file, weight_file


def panoptic_config():
    # panoptic directory information
    data_root_dir = '/home/c2-2/yogesh/datasets/panoptic/rgb_data/'
    train_split = '/home/yogesh/kara/data/panoptic/mod_train.list'
    test_split = '/home/yogesh/kara/data/panoptic/mod_test.list'
    if not os.path.exists('./weights'):
        os.mkdir('./weights')
    weight_file = './weights/net2_pan2_{}_{}_{}_{}_{}_{}.pt'.format(BATCH_SIZE, FRAMES, SKIP_LEN,
                                                                    PRECROP, NUM_EPOCHS, LR)
    return data_root_dir, train_split, test_split, weight_file


def training_loop(epoch):
    """
    Function carrying out the training loop for the Full Network for a single epoch.
    :param epoch: (int) The current epoch in which the model is training.
    :return: None
    """
    running_total_loss = 0.0
    running_con1_loss = 0.0
    running_con2_loss = 0.0
    running_recon1_loss = 0.0
    running_recon2_loss = 0.0

    model.train()

    for batch_idx, (vp_diff, vid1, vid2) in enumerate(trainloader):
        vp_diff = vp_diff.to(device)
        vid1, vid2 = vid1.to(device), vid2.to(device)
        img1, img2 = get_first_frame(vid1), get_first_frame(vid2)
        img1, img2 = img1.to(device), img2.to(device)

        optimizer.zero_grad()

        output_vid1, output_vid2, kp_v1, kp_v2, kp_v1_est, kp_v2_est = model(vp_diff=vp_diff,
                                                                             vid1=vid1, vid2=vid2,
                                                                             img1=img1, img2=img2)
        # loss
        con1_loss = criterion(kp_v1, kp_v1_est)
        con2_loss = criterion(kp_v2, kp_v2_est)
        recon1_loss = criterion(output_vid1, vid1)
        recon2_loss = criterion(output_vid2, vid2)
        loss = con1_loss + con2_loss + recon1_loss + recon2_loss
        loss.backward()
        optimizer.step()

        running_total_loss += loss.item()
        running_con1_loss += con1_loss.item()
        running_con2_loss += con2_loss.item()
        running_recon1_loss += recon1_loss.item()
        running_recon2_loss += recon2_loss.item()
        if (batch_idx + 1) % 10 == 0:
            print('\tBatch {}/{} Loss:{} con1:{} con2:{} recon1:{} recon2:{}'.format(batch_idx + 1, len(trainloader),
                                                                                     "{0:.5f}".format(loss),
                                                                                     "{0:.5f}".format(con1_loss),
                                                                                     "{0:.5f}".format(con2_loss),
                                                                                     "{0:.5f}".format(recon1_loss),
                                                                                     "{0:.5f}".format(recon2_loss)))

    print('Training Epoch {}/{} Loss:{} con1:{} con2:{} recon1:{} recon2:{}'.format(epoch + 1, NUM_EPOCHS,
                                                                                    "{0:.5f}".format(
                                                                                        (running_total_loss / len(
                                                                                            trainloader))),
                                                                                    "{0:.5f}".format(
                                                                                        (running_con1_loss / len(
                                                                                            trainloader))),
                                                                                    "{0:.5f}".format(
                                                                                        (running_con2_loss / len(
                                                                                            trainloader))),
                                                                                    "{0:.5f}".format(
                                                                                        (running_recon1_loss / len(
                                                                                            trainloader))),
                                                                                    "{0:.5f}".format(
                                                                                        (running_recon2_loss / len(
                                                                                            trainloader)))))


def testing_loop(epoch):
    """
    Function to carry out the testing/validation loop for the Full Network for a single epoch.
    :param epoch: (int) The current epoch in which the model is testing/validating.
    :return: None
    """
    running_total_loss = 0.0
    running_con1_loss = 0.0
    running_con2_loss = 0.0
    running_recon1_loss = 0.0
    running_recon2_loss = 0.0

    model.eval()

    for batch_idx, (vp_diff, vid1, vid2) in enumerate(testloader):
        vp_diff = vp_diff.to(device)
        vid1, vid2 = vid1.to(device), vid2.to(device)
        img1, img2 = get_first_frame(vid1), get_first_frame(vid2)
        img1, img2 = img1.to(device), img2.to(device)

        with torch.no_grad():
            output_vid1, output_vid2, kp_v1, kp_v2, kp_v1_est, kp_v2_est = model(vp_diff=vp_diff,
                                                                                 vid1=vid1, vid2=vid2,
                                                                                 img1=img1, img2=img2)
            # loss
            con1_loss = criterion(kp_v1, kp_v1_est)
            con2_loss = criterion(kp_v2, kp_v2_est)
            recon1_loss = criterion(output_vid1, vid1)
            recon2_loss = criterion(output_vid2, vid2)
            loss = con1_loss + con2_loss + recon1_loss + recon2_loss

        running_total_loss += loss.item()
        running_con1_loss += con1_loss.item()
        running_con2_loss += con2_loss.item()
        running_recon1_loss += recon1_loss.item()
        running_recon2_loss += recon2_loss.item()
        if (batch_idx + 1) % 10 == 0:
            print('\tBatch {}/{} Loss:{} con1:{} con2: {} recon1:{} recon2:{}'.format(batch_idx + 1, len(testloader),
                                                                                      "{0:.5f}".format(loss),
                                                                                      "{0:.5f}".format(con1_loss),
                                                                                      "{0:.5f}".format(con2_loss),
                                                                                      "{0:.5f}".format(recon1_loss),
                                                                                      "{0:.5f}".format(recon2_loss)))

    print('Validation Epoch {}/{} Loss:{} con1:{} con2:{} recon1:{} recon2:{}'.format(epoch + 1, NUM_EPOCHS,
                                                                                      "{0:.5f}".format(
                                                                                          (running_total_loss / len(
                                                                                              testloader))),
                                                                                      "{0:.5f}".format(
                                                                                          (running_con1_loss / len(
                                                                                              testloader))),
                                                                                      "{0:.5f}".format(
                                                                                          (running_con2_loss / len(
                                                                                              testloader))),
                                                                                      "{0:.5f}".format((
                                                                                              running_recon1_loss / len(
                                                                                          testloader))),
                                                                                      "{0:.5f}".format((
                                                                                              running_recon2_loss / len(
                                                                                          testloader)))))

    return running_total_loss


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


def train_model():
    """
    Function to train and validate the model for all epochs.
    :return: None
    """
    min_loss = 0.0
    start_time = time.time()
    for epoch in range(NUM_EPOCHS):
        print('Training...')
        training_loop(epoch)
        print('Validation...')
        loss = testing_loop(epoch)
        sms.send('Epoch {} Loss: {}'.format(epoch + 1, loss), "6304876751", "att")
        if epoch == 0 or loss < min_loss:
            min_loss = loss
            torch.save(model.state_dict(), weight_file)
            sms.send('Weights saved', "6304876751", "att")
    end_time = time.time()
    print('Time: {}'.format(end_time - start_time))


def print_params():
    """
    Function to print out all the custom parameter information for the experiment.
    :return: None
    """
    print('Parameters for training on {}'.format(DATASET))
    print('Batch Size: {}'.format(BATCH_SIZE))
    print('Tensor Size: ({},{},{},{})'.format(CHANNELS, FRAMES, HEIGHT, WIDTH))
    print('Skip Length: {}'.format(SKIP_LEN))
    print('Precrop: {}'.format(PRECROP))
    print('Total Epochs: {}'.format(NUM_EPOCHS))
    print('Learning Rate: {}'.format(LR))


if __name__ == '__main__':
    """
    Main function to carry out the training loop. 
    This function creates the model and data loaders. Then, it trains the model.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    RANDOM_ALL = True
    PRECROP = True if DATASET.lower() == 'ntu' else False

    if DATASET.lower() == 'ntu':
        data_root_dir, train_split, test_split, param_file, weight_file = ntu_config()

        # model
        model = FullNetwork(vp_value_count=1, output_shape=(BATCH_SIZE, CHANNELS, FRAMES, HEIGHT, WIDTH))
        model = model.to(device)

        if device == 'cuda':
            net = torch.nn.DataParallel(model)
            cudnn.benchmark = True

        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=LR)

        # data
        trainset = NTUDataset(root_dir=data_root_dir, data_file=train_split, param_file=param_file,
                              resize_height=HEIGHT, resize_width=WIDTH,
                              clip_len=FRAMES, skip_len=SKIP_LEN,
                              random_all=RANDOM_ALL, precrop=PRECROP)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

        testset = NTUDataset(root_dir=data_root_dir, data_file=test_split, param_file=param_file,
                             resize_height=HEIGHT, resize_width=WIDTH,
                             clip_len=FRAMES, skip_len=SKIP_LEN,
                             random_all=RANDOM_ALL, precrop=PRECROP)
        testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    elif DATASET.lower() == 'panoptic':
        data_root_dir, train_split, test_split, weight_file = panoptic_config()

        # model
        model = FullNetwork(vp_value_count=3, output_shape=(BATCH_SIZE, CHANNELS, FRAMES, HEIGHT, WIDTH))
        model = model.to(device)

        if device == 'cuda':
            net = torch.nn.DataParallel(model)
            cudnn.benchmark = True

        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=LR)

        # data
        trainset = PanopticDataset(root_dir=data_root_dir, data_file=train_split,
                                   resize_height=HEIGHT, resize_width=WIDTH,
                                   clip_len=FRAMES, skip_len=SKIP_LEN,
                                   random_all=RANDOM_ALL, precrop=PRECROP)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

        testset = PanopticDataset(root_dir=data_root_dir, data_file=test_split,
                                  resize_height=HEIGHT, resize_width=WIDTH,
                                  clip_len=FRAMES, skip_len=SKIP_LEN,
                                  random_all=RANDOM_ALL, precrop=PRECROP)
        testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    else:
        print('This network has only been set up to train on the NTU and panoptic datasets.')

    print_params()
    print(model)
    train_model()
