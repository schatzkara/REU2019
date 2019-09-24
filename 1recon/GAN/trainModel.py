import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from .networks.model import FullNetwork
from .networks.discriminator import Discriminator
from data.NTUDataLoader import NTUDataset
from data.PanopticDataLoader import PanopticDataset
import torch.backends.cudnn as cudnn
from utils.modelIOFuncs import get_first_frame
from utils import sms

DATASET = 'NTU'  # 'NTU' or 'Panoptic'

# data parameters
BATCH_SIZE = 14
CHANNELS = 3
FRAMES = 16
SKIP_LEN = 2
HEIGHT = 112
WIDTH = 112

# training parameters
NUM_EPOCHS = 1000
LR = 1e-4
STDEV = 0.1

pretrained = False
MIN_LOSS = 1.0
# if DATASET.lower() == 'ntu':
#     pretrained_weights = './weights/net_ntu_16_16_2_True_1000_0.0001.pt'
# else:
#     pretrained_weights = './weights/net_pan_16_16_2_False_1000_0.0001.pt'
pretrained_epochs = 0


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
    gen_weight_file = './weights/net_gen_ntu_{}_{}_{}_{}_{}_{}_{}.pt'.format(BATCH_SIZE, FRAMES, SKIP_LEN,
                                                                     PRECROP, NUM_EPOCHS, LR, STDEV)
    disc_weight_file = './weights/net_disc_ntu_{}_{}_{}_{}_{}_{}_{}.pt'.format(BATCH_SIZE, FRAMES, SKIP_LEN,
                                                                     PRECROP, NUM_EPOCHS, LR, STDEV)
    return data_root_dir, train_split, test_split, param_file, gen_weight_file, disc_weight_file


def panoptic_config():
    # panoptic directory information
    data_root_dir = '/home/c2-2/yogesh/datasets/panoptic/rgb_data/'
    train_split = '/home/yogesh/kara/data/panoptic/mod_train.list'
    test_split = '/home/yogesh/kara/data/panoptic/mod_test.list'
    close_cams_file = '/home/yogesh/kara/data/panoptic/closecams.list'
    if not os.path.exists('./weights'):
        os.mkdir('./weights')
    gen_weight_file = './weights/net_gen_pan_{}_{}_{}_{}_{}_{}_{}.pt'.format(BATCH_SIZE, FRAMES, SKIP_LEN,
                                                                             PRECROP, NUM_EPOCHS, LR, STDEV)
    disc_weight_file = './weights/net_disc_pan_{}_{}_{}_{}_{}_{}_{}.pt'.format(BATCH_SIZE, FRAMES, SKIP_LEN,
                                                                               PRECROP, NUM_EPOCHS, LR, STDEV)
    return data_root_dir, train_split, test_split, close_cams_file, gen_weight_file, disc_weight_file


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
    print('Close Views: {}'.format(CLOSE_VIEWS))
    print('Total Epochs: {}'.format(NUM_EPOCHS))
    print('Learning Rate: {}'.format(LR))


# Loss functions
# adversarial_loss = torch.nn.MSELoss()
# categorical_loss = torch.nn.CrossEntropyLoss()
# continuous_loss = torch.nn.MSELoss()

# Loss weights
# lambda_cat = 1
# lambda_con = 0.1

# Initialize generator and discriminator
# generator = FullNetwork()
# discriminator = Discriminator()

# if cuda:
#     generator.cuda()
#     discriminator.cuda()
#     adversarial_loss.cuda()
#     categorical_loss.cuda()
#     continuous_loss.cuda()

# Initialize weights
# generator.apply(weights_init_normal)
# discriminator.apply(weights_init_normal)

# Configure data loader
# os.makedirs("../../data/mnist", exist_ok=True)
# dataloader = torch.utils.data.DataLoader(
#     datasets.MNIST(
#         "../../data/mnist",
#         train=True,
#         download=True,
#         transform=transforms.Compose(
#             [transforms.Resize(opt.img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
#         ),
#     ),
#     batch_size=opt.batch_size,
#     shuffle=True,
# )

# Optimizers
# optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
# optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
# optimizer_info = torch.optim.Adam(
#     itertools.chain(generator.parameters(), discriminator.parameters()), lr=opt.lr, betas=(opt.b1, opt.b2)
# )

# FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
# LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

# Static generator inputs for sampling
# static_z = Variable(FloatTensor(np.zeros((opt.n_classes ** 2, opt.latent_dim))))
# static_label = to_categorical(
#     np.array([num for _ in range(opt.n_classes) for num in range(opt.n_classes)]), num_columns=opt.n_classes
# )
# static_code = Variable(FloatTensor(np.zeros((opt.n_classes ** 2, opt.code_dim))))


# def sample_image(n_row, batches_done):
#     """Saves a grid of generated digits ranging from 0 to n_classes"""
#     Static sample
# z = Variable(FloatTensor(np.random.normal(0, 1, (n_row ** 2, opt.latent_dim))))
# static_sample = generator(z, static_label, static_code)
# save_image(static_sample.data, "images/static/%d.png" % batches_done, nrow=n_row, normalize=True)
#
# Get varied c1 and c2
# zeros = np.zeros((n_row ** 2, 1))
# c_varied = np.repeat(np.linspace(-1, 1, n_row)[:, np.newaxis], n_row, 0)
# c1 = Variable(FloatTensor(np.concatenate((c_varied, zeros), -1)))
# c2 = Variable(FloatTensor(np.concatenate((zeros, c_varied), -1)))
# sample1 = generator(static_z, static_label, c1)
# sample2 = generator(static_z, static_label, c2)
# save_image(sample1.data, "images/varying_c1/%d.png" % batches_done, nrow=n_row, normalize=True)
# save_image(sample2.data, "images/varying_c2/%d.png" % batches_done, nrow=n_row, normalize=True)


# ----------
#  Training
# ----------
def train_model(starting_epoch):
    min_loss = 0.0
    for epoch in range(starting_epoch, NUM_EPOCHS):  # opt.n_epochs):
        running_g_loss = 0.0
        running_d_loss = 0.0
        for batch_idx, (vp_diff, vid1, vid2) in enumerate(trainloader):
            vp_diff = vp_diff.type(torch.FloatTensor).to(device)
            vid1, vid2 = vid1.to(device), vid2.to(device)
            img1, img2 = get_first_frame(vid1), get_first_frame(vid2)
            img1, img2 = img1.to(device), img2.to(device)

            batch_size = vp_diff.shape[0]

            # Adversarial ground truths
            valid = Variable(FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False)
            fake = Variable(FloatTensor(batch_size, 1).fill_(0.0), requires_grad=False)

            # Configure input
            real_vids_v1, real_vids_v2 = Variable(vid1.type(FloatTensor)), Variable(vid2.type(FloatTensor))
            # labels = to_categorical(labels.numpy(), num_columns=opt.n_classes)

            # -----------------
            #  Train Generator
            # -----------------
            optimizer_G.zero_grad()

            # Sample noise and labels as generator input
            # z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, opt.latent_dim))))
            # label_input = to_categorical(np.random.randint(0, opt.n_classes, batch_size), num_columns=opt.n_classes)
            # code_input = Variable(FloatTensor(np.random.uniform(-1, 1, (batch_size, opt.code_dim))))

            # Generate a batch of images
            # gen_imgs = generator(z, label_input, code_input)
            gen_v2, vp_est = generator(vp_diff=vp_diff, vid1=vid1, img2=img2)

            # Loss measures generator's ability to fool the discriminator
            validity = discriminator(gen_v2)
            g_loss = adversarial_loss(validity, valid)

            g_loss.backward()
            optimizer_G.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------

            optimizer_D.zero_grad()

            # Loss for real images
            real_pred = discriminator(real_vids_v2)
            d_real_loss = adversarial_loss(real_pred, valid)

            # Loss for fake images
            fake_pred = discriminator(gen_v2.detach())
            d_fake_loss = adversarial_loss(fake_pred, fake)

            # Total discriminator loss
            d_loss = (d_real_loss + d_fake_loss) / 2

            d_loss.backward()
            optimizer_D.step()

            # ------------------
            # Information Loss
            # ------------------

            # optimizer_info.zero_grad()
            #
            # # Sample labels
            # sampled_labels = np.random.randint(0, opt.n_classes, batch_size)
            #
            # # Ground truth labels
            # gt_labels = Variable(LongTensor(sampled_labels), requires_grad=False)
            #
            # # Sample noise, labels and code as generator input
            # z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, opt.latent_dim))))
            # label_input = to_categorical(sampled_labels, num_columns=opt.n_classes)
            # code_input = Variable(FloatTensor(np.random.uniform(-1, 1, (batch_size, opt.code_dim))))
            #
            # gen_imgs = generator(z, label_input, code_input)
            # _, pred_label, pred_code = discriminator(gen_imgs)
            #
            # info_loss = lambda_cat * categorical_loss(pred_label, gt_labels) + lambda_con * continuous_loss(
            #     pred_code, code_input
            # )
            #
            # info_loss.backward()
            # optimizer_info.step()

            # --------------
            # Log Progress
            # --------------

            # print(
            #     "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] [info loss: %f]"
            #     % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item(), info_loss.item())
            # )
            # batches_done = epoch * len(dataloader) + i
            # if batches_done % opt.sample_interval == 0:
            #     sample_image(n_row=10, batches_done=batches_done)

            running_g_loss += g_loss.item()
            running_d_loss += d_loss.item()
            if (batch_idx + 1) % 10 == 0:
                print('\tBatch {}/{} GLoss:{} DLoss:{}'.format(
                    batch_idx + 1,
                    len(trainloader),
                    "{0:.5f}".format(g_loss),
                    "{0:.5f}".format(d_loss)))

        print('Training Epoch {}/{} GLoss:{} DLoss:{}'.format(
            epoch + 1,
            NUM_EPOCHS,
            "{0:.5f}".format((running_g_loss / len(trainloader))),
            "{0:.5f}".format((running_d_loss / len(trainloader)))))
        if running_g_loss + running_d_loss < min_loss or epoch == 0:
            min_loss = running_g_loss + running_d_loss
            torch.save(generator.state_dict(), gen_weight_file)
            torch.save(discriminator.state_dict(), disc_weight_file)


if __name__ == '__main__':
    """
    Main function to carry out the training loop. 
    This function creates the generator and data loaders. Then, it trains the generator.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    RANDOM_ALL = True
    PRECROP = True if DATASET.lower() == 'ntu' else False
    VP_VALUE_COUNT = 1 if DATASET.lower() == 'ntu' else 3
    CLOSE_VIEWS = True if DATASET.lower() == 'panoptic' else False

    # generator
    generator = FullNetwork(vp_value_count=VP_VALUE_COUNT, stdev=STDEV,
                            output_shape=(BATCH_SIZE, CHANNELS, FRAMES, HEIGHT, WIDTH))
    generator = generator.to(device)
    # discriminator
    discriminator = Discriminator(in_channels=3)
    discriminator = discriminator.to(device)

    if device == 'cuda':
        net = torch.nn.DataParallel(generator)
        cudnn.benchmark = True

    # Loss functions
    criterion = nn.MSELoss()
    adversarial_loss = nn.MSELoss()
    # categorical_loss = torch.nn.CrossEntropyLoss()
    # continuous_loss = torch.nn.MSELoss()

    optimizer_G = optim.Adam(generator.parameters(), lr=LR)
    optimizer_D = optim.Adam(discriminator.parameters(), lr=LR)

    if DATASET.lower() == 'ntu':
        data_root_dir, train_split, test_split, param_file, gen_weight_file, disc_weight_file = ntu_config()

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
        data_root_dir, train_split, test_split, close_cams_file, gen_weight_file, disc_weight_file = panoptic_config()

        # data
        trainset = PanopticDataset(root_dir=data_root_dir, data_file=train_split,
                                   resize_height=HEIGHT, resize_width=WIDTH,
                                   clip_len=FRAMES, skip_len=SKIP_LEN,
                                   random_all=RANDOM_ALL, close_views=CLOSE_VIEWS,
                                   close_cams_file=close_cams_file, precrop=PRECROP)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

        testset = PanopticDataset(root_dir=data_root_dir, data_file=test_split,
                                  resize_height=HEIGHT, resize_width=WIDTH,
                                  clip_len=FRAMES, skip_len=SKIP_LEN,
                                  random_all=RANDOM_ALL, close_views=CLOSE_VIEWS,
                                  close_cams_file=close_cams_file, precrop=PRECROP)
        testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    else:
        print('This network has only been set up to train on the NTU and panoptic datasets.')

    FloatTensor = torch.cuda.FloatTensor if device == 'cuda' else torch.FloatTensor
    LongTensor = torch.cuda.LongTensor if device == 'cuda' else torch.LongTensor

    print_params()
    print(generator)
    if pretrained:
        starting_epoch = pretrained_epochs
    else:
        starting_epoch = 0
    train_model(starting_epoch)
