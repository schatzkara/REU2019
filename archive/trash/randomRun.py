import time
import torch
import torch.nn as nn
import torch.optim as optim
from old.phase0 import FullNetwork

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# model
model = FullNetwork()
model.to(device)

# parameters
PROP_TRAINING = 0.5
DATASET_SIZE = 2
TRAINING_SIZE = int(PROP_TRAINING * DATASET_SIZE)
TESTING_SIZE = DATASET_SIZE - TRAINING_SIZE
BATCH_SIZE = 1
NUM_BATCHES = int(PROP_TRAINING * DATASET_SIZE) // BATCH_SIZE
CHANNELS = 3
FRAMES = 8
HEIGHT = 112
WIDTH = 112

num_epochs = 2
lr = 1e-4
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=lr)  # other parameters???


# generate randomStuff data
def generate_random_data():
    view1vids = torch.randn(DATASET_SIZE, CHANNELS, FRAMES, HEIGHT, WIDTH)
    view2vids = torch.randn(DATASET_SIZE, CHANNELS, FRAMES, HEIGHT, WIDTH)

    train_test_cutoff = int(PROP_TRAINING * DATASET_SIZE)

    training_data = {'view1': view1vids[:train_test_cutoff], 'view2': view2vids[:train_test_cutoff]}
    testing_data = {'view1': view1vids[train_test_cutoff:], 'view2': view2vids[train_test_cutoff:]}

    return training_data, testing_data


def create_batches(training_data):
    train_view1 = training_data['view1']
    train_view2 = training_data['view2']
    training_batches = {}
    # training_batches1 = {}
    # training_batches2 = {}
    # for sample in train_view1:
    for i in range(NUM_BATCHES):
        s1 = train_view1[i * BATCH_SIZE: i * BATCH_SIZE + BATCH_SIZE]
        s2 = train_view2[i * BATCH_SIZE: i * BATCH_SIZE + BATCH_SIZE]
        training_batches[i] = {'view1': s1, 'view2': s2}

    # training_batches = {'view1': training_batches1, 'view2': training_batches2}
    return training_batches


# method to train for one epoch
def train(dataset, epoch):  # dataset is dict of nbatches, each batch is dict of 2 view with bsz tensors each

    # start_time = time.time()
    running_loss = 0.0

    model.train()

    for batch in range(NUM_BATCHES):
        # inputs
        view1vid = dataset[batch]['view1']
        view2vid = dataset[batch]['view2']
        imgs1 = [torch.squeeze(vid[:, :1, :, :]) for vid in view1vid]
        imgs2 = [torch.squeeze(vid[:, :1, :, :]) for vid in view2vid]
        # print(imgs1[0].size())
        view1img = torch.zeros(BATCH_SIZE, CHANNELS, HEIGHT, WIDTH)
        view2img = torch.zeros(BATCH_SIZE, CHANNELS, HEIGHT, WIDTH)
        for sample in range(BATCH_SIZE):
            view1img[sample] = imgs1[sample]
            view2img[sample] = imgs2[sample]
        # print(view1imgs)
        # print(view1imgs.size())

        view1img, view2img, view1vid, view2vid = view1img.to(device), view2img.to(device), view1vid.to(device), view2vid.to(device)

        optimizer.zero_grad()

        print('Running inputs through model.')
        output_v1, output_v2, rep_v1, rep_v2 = model(view1img, view2img, view1vid, view2vid)
        con_loss = criterion(rep_v1, rep_v2)
        recon_loss1 = criterion(output_v1, view1vid)
        recon_loss2 = criterion(output_v2, view2vid)
        loss = con_loss + recon_loss1 + recon_loss2
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print('Epoch {}/{} Loss: {}'.format(epoch, num_epochs, loss))
    # end_time = time.time()
    # print('Time: {}'.format(end_time - start_time))
    #
    # running_loss = 0.0


def test(dataset, epoch):  # dataset is dict of 2 views with (1-PROP_TRAINING)*DATASET_SIZE tensors each

    running_loss = 0.0

    model.eval()

    # inputs
    view1vid = dataset['view1']
    # print(view1vid.size())
    view2vid = dataset['view2']
    imgs1 = [torch.squeeze(vid[:, :1, :, :]) for vid in view1vid]
    imgs2 = [torch.squeeze(vid[:, :1, :, :]) for vid in view2vid]
    # print(imgs1[0].size())
    view1img = torch.zeros(BATCH_SIZE, CHANNELS, HEIGHT, WIDTH)
    view2img = torch.zeros(BATCH_SIZE, CHANNELS, HEIGHT, WIDTH)
    for sample in range(TESTING_SIZE):
        view1img[sample] = imgs1[sample]
        view2img[sample] = imgs2[sample]
    # print(view1imgs)
    # print(view1img.size())

    view1img, view2img, view1vid, view2vid = view1img.to(device), view2img.to(device), view1vid.to(device), view2vid.to(device)

    # optimizer.zero_grad()

    print('Running inputs through model.')
    with torch.no_grad():
        output_v1, output_v2, rep_v1, rep_v2 = model(view1img, view2img, view1vid, view2vid)
        con_loss = criterion(rep_v1, rep_v2)
        recon_loss1 = criterion(output_v1, view1vid)
        recon_loss2 = criterion(output_v2, view2vid)
        loss = con_loss + recon_loss1 + recon_loss2

    running_loss += loss.item()

    print('Epoch {}/{} Loss: {}'.format(epoch, num_epochs, loss))


def train_model():
    print('Generating randomStuff data.')
    training_data, testing_data = generate_random_data()
    training_batches = create_batches(training_data)
    # for batch in training_batches.keys():
    #     print(batch)
    #     for view in training_batches[batch].keys():
    #         print(view)
    #         print(training_batches[batch][view].size())
    start_time = time.time()
    print('Training model.')
    for epoch in range(num_epochs):
        # print('Epoch {}/{}'.format(epoch, NUM_EPOCHS))
        print('Training...')
        train(training_batches, epoch)
        print('Validation...')
        test(testing_data, epoch)
    end_time = time.time()
    print('Time: {}'.format(end_time - start_time))


if __name__ == '__main__':
    train_model()
