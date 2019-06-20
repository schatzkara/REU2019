import time
import torch
from phase1.network import FullNetwork

num_epochs = 100


def train(dataset, epoch):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = FullNetwork()

    criterion = None
    optimizer = None

    train_data = DataLoader(...)

    start_time = time.time()
    running_loss = 0.0

    model.train()

    for inputs, targets in train_data:
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        print('Epoch {}/{}, {} Phase, Loss: {}'.format(epoch, num_epochs, phase, loss))
        end_time = time.time()
        print('Time: {}'.format(end_time - start_time))

        running_loss = 0.0


def test(dataset, epoch):
    def train_model(dataset, num_epochs):
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


def train_model():
    for epoch in range(num_epochs):
        data =
        train(data, epoch)
        test(data, epoch)
