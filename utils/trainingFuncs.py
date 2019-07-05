import torch
import time
from utils import sms
from utils.modelIOFuncs import get_first_frame, convert_outputs
from utils.lossFuncs import calculate_loss, print_loss


def training_loop(epoch, num_epochs, model, optimizer, criterion, device, trainloader, loss_weights=None):
    """
    Function carrying out the training loop for the Full Network for a single epoch.
    :param epoch: (int) The current epoch in which the model is training.
    :return: None
    """
    running_losses = None

    model.train()

    for batch_idx, (vp_diff, vid1, vid2) in enumerate(trainloader):
        vp_diff = vp_diff.to(device)
        vid1, vid2 = vid1.to(device), vid2.to(device)
        img1, img2 = get_first_frame(vid1), get_first_frame(vid2)
        img1, img2 = img1.to(device), img2.to(device)

        optimizer.zero_grad()

        outputs = model(vp_diff=vp_diff, vid1=vid1, vid2=vid2, img1=img1, img2=img2)

        losses, running_losses = calculate_loss(criterion=criterion,
                                                inputs=[vp_diff, vid1, vid2, img1, img2], outputs=outputs,
                                                running_losses=running_losses, loss_weights=loss_weights)
        loss = losses['loss']
        loss.backward()
        optimizer.step()

        if (batch_idx + 1) % 10 == 0:
            print_loss(traintest='Training', epochbatch='Batch', step=batch_idx + 1, total_steps=len(trainloader),
                       loss_names=list(losses.keys()), loss_values=list(losses.values()))

    epoch_loss_values = {name: val / len(trainloader) for name, val in running_losses.items()}
    print_loss(traintest='Training', epochbatch='Epoch', step=epoch + 1, total_steps=num_epochs,
               loss_names=list(epoch_loss_values.keys()), loss_values=list(epoch_loss_values.values()))


def validation_loop(epoch, num_epochs, model, criterion, device, testloader, loss_weights=None):
    """
    Function to carry out the testing/validation loop for the Full Network for a single epoch.
    :param epoch: (int) The current epoch in which the model is testing/validating.
    :return: None
    """
    running_losses = None

    model.eval()

    for batch_idx, (vp_diff, vid1, vid2) in enumerate(testloader):
        vp_diff = vp_diff.to(device)
        vid1, vid2 = vid1.to(device), vid2.to(device)
        img1, img2 = get_first_frame(vid1), get_first_frame(vid2)
        img1, img2 = img1.to(device), img2.to(device)

        with torch.no_grad():
            outputs = model(vp_diff=vp_diff, vid1=vid1, vid2=vid2, img1=img1, img2=img2)

            losses, running_losses = calculate_loss(criterion=criterion,
                                                    inputs=[vp_diff, vid1, vid2, img1, img2], outputs=outputs,
                                                    running_losses=running_losses, loss_weights=loss_weights)
            loss = losses['loss']

        if (batch_idx + 1) % 10 == 0:
            print_loss(traintest='Validation', epochbatch='Batch', step=batch_idx + 1, total_steps=len(testloader),
                       loss_names=list(losses.keys()), loss_values=list(losses.values()))

    epoch_loss_values = {name: val / len(testloader) for name, val in running_losses.items()}
    print_loss(traintest='Validation', epochbatch='Epoch', step=epoch + 1, total_steps=num_epochs,
               loss_names=list(epoch_loss_values.keys()),
               loss_values=list(epoch_loss_values.values()))

    total_loss = epoch_loss_values['loss']
    return total_loss


def testing_loop(model, criterion, device, testloader, output_dir, loss_weights=None):
    """
    Function to carry out the testing/validation loop for the Full Network for a single epoch.
    :return: None
    """
    running_losses = None

    model.eval()

    for batch_idx, (vp_diff, vid1, vid2) in enumerate(testloader):
        vp_diff = vp_diff.to(device)
        vid1, vid2 = vid1.to(device), vid2.to(device)
        img1, img2 = get_first_frame(vid1), get_first_frame(vid2)
        img1, img2 = img1.to(device), img2.to(device)

        with torch.no_grad():
            outputs = model(vp_diff=vp_diff, vid1=vid1, vid2=vid2, img1=img1, img2=img2)

            # save videos
            convert_outputs(inputs=[vp_diff, vid1, vid2, img1, img2], outputs=outputs,
                            output_dir=output_dir, batch_num=batch_idx + 1)

            # loss
            losses, running_losses = calculate_loss(criterion=criterion,
                                                    inputs=[vp_diff, vid1, vid2, img1, img2], outputs=outputs,
                                                    running_losses=running_losses, loss_weights=loss_weights)
            loss = losses['loss']

        if (batch_idx + 1) % 10 == 0:
            print_loss(traintest='Testing', epochbatch='Batch', step=batch_idx + 1, total_steps=len(testloader),
                       loss_names=list(losses.keys()), loss_values=list(losses.values()))

        epoch_loss_values = {name: val / len(testloader) for name, val in running_losses.items()}
        print_loss(traintest='Testing', epochbatch='Epoch', step=1, total_steps=1,
                   loss_names=list(epoch_loss_values.keys()),
                   loss_values=list(epoch_loss_values.values()))


def train_model(num_epochs, model, optimizer, criterion,
                trainloader, testloader, device, weight_file, loss_weights=None):
    """
    Function to train and validate the model for all epochs.
    :return: None
    """
    min_loss = 0.0
    start_time = time.time()
    for epoch in range(num_epochs):
        print('Training...')
        training_loop(epoch, num_epochs, model, optimizer, criterion, device, trainloader, loss_weights)
        print('Validation...')
        loss = validation_loop(epoch, num_epochs, model, criterion, device, testloader, loss_weights)
        sms.send('Epoch {} Loss: {}'.format(epoch + 1, loss), "6304876751", "att")
        if epoch == 0 or loss < min_loss:
            min_loss = loss
            torch.save(model.state_dict(), weight_file)
            sms.send('Weights saved', "6304876751", "att")
    end_time = time.time()
    print('Time: {}'.format(end_time - start_time))


def test_model(model, criterion, device, testloader, output_dir, loss_weights=None):
    """
    Function to test the model.
    :return: None
    """
    start_time = time.time()
    testing_loop(model, criterion, device, testloader, output_dir, loss_weights)
    end_time = time.time()
    print('Time: {}'.format(end_time - start_time))
