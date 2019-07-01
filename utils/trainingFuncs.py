import torch
from utils.modelIOFuncs import get_first_frame
import time
import sms


def print_loss(traintest, epochbatch, step, total_steps, loss_names, loss_values):
    """
    Function to print out the loss values.
    :param traintest: "Training" or "Testing"
    :param epochbatch: "Epoch" or "Batch"
    :param step:
    :param total_steps:
    :param loss_names:
    :param loss_values:
    :return:
    """
    print_string = ''
    if epochbatch.lower() == 'epoch':
        print_string += traintest + ' '
    else:
        print_string += '\t'
    print_string += epochbatch + '{}/{}'.format(step, total_steps)
    for i in range(len(loss_values)):
        name, value = loss_names[i], loss_values[i]
        print_string += ' ' + name + ':{}'.format(value)
    print(print_string)


def training_loop(epoch, num_epochs, model, optimizer, criterion, device, trainloader):
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
    running_recon3_loss = 0.0
    running_recon4_loss = 0.0

    model.train()

    for batch_idx, (vp_diff, vid1, vid2) in enumerate(trainloader):
        vp_diff = vp_diff.to(device)
        vid1, vid2 = vid1.to(device), vid2.to(device)
        img1, img2 = get_first_frame(vid1), get_first_frame(vid2)
        img1, img2 = img1.to(device), img2.to(device)

        optimizer.zero_grad()

        gen_v1, gen_v2, recon_v1, recon_v2, rep_v1, rep_v2, rep_v1_est, rep_v2_est = model(vp_diff=vp_diff,
                                                                                           vid1=vid1, vid2=vid2,
                                                                                           img1=img1, img2=img2)
        # loss
        # consistency losses between video features
        con1_loss = criterion(rep_v1, rep_v1_est)
        con2_loss = criterion(rep_v2, rep_v2_est)
        # reconstruction losses for videos gen from new view
        recon1_loss = criterion(gen_v1, vid1)
        recon2_loss = criterion(gen_v2, vid2)
        # reconstruction losses for videos gen from features and same view
        recon3_loss = criterion(recon_v1, vid1)
        recon4_loss = criterion(recon_v2, vid2)
        loss = con1_loss + con2_loss + recon1_loss + recon2_loss + recon3_loss + recon4_loss
        loss.backward()
        optimizer.step()

        running_total_loss += loss.item()
        running_con1_loss += con1_loss.item()
        running_con2_loss += con2_loss.item()
        running_recon1_loss += recon1_loss.item()
        running_recon2_loss += recon2_loss.item()
        running_recon3_loss += recon3_loss.item()
        running_recon4_loss += recon4_loss.item()

        if (batch_idx + 1) % 10 == 0:
            print_loss(traintest='Training', epochbatch='Batch', step=batch_idx + 1, total_steps=len(trainloader),
                       loss_names=['loss', 'con1', 'con2', 'recon1', 'recon2', 'recon3', 'recon4'],
                       loss_values=[loss, con1_loss, con2_loss, recon1_loss, recon2_loss, recon3_loss, recon4_loss])

    epoch_loss_values = [running_total_loss, running_con1_loss, running_con2_loss, running_recon1_loss,
                         running_recon2_loss, running_recon3_loss, running_recon4_loss]
    epoch_loss_values = [val / len(trainloader) for val in epoch_loss_values]
    print_loss(traintest='Training', epochbatch='Epoch', step=epoch + 1, total_steps=num_epochs,
               loss_names=['loss', 'con1', 'con2', 'recon1', 'recon2', 'recon3', 'recon4'],
               loss_values=epoch_loss_values)


def testing_loop(epoch, num_epochs, model, criterion, device, testloader):
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
    running_recon3_loss = 0.0
    running_recon4_loss = 0.0

    model.eval()

    for batch_idx, (vp_diff, vid1, vid2) in enumerate(testloader):
        vp_diff = vp_diff.to(device)
        vid1, vid2 = vid1.to(device), vid2.to(device)
        img1, img2 = get_first_frame(vid1), get_first_frame(vid2)
        img1, img2 = img1.to(device), img2.to(device)

        with torch.no_grad():
            gen_v1, gen_v2, recon_v1, recon_v2, rep_v1, rep_v2, rep_v1_est, rep_v2_est = model(vp_diff=vp_diff,
                                                                                               vid1=vid1, vid2=vid2,
                                                                                               img1=img1, img2=img2)
            # loss
            # consistency losses between video features
            con1_loss = criterion(rep_v1, rep_v1_est)
            con2_loss = criterion(rep_v2, rep_v2_est)
            # reconstruction losses for videos gen from new view
            recon1_loss = criterion(gen_v1, vid1)
            recon2_loss = criterion(gen_v2, vid2)
            # reconstruction losses for videos gen from features and same view
            recon3_loss = criterion(recon_v1, vid1)
            recon4_loss = criterion(recon_v2, vid2)
            loss = con1_loss + con2_loss + recon1_loss + recon2_loss + recon3_loss + recon4_loss

        running_total_loss += loss.item()
        running_con1_loss += con1_loss.item()
        running_con2_loss += con2_loss.item()
        running_recon1_loss += recon1_loss.item()
        running_recon2_loss += recon2_loss.item()
        running_recon3_loss += recon3_loss.item()
        running_recon4_loss += recon4_loss.item()

        if (batch_idx + 1) % 10 == 0:
            print_loss(traintest='Training', epochbatch='Batch', step=batch_idx + 1, total_steps=len(testloader),
                       loss_names=['loss', 'con1', 'con2', 'recon1', 'recon2', 'recon3', 'recon4'],
                       loss_values=[loss, con1_loss, con2_loss, recon1_loss, recon2_loss, recon3_loss, recon4_loss])

    epoch_loss_values = [running_total_loss, running_con1_loss, running_con2_loss, running_recon1_loss,
                         running_recon2_loss, running_recon3_loss, running_recon4_loss]
    epoch_loss_values = [val / len(testloader) for val in epoch_loss_values]
    print_loss(traintest='Training', epochbatch='Epoch', step=epoch + 1, total_steps=num_epochs,
               loss_names=['loss', 'con1', 'con2', 'recon1', 'recon2', 'recon3', 'recon4'],
               loss_values=epoch_loss_values)

    return running_total_loss / len(testloader)


def train_model(num_epochs, model, optimizer, criterion, trainloader, testloader, device, weight_file):
    """
    Function to train and validate the model for all epochs.
    :return: None
    """
    min_loss = 0.0
    start_time = time.time()
    for epoch in range(num_epochs):
        print('Training...')
        training_loop(epoch, num_epochs, model, optimizer, criterion, device, trainloader)
        print('Validation...')
        loss = testing_loop(epoch, num_epochs, model, criterion, device, testloader)
        sms.send('Epoch {} Loss: {}'.format(epoch + 1, loss), "6304876751", "att")
        if epoch == 0 or loss < min_loss:
            min_loss = loss
            torch.save(model.state_dict(), weight_file)
            sms.send('Weights saved', "6304876751", "att")
    end_time = time.time()
    print('Time: {}'.format(end_time - start_time))
