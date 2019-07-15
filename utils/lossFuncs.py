import numpy as np


def print_loss(traintest, epochbatch, step, total_steps, loss_names, loss_values):
    """
    Function to print out the loss values.
    :param traintest: (str) Indicates the current stage of the experiment, i.e. "Training", "Validation", or "Testing".
    :param epochbatch: (str) Indicates what stage of loss values are being reported, i.e. "Epoch" or "Batch".
    :param step: (int) The current loop index, i.e. the batch # or the epoch #.
    :param total_steps: (int) The total number of loops that will occur, i.e. The # of batches or the # of epochs.
    :param loss_names: (list) The names of each of the losses calculated by the model.
    :param loss_values: (list) The current values of each of the losses calculated by the model. Must be in the same
                        order as loss_names.
    :return: None
    """
    print_string = ''
    if epochbatch.lower() == 'epoch':
        print_string += traintest + ' '
    else:
        print_string += '\t'
    print_string += epochbatch + ' {}/{}'.format(step, total_steps)
    for i in range(len(loss_values)):
        name, value = loss_names[i], loss_values[i]
        print_string += ' ' + name + ':{}'.format("{0:.6f}".format(value))
    print(print_string)


def calculate_loss(criterion, inputs, outputs, running_losses, loss_weights):
    """
    Function to calculate the loss values for the model.
    :param criterion: (function) The loss function that is to be used to calculate each of the losses.
    :param inputs: (list) The model's input tensors.
    :param outputs: (list) The model's output tensors.
    :param running_losses: (dict) The current running loss values to increment; key: loss name, value: loss value.
    :param loss_weights: (dict) The weights to use (if any) for the different losses; key: loss name, value: loss value.
    :return: 2 dicts containing the current and running loss values, key: loss name, value: loss value.
    """
    if len(outputs) == 4:
        losses, running_losses = calculate_loss_4(criterion, inputs, outputs, running_losses, loss_weights)
    elif len(outputs) == 6:
        losses, running_losses = calculate_loss_6(criterion, inputs, outputs, running_losses, loss_weights)
    elif len(outputs) == 8:
        losses, running_losses = calculate_loss_8(criterion, inputs, outputs, running_losses, loss_weights)
    elif len(outputs) == 10:
        losses, running_losses = calculate_loss_10(criterion, inputs, outputs, running_losses, loss_weights)

    return losses, running_losses


def calculate_total_weighted_loss(losses, loss_weights):
    total_loss = 0.0
    for name in losses.keys():
        try:
            total_loss += losses[name] * loss_weights[name]
        except:
            total_loss += losses[name]

    return total_loss


def calculate_loss_4(criterion, inputs, outputs, running_losses, loss_weights):
    vp_diff, vid1, vid2, img1, img2 = inputs
    gen_v1, gen_v2, rep_v1, rep_v2 = outputs
    # loss
    # consistency losses between video features
    con_loss = criterion(rep_v1, rep_v2)
    # reconstruction losses for videos gen from new view
    recon1_loss = criterion(gen_v1, vid1)
    recon2_loss = criterion(gen_v2, vid2)

    losses = {'con': con_loss, 'recon1': recon1_loss, 'recon2': recon2_loss}

    if loss_weights is None:
        total_loss = con_loss + recon1_loss + recon2_loss
    else:
        total_loss = calculate_total_weighted_loss(losses, loss_weights)

    losses['loss'] = total_loss

    running_losses = update_running_losses(losses, running_losses)

    return losses, running_losses


def calculate_loss_6(criterion, inputs, outputs, running_losses, loss_weights):
    vp_diff, vid1, vid2, img1, img2 = inputs
    gen_v1, gen_v2, kp_v1, kp_v2, kp_v1_est, kp_v2_est = outputs
    # loss
    # consistency losses between video features
    # con_loss = criterion(rep_v1, rep_v2)
    con1_loss = criterion(kp_v1, kp_v1_est)
    con2_loss = criterion(kp_v2, kp_v2_est)
    # reconstruction losses for videos gen from new view
    recon1_loss = criterion(gen_v1, vid1)
    recon2_loss = criterion(gen_v2, vid2)

    losses = {'con1': con1_loss, 'con2': con2_loss,
              'recon1': recon1_loss, 'recon2': recon2_loss}

    if loss_weights is None:
        total_loss = con1_loss + con2_loss + recon1_loss + recon2_loss
    else:
        total_loss = calculate_total_weighted_loss(losses, loss_weights)

    losses['loss'] = total_loss

    running_losses = update_running_losses(losses, running_losses)

    return losses, running_losses


def calculate_loss_8(criterion, inputs, outputs, running_losses, loss_weights):
    vp_diff, vid1, vid2, img1, img2 = inputs
    gen_v1, gen_v2, recon_v1, recon_v2, rep_v1, rep_v2, rep_v1_est, rep_v2_est = outputs
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

    losses = {'con1': con1_loss, 'con2': con2_loss,
              'recon1': recon1_loss, 'recon2': recon2_loss,
              'recon3': recon3_loss, 'recon4': recon4_loss}
    if loss_weights is None:
        total_loss = con1_loss + con2_loss + recon1_loss + recon2_loss + recon3_loss + recon4_loss
    else:
        total_loss = calculate_total_weighted_loss(losses, loss_weights)

    losses['loss'] = total_loss

    running_losses = update_running_losses(losses, running_losses)

    return losses, running_losses


def calculate_loss_10(criterion, inputs, outputs, running_losses, loss_weights):
    vp_diff, vid1, vid2, img1, img2 = inputs
    gen_v1, gen_v2, rep_v1, rep_v2, rep_v1_est, rep_v2_est, kp_v1, kp_v2, kp_v1_est, kp_v2_est = outputs
    # loss
    # consistency losses between video features
    con1_loss = criterion(rep_v1, rep_v1_est)
    con2_loss = criterion(rep_v2, rep_v2_est)
    con3_loss = criterion(kp_v1, kp_v1_est)
    con4_loss = criterion(kp_v2, kp_v2_est)
    # reconstruction losses for videos gen from new view
    recon1_loss = criterion(gen_v1, vid1)
    recon2_loss = criterion(gen_v2, vid2)

    losses = {'con1': con1_loss, 'con2': con2_loss,
              'con3': con3_loss, 'con4': con4_loss,
              'recon1': recon1_loss, 'recon2': recon2_loss}

    if loss_weights is None:
        total_loss = con1_loss + con2_loss + con3_loss + con4_loss + recon1_loss + recon2_loss
    else:
        total_loss = calculate_total_weighted_loss(losses, loss_weights)

    losses['loss'] = total_loss

    running_losses = update_running_losses(losses, running_losses)

    return losses, running_losses


def update_running_losses(losses, running_losses):
    if running_losses is None:
        running_losses = losses.copy()
    else:
        for name, value in losses.items():
            running_losses[name] += value.item()

    return running_losses
