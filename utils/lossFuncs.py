import numpy as np


def print_loss(traintest, epochbatch, step, total_steps, loss_names, loss_values):
    """
    Function to print out the loss values.
    :param traintest: "Training", "Validation", or "Testing"
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
    print_string += epochbatch + ' {}/{}'.format(step, total_steps)
    for i in range(len(loss_values)):
        name, value = loss_names[i], loss_values[i]
        print_string += ' ' + name + ':{}'.format("{0:.6f}".format(value))
    print(print_string)


def calculate_loss(criterion, inputs, outputs, running_losses, loss_weights):
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
