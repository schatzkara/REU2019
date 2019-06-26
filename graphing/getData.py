"""
output file format:
    print('Parameters:')
    print('Views: {}, {}'.format(VIEW1, VIEW2))
    print('Batch size: {}'.format(BATCH_SIZE))
    print('Tensor size: ({},{},{},{})'.format(CHANNELS, FRAMES, HEIGHT, WIDTH))
    print('Skip Length: {}'.format(SKIP_LEN))
    print('Total Epochs: {}'.format(NUM_EPOCHS))
    print('Learning Rate: {}'.format((LR)))
    print('Training...')
        print('\tBatch {}/{} Loss: {}'.format(batch_idx + 1, len(trainloader), loss.item())) x m
    print('Training Epoch {}/{} Loss: {}'.format(epoch + 1, NUM_EPOCHS, (running_loss / len(trainloader))))
    print('Validation...')
        print('\tBatch {}/{} Loss: {}'.format(batch_idx + 1, len(testloader), loss.item())) x n
    print('Validation Epoch {}/{} Loss: {}'.format(epoch + 1, NUM_EPOCHS, (running_loss/len(testloader))))    x p
    print('Time: {}'.format(end_time - start_time))
"""

model_phase = 2  # 1 or 2


def get_parameters(output_file):
    """
    Function to get the model and data parameters for the experiment.
    :param output_file: (str) The path for the file that contains the experiment terminal output.
    :return: (dict) A dictionary with keys as parameter names and values as parameter values.
    """
    parameter_keys = ['batch size', 'tensor size', 'skip length', 'precrop', 'total epochs', 'learning rate', 'time']

    parameters = {}
    with open(output_file, 'r') as f:
        for line in f:
            line = line.lower().strip()
            for param in parameter_keys:
                # if it's a parameter line
                if line.startswith(param):
                    value = line[line.index(':')+1:]
                    try:
                        parameters[param] = int(value)
                    except:
                        try:
                            parameters[param] = float(value)
                        except:
                            parameters[param] = value

    return parameters


def get_epoch_metrics(output_file, model_phase):
    """
    Function to get the metric values for each epoch.
    :param output_file: (str) The path for the file that contains the experiment terminal output.
    :param model_phase: (int) The phase of the model that the output file is from.
    :return: (dict, dict) A dictionary representing the training metrics and another representing the val metrics.
              Keys are metric names and values are a list of the metric values for each epoch.
    """
    epoch_metrics_keys = {'training': 'training epoch', 'val': 'validation epoch'}
    if model_phase == 1:
        metrics = ['loss', 'con', 'recon1', 'recon2']
    elif model_phase == 2:
        metrics = ['loss', 'con1', 'con2', 'recon1', 'recon2']
    else:
        print('There are only 2 model phases.')

    # these dictionaries will hold lists for each metric
    training_metrics = {}
    val_metrics = {}

    # initialize an empty list to fill for each metric
    for metric in metrics:
        training_metrics[metric] = []
        val_metrics[metric] = []
    with open(output_file, 'r') as f:
        for line in f:
            line = line.lower().strip()
            # if it's in the training phase
            if line.startswith(epoch_metrics_keys['training']):
                line = line.replace(epoch_metrics_keys['training'] + " ", "")
                line_parts = line.split(" ")
                values = line_parts[1:]
                for i in range(0, len(values)):
                    metric_name, metric_value = values[i].split(":")
                    # metric_name = values[i][:values[i].index(':')]
                    # metric_value = values[i+1]
                    try:
                        metric_value = float(metric_value)
                    except:
                        print('Error: metric value was not a float.')
                    training_metrics[metric_name].append(metric_value)
            # if it's in the validation phase
            elif line.startswith(epoch_metrics_keys['val']):
                line = line.replace(epoch_metrics_keys['val'] + " ", "")
                line_parts = line.split(" ")
                values = line_parts[1:]
                for i in range(0, len(values)):
                    metric_name, metric_value = values[i].split(":")
                    # metric_name = values[i][:values[i].index(':')]
                    # metric_value = values[i + 1]
                    try:
                        metric_value = float(metric_value)
                    except:
                        print('Error: metric value was not a float.')
                    val_metrics[metric_name].append(metric_value)

    return training_metrics, val_metrics


def get_old_epoch_metrics(output_file):
    # these dictionaries will hold lists for each metric
    training_metrics = {}
    val_metrics = {}

    # initialize an empty list to fill for each metric
    for metric in metrics:
        training_metrics[metric] = []
        val_metrics[metric] = []
    with open(output_file, 'r') as f:
        for line in f:
            line = line.lower().strip()
            # if it's in the training phase
            if line.startswith(epoch_metrics_keys['training']):
                line = line.replace(epoch_metrics_keys['training'] + " ", "")
                line_parts = line.split(" ")
                values = line_parts[1:]
                for i in range(0, len(values), 2):
                    metric_name = values[i][:values[i].index(':')]
                    metric_value = values[i+1]
                    try:
                        metric_value = float(metric_value)
                    except:
                        print('Error: metric value was not a float.')
                    training_metrics[metric_name].append(metric_value)
            # if it's in the validation phase
            elif line.startswith(epoch_metrics_keys['val']):
                line = line.replace(epoch_metrics_keys['val'] + " ", "")
                line_parts = line.split(" ")
                values = line_parts[1:]
                for i in range(0, len(values), 2):
                    metric_name = values[i][:values[i].index(':')]
                    metric_value = values[i + 1]
                    val_metrics[metric_name].append(metric_value)

    return training_metrics, val_metrics


def get_old_old_epoch_metrics(output_file):
    # these dictionaries will hold lists for each metric
    training_metrics = {}
    val_metrics = {}

    # initialize an empty list to fill for each metric
    for metric in metrics:
        training_metrics[metric] = []
        val_metrics[metric] = []
    training = True  # toggle bool to mark that it's a training line, not a val line
    with open(output_file, 'r') as f:
        for line in f:
            line = line.lower().strip()
            if line.startswith('epoch') and training:
                line = line.replace("epoch ", "")
                line_parts = line.split(" ")
                values = line_parts[1:]
                for i in range(0, len(values), 2):
                    metric_name = values[i][:values[i].index(':')]
                    metric_value = values[i + 1]
                    try:
                        metric_value = float(metric_value)
                    except:
                        print('Error: metric value was not a float.')
                    training_metrics[metric_name].append(metric_value)
                training = False
            elif line.startswith('epoch') and not training:
                line = line.replace("epoch ", "")
                line_parts = line.split(" ")
                values = line_parts[1:]
                for i in range(0, len(values), 2):
                    metric_name = values[i][:values[i].index(':')]
                    metric_value = values[i + 1]
                    try:
                        metric_value = float(metric_value)
                    except:
                        print('Error: metric value was not a float.')
                    val_metrics[metric_name].append(metric_value)
                training = True

    return training_metrics, val_metrics


if __name__ == '__main__':
    file_path = './logstograph/output_61860.out'
    param_dict = get_parameters(file_path)
    print(param_dict)
    epoch_metrics = get_epoch_metrics(file_path, model_phase)
    print(epoch_metrics)
    training_metrics, val_metrics = epoch_metrics
    val_loss = val_metrics['loss']
    print(val_loss)
    min_loss = min(val_loss)
    print(min_loss)
