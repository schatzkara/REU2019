from graphing.plotting import single_plot, multi_line_plot
from graphing.getData import get_parameters, get_epoch_metrics

model_phase = 5  # 0 to 4

job_numbers = [66984, 67219, 67232, 67288]

root_dir = './logstograph/gruTrans/'
file_name_start, file_name_end = 'output_', '.out'
starting_epoch = 50
skip_epoch = 1
ending_epoch = 80
if model_phase == 0:
    metrics = ['loss', 'con', 'recon1', 'recon2']
elif model_phase == 1:
    metrics = ['loss', 'con1', 'con2', 'recon1', 'recon2']
elif model_phase == 2:
    metrics = ['loss', 'con1app', 'con2app', 'con1kp', 'con2kp', 'recon1', 'recon2']
elif model_phase == 3:
    metrics = ['loss', 'con1', 'con2', 'con3', 'con4', 'recon1', 'recon2']
elif model_phase == 4:
    metrics = ['loss']
elif model_phase == 5:
    metrics = ['reconloss', 'vploss']
else:
    print('Invalid generator phase number.')


def plot_multiple_files(file1, file2, *args):
    """
    Function to plot multiple flies side by side.
    :param file1: (str) The path of the first file to plot.
    :param file2: (str) The path of the second file to plot.
    :param args: The paths of any subsequent files to plot.
    :return: None
    """
    file_list = [file1, file2]
    for file in args:
        file_list.append(file)

    for file in file_list:
        print(file)
        plot_one_file(file)


def plot_multiple_files_together(file1, file2, *args):
    """
    Function to plot multiple files on the same plot.
    :param file1: (str) The path of the first file to plot.
    :param file2: (str) The path of the second file to plot.
    :param args: The paths of any subsequent files to plot.
    :return: None
    """
    file_list = [file1, file2]
    for file in args:
        file_list.append(file)

    for metric in metrics:
        training_x = []
        training_y = []
        val_x = []
        val_y = []
        for file in file_list:
            print(file)
            param_dict = get_parameters(file)
            print(param_dict)

            total_epochs = param_dict['total epochs']
            training_metrics, val_metrics = get_epoch_metrics(file, model_phase)
            print('Num Epochs Run: {}'.format(len(training_metrics[metric])))
            all_y_data = training_metrics[metric][starting_epoch:ending_epoch]
            # print(y_data)
            y_data = [all_y_data[i] for i in range(0, len(all_y_data), skip_epoch)]
            # print(y_data)
            x_data = range(starting_epoch, starting_epoch + len(all_y_data), skip_epoch)
            # print(x_data)
            training_x.append(x_data)
            training_y.append(y_data)

            all_y_data = val_metrics[metric][starting_epoch:ending_epoch]
            y_data = [all_y_data[i] for i in range(0, len(all_y_data), skip_epoch)]
            x_data = range(starting_epoch, starting_epoch + len(all_y_data), skip_epoch)
            val_x.append(x_data)
            val_y.append(y_data)

        multi_line_plot(x_data=training_x, y_data=training_y,
                        title='Training {} vs. Epochs'.format(metric),
                        x_label='Epochs', y_label='Training {}'.format(metric))
        multi_line_plot(x_data=val_x, y_data=val_y,
                        title='Validation {} vs. Epochs'.format(metric),
                        x_label='Epochs', y_label='Validation {}'.format(metric))


def plot_one_file(file_path):
    """
    Funtion to plot a single file.
    :param file_path: (str) The path of the file to plot.
    :return: None
    """
    print(file_path)
    param_dict = get_parameters(file_path)
    print(param_dict)
    total_epochs = param_dict['total epochs']
    training_metrics, val_metrics = get_epoch_metrics(file_path, model_phase)
    print('Num Epochs Run: {}'.format(len(training_metrics['loss'])))

    for metric in metrics:
        all_y_data = training_metrics[metric][starting_epoch:ending_epoch]
        # print(y_data)
        y_data = [all_y_data[i] for i in range(0, len(all_y_data), skip_epoch)]
        # print(y_data)
        x_data = range(starting_epoch, starting_epoch + len(all_y_data), skip_epoch)
        # print(x_data)
        single_plot(x_data=x_data, y_data=y_data,
                    title='Training {} vs. Epochs'.format(metric),
                    x_label='Epochs', y_label='Training {}'.format(metric))

        all_y_data = val_metrics[metric][starting_epoch:ending_epoch]
        y_data = [all_y_data[i] for i in range(0, len(all_y_data), skip_epoch)]
        x_data = range(starting_epoch, starting_epoch + len(all_y_data), skip_epoch)
        single_plot(x_data=x_data, y_data=y_data,
                    title='Validation {} vs. Epochs'.format(metric),
                    x_label='Epochs', y_label='Validation {}'.format(metric))

        print(metric)
        print(x_data)
        print(y_data)


if __name__ == '__main__':
    # file_list = os.listdir(root_dir)
    # file_list = [os.path.join(root_dir, item) for item in file_list]
    # for item in file_list:
    #     if os.path.isdir(item):
    #         file_list.remove(item)
    # plot_multiple_files(*file_list)

    # file_name = './logs/output_61272.out'
    # plot_one_file(file_name)
    #
    # file_list = ['./logs/output_61272.out', './logs/output_61276.out', './logs/output_61277.out']
    # file_list = ['./logs/output_61389.out', './logs/output_61390.out']
    # , './logs/output_61391.out']  './logs/output_61389.out',
    # file_list = ['./cluster/REU2019/logs/output_61462.out',
    #              './cluster/REU2019/logs/output_61469.out',
    #              './cluster/REU2019/logs/output_61471.out']

    file_list = [root_dir + file_name_start + str(job_id) + file_name_end for job_id in job_numbers]
    if len(file_list) == 1:
        plot_one_file(file_list[0])
    else:
        plot_multiple_files_together(*file_list)
