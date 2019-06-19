from graphing.plotting import single_plot, multi_line_plot
from graphing.getData import get_parameters, get_epoch_metrics

root_dir = 'logs/'
starting_epoch = 200
skip_epoch = 2
ending_epoch = 420
metrics = ['loss', 'con', 'recon1', 'recon2']


def plot_multiple_files(file1, file2, *args):
    file_list = [file1, file2]
    for file in args:
        file_list.append(file)

    for file in file_list:
        print(file)
        plot_one_file(file)


def plot_multiple_files_together(file1, file2, *args):
    file_list = [file1, file2]
    for file in args:
        file_list.append(file)

    for metric in metrics:
        training_x = []
        training_y = []
        val_x = []
        val_y = []
        for file in file_list:
            param_dict = get_parameters(file)
            print(param_dict)
            total_epochs = param_dict['total epochs']
            training_metrics, val_metrics = get_epoch_metrics(file)
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
    param_dict = get_parameters(file_path)
    print(param_dict)
    total_epochs = param_dict['total epochs']
    training_metrics, val_metrics = get_epoch_metrics(file_path)

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

        all_y_data = val_metrics[metric][starting_epoch:]
        y_data = [all_y_data[i] for i in range(0, len(all_y_data), skip_epoch)]
        x_data = range(starting_epoch, starting_epoch + len(all_y_data), skip_epoch)
        single_plot(x_data=x_data, y_data=y_data,
                    title='Validation {} vs. Epochs'.format(metric),
                    x_label='Epochs', y_label='Validation {}'.format(metric))


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
    file_list = ['./cluster/REU2019/logs/output_61462.out',
                 './cluster/REU2019/logs/output_61469.out',
                 './cluster/REU2019/logs/output_61471.out']
    plot_multiple_files_together(*file_list)
