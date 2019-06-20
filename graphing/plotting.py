import matplotlib.pyplot as plt


def single_plot(x_data, y_data, title, x_label, y_label):
    """
    Function to make a single plot.
    :param x_data: (list) The x-values.
    :param y_data: (list) The y-values corresponding to the x-values.
    :param title: (str) The title of the plot.
    :param x_label: (str) The x-axis label.
    :param y_label: (str) The y-axis label.
    :return: None
    """
    plt.figure(1, (18, 8))  # something, plot size
    plt.subplot(111)
    plt.plot(x_data, y_data)
    # plt.title(title)
    plt.xlabel(x_label, fontsize=12)
    plt.ylabel(y_label, fontsize=12)
    # plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()


def multi_line_plot(x_data, y_data, title, x_label, y_label):
    """
    Function to plot multiple lines on the same plot.
    :param x_data: (list of lists) Each list corresponds to a set of x-values to plot.
    :param y_data: (list of lists) Each list corresponds to the corresponding set of y-values to plot.
    :param title: (str) The title of the plot.
    :param x_label: (str) The x-axis label.
    :param y_label: (str) The y-axis label.
    :return: None
    """
    plt.figure(1, (18, 8))  # something, plot size
    plt.subplot(111)
    legend = []
    for i in range(len(x_data)):
        plt.plot(x_data[i], y_data[i])
        legend.append((i+1))
    plt.title(title)
    plt.xlabel(x_label, fontsize=12)
    plt.ylabel(y_label, fontsize=12)
    plt.legend(legend, loc='upper left')
    plt.show()


def multi_plot(plot_1, plot_2, *args):
    plot_list = [plot_1, plot_2]
    for plot in args:
        plot_list.append(plot)
    plt.figure(1, (18, 8))

    n = len(plot_list)
    for i in range(n):
        id = '1' + str(n) + str(i)  # 1 plot of n plots, in position i
        plot_list[i].subplt(int(id))

    plt.show()


def make_subplot(x_data, y_data, title, x_label, y_label):
    plt.plot(x_data, y_data)
    plt.title(title)
    plt.xlabel(x_label, fontsize=12)
    plt.ylabel(y_label, fontsize=12)
