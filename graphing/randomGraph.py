from graphing.plotting import multi_line_plot


epochs_61796 = range(0, 36)
loss_61796 = [0.01657, 0.01467, 0.01357, 0.01307, 0.01267, 0.01238, 0.01184, 0.01171, 0.01129, 0.01095, 0.01073,
              5.76411, 0.04017, 0.02848, 0.02059, 0.01809, 0.01664, 0.01547, 0.01468, 0.01419, 0.01363, 0.01328,
              0.01292, 0.01244, 0.01251, 0.01194, 0.01165, 0.01151, 0.01131, 0.01118, 0.01091, 0.01088, 0.01073,
              0.01064, 0.41497, 0.14566]

epochs_61462 = range(508, 543)
loss_61642 = [0.05309, 0.05307, 0.05301, 0.05297, 0.05299, 0.05302, 0.0531, 0.05304, 0.05305, 0.05305, 0.05309, 0.05305,
              0.05302, 0.05301, 0.05304, 0.05309, 0.05303, 0.05309, 0.0529, 0.05287, 0.05303, 0.053, 0.05291, 0.05302,
              0.05295, 0.05292, 0.05308, 0.05287, 0.05316, 0.05295, 0.05298, 0.0529, 0.05302, 0.05299, 0.05293, 0.05293]

epochs_61813 = range(0, 36)
loss_61813 = [0.01296, 0.01095, 0.01029, 0.00965, 0.00927, 0.00916, 0.00903, 0.0088, 0.00902, 0.00859, 0.00844, 0.00845,
              0.00844, 0.00834, 0.00832, 0.00821, 0.00874, 0.00858, 0.00807, 0.0081, 0.00825, 0.008, 0.00813, 0.00793,
              0.00796, 0.00801, 0.33918, 0.18119, 0.14208, 0.12449, 0.11046, 0.10159, 0.09502, 0.08841, 0.08362]

epochs_61722 = range(63, 99)
loss_61722 = [0.05218, 0.05118, 0.05105, 0.05098, 0.05076, 0.05048, 0.05099, 0.05935, 0.05049, 0.04982, 0.04958,
              0.04987, 0.04917, 0.04883, 0.0487, 0.04834, 0.04841, 0.048, 0.04892, 0.04816, 0.04769, 0.04828, 0.04754,
              0.0475, 0.04774, 0.04738, 0.04748, 0.04811, 0.04737, 0.04729, 0.04705, 0.04751, 0.04701, 0.04826, 0.04701,
              0.047]


if __name__ == '__main__':
    # ntu old vs new
    # starting_point = 15
    # ending_point = 34
    # print(len(epochs_61796))
    # print(len(loss_61796))
    # print(len(epochs_61462))
    # print(len(loss_61642))
    # x_data = [epochs_61796[starting_point:ending_point], epochs_61796[starting_point:ending_point]]
    # y_data = [loss_61796[starting_point:ending_point], loss_61642[starting_point:ending_point]]
    # title = 'Validation Loss Old vs. New Network'
    # x_label = 'Epochs'
    # y_label = 'Loss'
    # multi_line_plot(x_data, y_data, title, x_label, y_label)

    # pan old vs new
    starting_point = 0
    ending_point = 26
    print(len(epochs_61813))
    print(len(loss_61813))
    print(len(epochs_61722))
    print(len(loss_61722))
    x_data = [epochs_61813[starting_point:ending_point], epochs_61813[starting_point:ending_point]]
    y_data = [loss_61813[starting_point:ending_point], loss_61722[starting_point:ending_point]]
    title = 'Validation Loss Old vs. New Network'
    x_label = 'Epochs'
    y_label = 'Loss'
    multi_line_plot(x_data, y_data, title, x_label, y_label)
