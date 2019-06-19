import os


def remove_commas(file_path):
    contents = ''
    with open(file_path, 'r') as f:
        for line in f:
            contents += line.replace(",", "")

    with open(file_path, 'w') as f:
        f.write(contents)


if __name__ == "__main__":
    # root_dir = "./logs"
    # file_list = os.listdir(root_dir)
    # file_list = [os.path.join(root_dir, item) for item in file_list]
    # for item in file_list:
    #     if os.path.isdir(item):
    #         file_list.remove(item)
    # print(file_list)
    # file_list.remove('./logs\\old_format')
    # print(file_list)
    # for file in file_list:
    #     remove_commas(file)
    file = './logs/output_61272.out'
    remove_commas(file)
