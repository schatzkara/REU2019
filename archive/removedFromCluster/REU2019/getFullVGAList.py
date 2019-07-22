import os


root_dir = '/home/c2-2/yogesh/datasets/panoptic/rgb_data/'


def get_existing_vga_list(root_dir):
    vga_list = []
    samples = os.listdir(root_dir)
    samples = [os.path.join(s, 'samples') for s in samples]
    for sample in samples:
        cams = os.listdir(os.path.join(root_dir, sample))
        for cam in cams:
            if cam not in vga_list:
                vga_list.append(cam)

    return vga_list


if __name__ == '__main__':
    vga_list = get_existing_vga_list(root_dir)
    print(vga_list)
    print(len(vga_list))
