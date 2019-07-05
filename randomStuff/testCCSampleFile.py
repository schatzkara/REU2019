
cam_file = 'C:/Users/Owner/Documents/UCF/REU2019/data/panoptic/closecams.list'


def make_close_cams_dict(cam_file):
    close_cams_dict = {}
    with open(cam_file, 'r') as f:
        f = f.readlines()
        f = [line.strip() for line in f]
        current_sample = ''
        for i in range(len(f)):
            line = f[i]
            if not line.startswith('vga'):
                current_sample = line
                close_cams_dict[current_sample] = {}
            else:
                cams = line.strip().split(' ')
                close_cams_dict[current_sample][cams[0]] = cams[1:]

    return close_cams_dict


if __name__ == '__main__':
    close_cams_dict = make_close_cams_dict(cam_file)
    print(close_cams_dict)
    # print('\n\n')
    # for key, value in close_cams_dict.items():
    #     print(key)
    #     for key2, value2 in value.items():
    #         print(key2)
    #         print(value2)
