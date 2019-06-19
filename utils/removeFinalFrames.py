
# split_file = '/home/c2-2/yogesh/datasets/panoptic/train.list'
split_file = 'C:/Users/Owner/Documents/UCF/Project/REU2019/data/panoptic/test.list'
# new_split_file = '/home/c2-2/yogesh/datasets/panoptic/mod_train.list'
new_split_file = 'C:/Users/Owner/Documents/UCF/Project/REU2019/data/panoptic/mod_test.list'


def remove_final_frames_from_split(split_file):
    with open(split_file, 'r') as file:
        file = file.readlines()
        file = [line.strip() for line in file]
        # print(file)
        sample_frames = []
        sample = ''
        for i in range(len(file)):
            # print('i:'+str(i))
            line = file[i]
            # print(line)
            line = line.split(' ')
            # print(line)
            # print(len(line))
            # print(line[1])
            s = line[0]
            f = line[1]
            if i == 0:
                sample = s
            if s == sample:
                sample_frames.append(f)
            if s != sample:
                sf = remove_final_frame_set(sample, sample_frames)
                lines = [(sample + ' ' + f + '\n') for f in sf]
                with open(new_split_file, 'a') as g:
                    # print(lines)
                    g.write(''.join(lines))
                sample = s
                sample_frames = []


def remove_final_frame_set(sample, sample_frames):
    # print(sample_frames)
    # print(sample_frames[0].split('_'))
    end_frames_dict = {int(f.split('_')[1]): f for f in sample_frames}
    end_frames = list(end_frames_dict.keys())
    end_frames.sort()
    # print(end_frames)
    last = end_frames[-1]
    print(sample + ' ' + end_frames_dict[last])
    end_frames_dict.pop(last)
    return list(end_frames_dict.values())


if __name__ == '__main__':
    remove_final_frames_from_split(split_file=split_file)
