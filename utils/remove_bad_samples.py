
split_file = 'C:/Users/Owner/Documents/UCF/Project/REU2019/data/panoptic/test.list'
new_split_file = 'C:/Users/Owner/Documents/UCF/Project/REU2019/data/panoptic/mod_test.list'
bad_file = 'C:/Users/Owner/Documents/UCF/Project/REU2019/dir_sizes.txt'

with open(new_split_file, 'w') as f_write:
    with open(split_file, 'r') as f_split:
        f_split = f_split.readlines()
        # print(f_split)
        with open(bad_file, 'r') as f_bad:
            f_bad = f_bad.readlines()
            for i in range(len(f_bad)):
                if i % 2 == 0:
                    line = f_bad[i].strip()
                    parts = line.split('/')
                    sample = '/'.join([parts[7], parts[8]])
                    frames = parts[10]
                    # print(sample)
                    # print(frames)
                    # 7,8,10
                    line = sample + ' ' + frames + '\n'
                    try:
                        f_split.remove(line)
                    except:
                        print(line)
    f_write.writelines(f_split)
