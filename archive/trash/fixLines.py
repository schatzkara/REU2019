
file_path = 'C:/Users/Owner/Documents/UCF/Project/REU2019/dir_sizes.txt'
new_file_path = 'C:/Users/Owner/Documents/UCF/Project/REU2019/dir_sizes2.txt'


if __name__ == '__main__':
    with open(new_file_path, 'w') as g:
        with open(file_path, 'r') as f:
            f = f.readlines()
            samples = f[0].split('/home')
            samples = ['/home' + path for path in samples[1:]]
            samples, one = samples[:-1], samples[-1]
            # print(samples)
            # print(one)
            for sample in samples:
                parts = sample.split('/')
                frames = parts[-1]
                # print(frames)
                start, end_and_count = frames.split('_')
                start = int(start)
                end = start + 125
                count = end_and_count.replace(str(end), '')
                # print(start)
                # print(end)
                # print(count)
                # print(parts[:-1])
                new_parts = parts[:-1]
                new_parts.append(str(start) + '_' + str(end))
                # print(new_parts)
                line = '/'.join(new_parts)
                # print(line)
                g.write(line + '\n')
                g.write(str(count) + '\n')
            g.write(one)
            g.writelines(f[1:])
