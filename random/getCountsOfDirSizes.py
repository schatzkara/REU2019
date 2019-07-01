
file_path = 'C:/Users/Owner/Documents/UCF/Project/REU2019/dir_sizes.txt'

if __name__ == '__main__':
    with open(file_path, 'r') as f:
        f = f.readlines()
        sample = ''
        count = 0
        num = 0
        for i in range(len(f)):
            if i % 2 == 0:
                sample = f[i].strip()
            elif i % 2 == 1:
                count = int(f[i])
                if count != 0:
                    num += 1
                    print(sample)
                    print(count)
        print(num)

