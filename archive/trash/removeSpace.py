
file_path = 'C:/Users/Owner/Documents/UCF/REU2019/graphing/logstograph/output_62232.out'

with open(file_path + '2', 'w') as f:
    with open(file_path, 'r') as g:
        for line in g:
            if 'con2app:' in line:
                new_line = line[:line.index('con2app:')+8] + line[line.index('con2app:')+9:]
                print(new_line)
                f.write(new_line)
            else:
                f.write(line)
