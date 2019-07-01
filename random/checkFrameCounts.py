contents = ''

with open('val.list', 'r') as f:
    i = 0
    for line in f:
        sample_id, nf_v1, nf_v2, nf_v3 = line.split(' ')
        nf_v1, nf_v2, nf_v3 = int(nf_v1), int(nf_v2), int(nf_v3)
        frame_count = min(nf_v1, nf_v2, nf_v3)
        # print(frame_count)
        skip_len = 2
        clip_len = 16
        # print(skip_len * clip_len)
        max_frame = frame_count - (skip_len * clip_len)

        if max_frame < 1:
            print(i, sample_id, frame_count)
        else:
            contents += line
        i += 1

with open('val16.list', 'w') as f:
    f.write(contents)
