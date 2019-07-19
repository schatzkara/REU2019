from data.NTUDataLoader import NTUDataset
import torch

data_root_dir = ''
train_split = 'C:/Users/Owner/Documents/UCF/REU2019/data/NTU/train.list'
param_file = 'C:/Users/Owner/Documents/UCF/REU2019/data/NTU/view.params'
BATCH_SIZE = 1
CHANNELS = 3
FRAMES = 16
SKIP_LEN = 2
HEIGHT = 112
WIDTH = 112
RANDOM_ALL = True
PRECROP = True
DIFF_ACTORS = False
DIFF_SCENES = False

# def decrypt_vid_name(vid_name):
#     """
#     Function to break up the meaning of the video name.
#     :param vid_name: (string) The name of the video.
#     :return: 4 ints representing the scene, person, repetition, and action that the video captures.
#     """
#     scene = int(vid_name[1:4])
#     pid = int(vid_name[5:8])
#     rid = int(vid_name[9:12])
#     action = int(vid_name[13:16])
#     sra = vid_name[:4] + vid_name[8:]
#
#     return scene, pid, rid, action, sra
#
#
if __name__ == '__main__':
    #     with open(data_file) as f:
    #         data_file = f.readlines()
    #     data_file = [line.strip() for line in data_file]
    #
    #     SRA = {}  # SceneRepetitionAction
    #
    #     for sample in data_file:
    #         sample = sample.split(' ')
    #         sample_id = sample[0][sample[0].index('/') + 1:]
    #         scene, pid, rid, action, sra = decrypt_vid_name(sample_id)
    #         print(sra)
    # if sra not in SRA.keys():
    #     SRA[sra] = [sample_id]
    # else:
    #     SRA[sra].append(sample_id)
    #
    # for sra, samples in SRA.items():
    #     print(sra)
    #     print(samples)
    #
    #
    # actors = []
    #
    # for sample in data_file:
    #     sample = sample.split(' ')
    #     sample_id = sample[0][sample[0].index('/') + 1:]
    #     scene, pid, rid, action = decrypt_vid_name(sample_id)
    #     print(action)
    #     if action == 18:
    #         if pid not in actors:
    #             actors.append(pid)
    #
    # actors.sort()
    #
    # print(actors)

    trainset = NTUDataset(root_dir=data_root_dir, data_file=train_split, param_file=param_file,
                          resize_height=HEIGHT, resize_width=WIDTH,
                          clip_len=FRAMES, skip_len=SKIP_LEN,
                          random_all=RANDOM_ALL, precrop=PRECROP, diff_actors=DIFF_ACTORS, diff_scenes=DIFF_SCENES)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    for batch_idx, info in enumerate(trainloader):
        if info[0] != info[1]:
            print('cry')
        # print(info)
