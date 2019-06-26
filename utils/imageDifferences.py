import os
import cv2
import numpy as np

root_dir = 'C:/Users/Owner/Documents/UCF/REU2019/phase2/videos/ntu_100epochs/'

if __name__ == '__main__':
    batch_dirs = [os.path.join(root_dir, batch) for batch in os.listdir(root_dir)]
    vid_dirs = []
    for batch in batch_dirs:
        vids = [os.path.join(batch, vid) for vid in os.listdir(batch)]
        vid_dirs.extend(vids)
    view_dirs = []
    for vid in vid_dirs:
        views = [os.path.join(vid, view) for view in os.listdir(vid)]
        view_dirs.extend(views)
    full_dirs_in = [os.path.join(view, 'input') for view in view_dirs]
    full_dirs_out = [os.path.join(view, 'output') for view in view_dirs]
    all_dirs = full_dirs_in + full_dirs_out
    # print(all_dirs)

    # for dir in all_dirs:
    dir = all_dirs[0]
    print(dir)
    frames = [os.path.join(dir, img) for img in os.listdir(dir)]
    frame1 = cv2.imread(frames[0])
    frame1 = np.array(frame1).astype(np.float32)
    frame_abs_diffs = []
    frame_net_diffs = []
    for img in frames:
        img = cv2.imread(img)
        img = np.array(img).astype(np.float32)
        frame_abs_diffs.append(np.sum(np.abs(img - frame1)) / (img.shape[0] * img.shape[1]))
        frame_net_diffs.append(np.sum(img - frame1))

    #     cv2.imshow('dir', img)
    #     cv2.waitKey(10)

    print(frame_abs_diffs)
    print(frame_net_diffs)
