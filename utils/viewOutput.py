import os
import cv2

root_dir = 'C:/Users/Owner/Documents/UCF/Project/REU2019/phase1/videos/pan_100epochs'

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
    print(all_dirs)

    for dir in all_dirs:
        print(dir)
        frames = [os.path.join(dir, img) for img in os.listdir(dir)]
        for img in frames:
            img = cv2.imread(img)
            cv2.imshow('dir', img)
            cv2.waitKey(10)

    print('done')
