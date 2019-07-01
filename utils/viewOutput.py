import os
import cv2

root_dir = 'C:/Users/Owner/Documents/UCF/output/phase3/videos/ntu_net3_5epochs/'
types_to_show = [
    # 'input',
    # 'output',
    # 'recon',
    'rep',
    'rep_est'
]

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
    item_types = ['input', 'output', 'recon', 'rep', 'rep_est']
    all_dirs_by_type = {}
    all_dirs = []
    for item in item_types:
        full_dirs = [os.path.join(view, item) for view in view_dirs]
        all_dirs_by_type[item] = full_dirs
        all_dirs.extend(full_dirs)
    print(all_dirs)

    for type, dirs in all_dirs_by_type.items():
        if type in types_to_show:
            for dir in dirs:
                print(dir)
                frames = [os.path.join(dir, img) for img in os.listdir(dir)]
                for img in frames:
                    img = cv2.imread(img)
                    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
                    cv2.resizeWindow('image', 600, 600)
                    cv2.imshow('image', img)
                    cv2.waitKey(15)
                    print(img)

    print('done')
