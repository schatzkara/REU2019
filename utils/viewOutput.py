import os
import cv2
import numpy as np

root_dir = '/home/yogesh/kara/REU2019/'
# root_dir = 'C:/Users/Owner/Documents/UCF/output/'

# vid_dir = 'phase3/videos/ntu_net3_41epochs/'
# vid_dir = 'phase3/videos/pan_net3_57epochs/'
vid_dir = 'noconsistency/videos/ntu_netnocon_50epochs/'

full_dir = os.path.join(root_dir, vid_dir)
print(full_dir)

possible_views = [1, 2]
types_to_show = [
    'input',
    'output',
    # 'recon',
    # 'rep',
    # 'rep_est'
]
height = width = 112
frames = 16


def make_frame_name(frame_num):
    """
    Function to correctly generate the correctly formatted .png file name for the frame.
    :param frame_num: The frame number captured in the file.
    :return: str representing the file name.
    """
    return str(frame_num).zfill(3) + '.jpg'


if __name__ == '__main__':
    batch_dirs = [os.path.join(full_dir, batch) for batch in os.listdir(full_dir)]
    vid_dirs = []
    for batch in batch_dirs:
        vids = [os.path.join(batch, vid) for vid in os.listdir(batch)]
        vid_dirs.extend(vids)
    view_dirs = []
    for vid in vid_dirs:
        views = [os.path.join(vid, view) for view in os.listdir(vid)]
        view_dirs.extend(views)
    item_types = ['input', 'output', 'recon', 'rep', 'rep_est']

    # view all frames 1 by 1
    '''all_dirs_by_type = {}
    all_dirs = []
    for item in item_types:
        full_dirs = [os.path.join(view, item) for view in view_dirs]
        all_dirs_by_type[item] = full_dirs
        all_dirs.extend(full_dirs)
    # print(all_dirs)

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
                    # cv2.destroyAllWindows()'''

    # view just gt and recon
    '''all_view_dirs = {}
    for view in view_dirs:
        full_dirs = [os.path.join(view, item) for item in types_to_show]
        all_view_dirs[view] = full_dirs

    for view, dirs in all_view_dirs.items():
        for i in range(frames):
            frame = make_frame_name(i+1)
            path = os.path.join(dirs[0], frame)
            display = cv2.imread(path)
            print(dirs[0])
            # print(path)
            # display = cv2.resize(display, dsize=(height, width), interpolation=cv2.INTER_AREA)
            for j in range(1, len(dirs)):
                path = os.path.join(dirs[j], frame)
                addFrame = cv2.imread(path)
                print(dirs[j])
                # print(path)
                # addFrame = cv2.resize(addFrame, dsize=(height, width), interpolation=cv2.INTER_AREA)
                display = np.hstack((display, addFrame))
            # img = cv2.imread(img)
            cv2.namedWindow('image', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('image', 900, 300)
            cv2.imshow('image', display)
            cv2.waitKey(250)
            # print(img)
            # cv2.destroyAllWindows()'''

    # view gt and recon for both views
    for video in vid_dirs:
        print(video)
        for i in range(frames):
            frame = make_frame_name(i + 1)
            # k = 0
            # first_path = os.path.join(vid, views[0], types_to_show[0], frame)
            # display = cv2.imread(first_path)
            # print(first_path)
            # full_display = None
            # display = None
            all_displays = []
            for v in possible_views:
                view_path_ = os.path.join(video, str(v))
                # print(view_path_)
                displays = []
                for type_ in types_to_show:
                    # if view == views[0] and type == types_to_show[0]:
                        # continue
                    path = os.path.join(view_path_, type_, frame)
                    addFrame = cv2.imread(path)
                    # print(path)
                    # if k == 0:
                        # display = addFrame
                    # else:
                    # print(display.shape)
                    # print(addFrame.shape)
                    displays.append(addFrame)
                display = np.hstack((displays))
                all_displays.append(display)
                # full_display = np.vstack((full_display, display))
                # k += 1
            # print(len(all_displays))
            full_display = np.vstack((all_displays))
            cv2.namedWindow('image', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('image', 600, 600)
            cv2.imshow('image', full_display)
            cv2.waitKey(250)
            # print(display)
            # cv2.destroyAllWindows()

    print('done')
