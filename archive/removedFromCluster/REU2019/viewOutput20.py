import os
import cv2
import numpy as np

root_dir = '/home/yogesh/kara/REU2019/'
# root_dir = 'C:/Users/Owner/Documents/UCF/output/'

# vid_dir = '1pipeTrans/videos/ntu_400e/'
vid_dir = 'appSkipConns/videos/ntu_70e/'
# vid_dir = 'repSkipConns/videos/ntu_/'
# vid_dir = 'bothSkipConns/videos/ntu_/'
# vid_dir = 'repSkipConns/videos/ntu_/'
# vid_dir = 'lstmTrans/videos/ntu_64e/'
# vid_dir = '7x7lstm/videos/ntu_44e/'

full_dir = os.path.join(root_dir, vid_dir)
print(full_dir)

batches_to_view = [166, 167]  # None  # [208]  # range(165, 175)  # range(200, 210)  # range(80, 90)  #  [11, 15, 17, 18]
# range(15, 16)  # range(80, 90)  # None  # [11, 15, 17, 18]  # [18, 20, 84, 119, 145, 193, 199]  # , 303, 309, 341]

possible_views = [1, 2]
types_to_show = [
    'input',
    'output',
    # 'recon',
    # 'rep',
    # 'rep_est',
    # 'kp',
    # 'kp_est'
]
height = width = 112
frames = 16
if 'rep' in types_to_show or 'kp' in types_to_show:
    frames = 4
nkp = 32

display_time = 100
display_size = 400


def make_frame_name(frame_num):
    """
    Function to correctly generate the correctly formatted .png file name for the frame.
    :param frame_num: The frame number captured in the file.
    :return: str representing the file name.
    """
    return str(frame_num).zfill(3) + '.jpg'


def view_all_single_frames():
    all_dirs_by_type = {}
    all_dirs = []
    for item in item_types:
        full_dirs = [os.path.join(view, item) for view in view_dirs]
        all_dirs_by_type[item] = full_dirs
        all_dirs.extend(full_dirs)
    # print(all_dirs)

    for type_, dirs_ in all_dirs_by_type.items():
        if type_ in types_to_show:
            for dir_ in dirs_:
                print(dir_)
                frames_ = [os.path.join(dir_, img) for img in os.listdir(dir_)]
                for img in frames_:
                    img = cv2.imread(img)
                    cv2.namedWindow(vid_dir, cv2.WINDOW_NORMAL)
                    cv2.resizeWindow(vid_dir, display_size, display_size)
                    cv2.imshow(vid_dir, img)
                    cv2.waitKey(display_time)


def view_input_gt_recon():
    sample_dirs = {}
    for vid in vid_dirs:
        input_path = os.path.join(vid, str(1), 'input')
        gt_path = os.path.join(vid, str(2), 'input')
        recon_path = os.path.join(vid, str(2), 'output')
        sample_dirs[vid] = [input_path, gt_path, recon_path]

    for vid, dirs in sample_dirs.items():
        print(vid)
        for i in range(frames):
            frame = make_frame_name(i + 1)
            path = os.path.join(dirs[0], frame)
            display = cv2.imread(path)
            for j in range(1, len(dirs)):
                path = os.path.join(dirs[j], frame)
                add_frame = cv2.imread(path)
                display = np.hstack((display, add_frame))
            cv2.namedWindow(vid_dir, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(vid_dir, 600, 300)
            cv2.imshow(vid_dir, display)
            cv2.waitKey(display_time)


def view_gt_recon():
    all_view_dirs = {}
    for view in view_dirs:
        full_dirs = [os.path.join(view, item) for item in types_to_show]
        all_view_dirs[view] = full_dirs

    for view, dirs in all_view_dirs.items():
        print(view)
        for i in range(frames):
            frame = make_frame_name(i + 1)
            path = os.path.join(dirs[0], frame)
            display = cv2.imread(path)
            for j in range(1, len(dirs)):
                path = os.path.join(dirs[j], frame)
                add_frame = cv2.imread(path)
                display = np.hstack((display, add_frame))
            cv2.namedWindow(vid_dir, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(vid_dir, 600, 300)
            cv2.imshow(vid_dir, display)
            cv2.waitKey(display_time)


def view_both_gt_recon():
    for video in vid_dirs:
        print(video)
        for i in range(frames):
            if 'output' in types_to_show:
                frame = make_frame_name(i + 1)
                all_displays = []
                for v in possible_views:
                    view_path_ = os.path.join(video, str(v))
                    displays = []
                    for type_ in types_to_show:
                        path = os.path.join(view_path_, type_, frame)
                        add_frame = cv2.imread(path)
                        displays.append(add_frame)
                    display = np.hstack((displays))
                    all_displays.append(display)
                full_display = np.vstack((all_displays))
                cv2.namedWindow(vid_dir, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(vid_dir, display_size, display_size)
                cv2.imshow(vid_dir, full_display)
                cv2.waitKey(display_time)

            elif 'kp' in types_to_show:
                for kp in range(nkp):
                    frame = make_frame_name(i + 1)[:-4] + make_frame_name(kp + 1)
                    all_displays = []
                    for v in possible_views:
                        view_path_ = os.path.join(video, str(v))
                        displays = []
                        for type_ in types_to_show:
                            path = os.path.join(view_path_, type_, frame)
                            add_frame = cv2.imread(path)
                            displays.append(add_frame)
                        display = np.hstack((displays))
                        all_displays.append(display)
                    full_display = np.vstack((all_displays))
                    cv2.namedWindow(vid_dir, cv2.WINDOW_NORMAL)
                    cv2.resizeWindow(vid_dir, display_size, display_size)
                    cv2.imshow(vid_dir, full_display)
                    cv2.waitKey(display_time)

            elif 'rep' in types_to_show:
                for kp in range(2):
                    frame = make_frame_name(i + 1)[:-4] + make_frame_name(kp + 1)
                    all_displays = []
                    for v in possible_views:
                        view_path_ = os.path.join(video, str(v))
                        displays = []
                        for type_ in types_to_show:
                            path = os.path.join(view_path_, type_, frame)
                            add_frame = cv2.imread(path)
                            displays.append(add_frame)
                        display = np.hstack((displays))
                        all_displays.append(display)
                    full_display = np.vstack((all_displays))
                    cv2.namedWindow(vid_dir, cv2.WINDOW_NORMAL)
                    cv2.resizeWindow(vid_dir, display_size, display_size)
                    cv2.imshow(vid_dir, full_display)
                    cv2.waitKey(display_time)


if __name__ == '__main__':
    if batches_to_view is None or len(batches_to_view) == 0:
        batch_dirs = [os.path.join(full_dir, str(batch+1)) for batch in range(len(os.listdir(full_dir)))]
    else:
        batch_dirs = [os.path.join(full_dir, str(batch)) for batch in batches_to_view]
    vid_dirs = []
    for batch in batch_dirs:
        vids = [os.path.join(batch, str(vid+1)) for vid in range(len(os.listdir(batch)))]
        vid_dirs.extend(vids)
    view_dirs = []
    for vid in vid_dirs:
        views = [os.path.join(vid, view) for view in [str(2)]]
        view_dirs.extend(views)
    item_types = ['input', 'output', 'recon', 'rep', 'rep_est']

    # view all frames 1 by 1
    # view_all_single_frames()
    # view input gt and recon
    view_input_gt_recon()
    # view gt and recon
    # view_gt_recon()
    # view gt and recon for both views
    # view_both_gt_recon()

    print('done')
