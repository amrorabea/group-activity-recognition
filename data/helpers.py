import cv2
import os
import pickle
from typing import List

class BoxInfo:
    def __init__(self, line):
        words = line.split()
        self.category = words.pop()
        words = [int(string) for string in words]
        self.player_ID = words[0]
        del words[0]

        x1, y1, x2, y2, frame_ID, lost, grouping, generated = words
        self.box = x1, y1, x2, y2
        self.frame_ID = frame_ID
        self.lost = lost
        self.grouping = grouping
        self.generated = generated


def load_tracking_annot(path):
    with open(path, 'r') as file:
        player_boxes = {idx:[] for idx in range(12)}
        frame_boxes_dct = {}

        for idx, line in enumerate(file):
            box_info = BoxInfo(line)
            if box_info.player_ID > 11:
                continue
            player_boxes[box_info.player_ID].append(box_info)

        # let's create view from frame to boxes
        for player_ID, boxes_info in player_boxes.items():
            # let's keep the middle 9 frames only (enough for this task empirically)
            boxes_info = boxes_info[5:]
            boxes_info = boxes_info[:-6]

            for box_info in boxes_info:
                if box_info.frame_ID not in frame_boxes_dct:
                    frame_boxes_dct[box_info.frame_ID] = []

                frame_boxes_dct[box_info.frame_ID].append(box_info)

        return frame_boxes_dct

def vis_clip(annot_path, video_dir):
    frame_boxes_dct = load_tracking_annot(annot_path)
    font = cv2.FONT_HERSHEY_SIMPLEX

    for frame_id, boxes_info in sorted(frame_boxes_dct.items()):
        img_path = os.path.join(video_dir, f'{frame_id}.jpg')
        if not os.path.exists(img_path):
            continue
        image = cv2.imread(img_path)

        for box_info in boxes_info:
            x1, y1, x2, y2 = box_info.box

            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, box_info.category, (x1, y1 - 10), font, 0.5, (0, 255, 0), 2)

        cv2.imshow('Image', image)
        if cv2.waitKey(180) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()


def load_video_annot(video_annot):
    with open(video_annot, 'r') as file:
        clip_category_dct = {}

        for line in file:
            items = line.strip().split(' ')[:2]
            clip_dir = items[0].replace('.jpg', '')
            clip_category_dct[clip_dir] = items[1]

        return clip_category_dct


def load_volleyball_dataset(videos_root, annot_root):
    videos_dirs = os.listdir(videos_root)
    videos_dirs.sort()

    videos_annot = {}

    # Iterate on each video and for each video iterate on each clip
    for idx, video_dir in enumerate(videos_dirs):
        video_dir_path = os.path.join(videos_root, video_dir)

        if not os.path.isdir(video_dir_path):
            continue

        print(f'{idx}/{len(videos_dirs)} - Processing Dir {video_dir_path}')

        video_annot = os.path.join(video_dir_path, 'annotations.txt')
        if not os.path.exists(video_annot):
            continue
            
        clip_category_dct = load_video_annot(video_annot)

        clips_dir = os.listdir(video_dir_path)
        clips_dir.sort()

        clip_annot = {}

        for clip_dir in clips_dir:
            clip_dir_path = os.path.join(video_dir_path, clip_dir)

            if not os.path.isdir(clip_dir_path):
                continue

            if clip_dir not in clip_category_dct:
                continue

            annot_file = os.path.join(annot_root, video_dir, clip_dir, f'{clip_dir}.txt')
            if not os.path.exists(annot_file):
                continue
                
            frame_boxes_dct = load_tracking_annot(annot_file)

            clip_annot[clip_dir] = {
                'category': clip_category_dct[clip_dir],
                'frame_boxes_dct': frame_boxes_dct
            }

        videos_annot[video_dir] = clip_annot

    return videos_annot


def create_pkl_version(dataset_root):
    # You can use this function to create and save pkl version of the dataset
    videos_root = os.path.join(dataset_root, 'videos')
    annot_root = os.path.join(dataset_root, 'volleyball_tracking_annotation')

    videos_annot = load_volleyball_dataset(videos_root, annot_root)

    with open(os.path.join(dataset_root, 'annot_all.pkl'), 'wb') as file:
        pickle.dump(videos_annot, file)


def test_pkl_version(dataset_root):
    pkl_path = os.path.join(dataset_root, 'annot_all.pkl')
    if not os.path.exists(pkl_path):
        print(f"Pickle file not found: {pkl_path}")
        return

    with open(pkl_path, 'rb') as file:
        videos_annot = pickle.load(file)

    # Example access, assuming these keys exist in your data
    try:
        # Accessing just to test structure
        first_video = list(videos_annot.keys())[0]
        first_clip = list(videos_annot[first_video].keys())[0]
        frame_boxes_dct = videos_annot[first_video][first_clip]['frame_boxes_dct']
        first_frame = list(frame_boxes_dct.keys())[0]
        
        boxes: List[BoxInfo] = frame_boxes_dct[first_frame]
        print(f"Loaded successfully. First box category: {boxes[0].category}")
    except Exception as e:
        print(f"Error while testing pkl version: {e}")
