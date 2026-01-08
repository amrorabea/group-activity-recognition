from torch.utils.data import DataLoader
import torch
import cv2
import sys
from pathlib import Path
from torchvision import transforms
from typing import Tuple
import numpy as np
import albumentations as A
from BaseDataset import BaseDataset

try:
    from data.config import Config
    sys.path.insert(0, str(Path(__file__).parent / "ref_scripts"))
    from boxinfo import BoxInfo
except ImportError:
    from config import Config
    sys.path.insert(0, str(Path(__file__).parent / "ref_scripts"))
    from boxinfo import BoxInfo


class PersonDataset(BaseDataset):

    def __init__(self, data_root=Config.DATA_ROOT, split='train',
                 labels=Config.PERSON_LABELS, crop_size=(224, 224),
                 seq=False, only_target=False, pkl_path=None, transform=None, print_logs=True):
        self.crop_size = crop_size
        self.seq = seq
        self.only_target = only_target
        super().__init__(data_root, split, labels, pkl_path, transform, print_logs)


    def _build_index(self):
        target_video_ids = [str(x) for x in self.splits[self.split]]

        for video_id in target_video_ids:
            if video_id not in self.videos_annot:
                if self.print_logs:
                    print(f"Warning: Video {video_id} not found in PKL")
                continue

            video_clips = self.videos_annot[video_id]

            for clip_id, clip_data in video_clips.items():
                frame_boxes_dct = clip_data['frame_boxes_dct']

                if self.seq:
                    self.samples.append({
                        'video_id': video_id,
                        'clip_id': clip_id,
                        'frame_boxes_dct': frame_boxes_dct
                    })
                else:
                    frames_ids = sorted(frame_boxes_dct.keys())
                    
                    # Select frames based on only_target flag
                    if self.only_target:
                        selected_frames = [frames_ids[len(frames_ids) // 2]]
                    else:
                        selected_frames = frames_ids

                    for frame_id in selected_frames:
                        frame_path = self.data_root / video_id / clip_id / f"{frame_id}.jpg"

                        if not frame_path.exists():
                            continue

                        for box_info in frame_boxes_dct[frame_id]:
                            x1, y1, x2, y2 = box_info.box
                            action = box_info.category

                            self.samples.append({
                                'frame_path': frame_path,
                                'box': (x1, y1, x2 - x1, y2 - y1),
                                'action': action
                            })
        
        if self.print_logs:
            print(f"PersonDataset '{self.split}': {len(self.samples)} samples")

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[idx]

        if self.seq:
            all_crops = []
            all_labels = []

            video_id = sample['video_id']
            clip_id = sample['clip_id']
            frame_boxes_dct = sample['frame_boxes_dct']
            
            for frame_id in sorted(frame_boxes_dct.keys()):
                frame_path = self.data_root / video_id / clip_id / f"{frame_id}.jpg"
                frame = self._load_frame(frame_path=frame_path)
                frame_crops = []
                frame_labels = []

                for box_info in frame_boxes_dct[frame_id]:
                    x1, y1, x2, y2 = box_info.box
                    box = (x1, y1, x2 - x1, y2 - y1)  # Convert to (x, y, w, h)
                    crop = self._extract_crop(frame, box)
                    frame_crops.append(crop)
                    frame_labels.append(self.labels.get(box_info.category, 0))
                
                all_crops.append(torch.stack(frame_crops))
                all_labels.append(torch.tensor(frame_labels, dtype=torch.long))
            
            crops_tensor = torch.stack(all_crops).permute(1, 0, 2, 3, 4)
            labels_tensor = torch.stack(all_labels).permute(1, 0)
            return crops_tensor, labels_tensor
        else:
            frame = self._load_frame(sample['frame_path'])
            crop = self._extract_crop(frame, sample['box'])
            label = torch.tensor(self.labels.get(sample['action'], 0), dtype=torch.long)
            return crop, label

    def _extract_crop(self, frame: np.ndarray, box: Tuple[int, int, int, int]) -> torch.Tensor:
        x, y, w, h = box
        h_img, w_img = frame.shape[:2]

        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(w_img, x + w), min(h_img, y + h)

        if x2 <= x1 or y2 <= y1:
            crop = torch.zeros((3, *self.crop_size))
        else:
            crop = frame[y1:y2, x1:x2]
            crop = cv2.resize(crop, self.crop_size)

            if self.transform:
                if isinstance(self.transform, A.Compose):
                    crop = self.transform(image=crop)['image']
                else:
                    crop = self.transform(crop)
            else:
                crop = transforms.functional.to_tensor(crop)
        
        return crop


# Helper functions
def get_person_loader(split='train', batch_size=1, seq=False, pkl_path=None, **kwargs):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=Config.NORM_MEAN, std=Config.NORM_STD)
    ])
    
    dataset = PersonDataset(split=split, seq=seq, pkl_path=pkl_path, transform=transform, **kwargs)
    return DataLoader(dataset, batch_size=batch_size, shuffle=(split=='train'), 
                     num_workers=Config.NUM_WORKERS, pin_memory=True)



if __name__ == "__main__":
    pkl_path = "/home/amro/Desktop/group-activity-recognition/data/annot_all.pkl"
    
    print("Testing B2 (Person, single frame)...")
    loader = get_person_loader(split='train', batch_size=12, only_target=True, pkl_path=pkl_path)
    crops, labels = next(iter(loader))
    print(f"B2: crops={crops.shape}, labels={labels.shape}\n")