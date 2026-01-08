from torch.utils.data import DataLoader
import torch
import cv2
import sys
from pathlib import Path
from torchvision import transforms
from typing import Tuple
import numpy as np
import albumentations as A


try:
    from data.BaseDataset import BaseDataset
    from data.config import Config
    sys.path.insert(0, str(Path(__file__).parent / "ref_scripts"))
    from boxinfo import BoxInfo
except ImportError:
    from BaseDataset import BaseDataset
    from config import Config
    sys.path.insert(0, str(Path(__file__).parent / "ref_scripts"))
    from boxinfo import BoxInfo


class GroupDataset(BaseDataset):

    def __init__(self, data_root=Config.DATA_ROOT, split='train',
                labels=Config.GROUP_LABELS, resize_dims=(224, 224),
                crop_size=(224, 224), seq=False, crops=False, 
                only_target=False, pkl_path=None, transform=None, print_logs=True):
        self.resize_dims = resize_dims
        self.crop_size = crop_size
        self.seq = seq
        self.crops = crops
        self.only_target = only_target
        super().__init__(data_root, split, labels, pkl_path, transform, print_logs)


    def _build_index(self):
        target_video_ids = [str(x) for x in self.splits[self.split]]

        for video_id in target_video_ids:
            if video_id not in self.videos_annot:
                if self.print_logs:
                    print(f"Video {video_id} not found in PKL")
                continue
            
            video_clips = self.videos_annot[video_id]

            for clip_id, clip_data in video_clips.items():
                group_label = clip_data['category'].replace('-', '_')
                
                if group_label not in self.labels:
                    if self.print_logs:
                        print(f"Warning: Unknown label '{group_label}'")
                    continue
                
                frame_boxes_dct = clip_data['frame_boxes_dct']
                frame_ids = sorted(frame_boxes_dct.keys())
                
                if self.seq:
                    # Store entire clip
                    self.samples.append({
                        'video_id': video_id,
                        'clip_id': clip_id,
                        'group_label': group_label,
                        'frame_ids': frame_ids,
                        'frame_boxes_dct': frame_boxes_dct
                    })
                else:
                    middle_idx = len(frame_ids) // 2
                    frame_id = frame_ids[middle_idx]
                    
                    frame_path = self.data_root / video_id / clip_id / f"{frame_id}.jpg"
                    
                    if not frame_path.exists():
                        continue
                    
                    # Get boxes for this frame
                    boxes = []
                    for box_info in frame_boxes_dct[frame_id]:
                        x1, y1, x2, y2 = box_info.box
                        boxes.append({
                            'box': (x1, y1, x2 - x1, y2 - y1),
                            'action': box_info.category
                        })
                    
                    self.samples.append({
                        'frame_path': frame_path,
                        'group_label': group_label,
                        'boxes': boxes
                    })
    
        if self.print_logs:
            print(f"GroupDataset '{self.split}': {len(self.samples)} samples")

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[idx]
        label = torch.tensor(self.labels[sample['group_label']], dtype=torch.long)

        # B1: Single full frame
        if not self.crops and not self.seq:
            frame = self._load_frame(sample['frame_path'])
            frame = cv2.resize(frame, self.resize_dims)

            if self.transform:
                if isinstance(self.transform, A.Compose):
                    frame = self.transform(image=frame)['image']
                else:
                    frame = self.transform(frame)
            else:
                frame = transforms.functional.to_tensor(frame)
            
            return frame, label

        # B4: Sequence of full frames
        elif not self.crops and self.seq:
            frames = []
            video_id = sample['video_id']
            clip_id = sample['clip_id']
            
            for frame_id in sample['frame_ids']:
                frame_path = self.data_root / video_id / clip_id / f"{frame_id}.jpg"
                frame = self._load_frame(frame_path)
                frame = cv2.resize(frame, self.resize_dims)

                if self.transform:
                    if isinstance(self.transform, A.Compose):
                        frame = self.transform(image=frame)['image']
                    else:
                        frame = self.transform(frame)
                else:
                    frame = transforms.functional.to_tensor(frame)
                
                frames.append(frame)
            
            return torch.stack(frames), label
        
        # B3: Person crops from single frame
        elif self.crops and not self.seq:
            frame = self._load_frame(sample['frame_path'])
            crops = [self._extract_crop(frame, box['box']) for box in sample['boxes']]
            return torch.stack(crops), label
        
        # B6: Sequence of person crops
        else:
            all_crops = []
            video_id = sample['video_id']
            clip_id = sample['clip_id']
            frame_boxes_dct = sample['frame_boxes_dct']
            
            for frame_id in sample['frame_ids']:
                frame_path = self.data_root / video_id / clip_id / f"{frame_id}.jpg"
                frame = self._load_frame(frame_path)
                
                frame_crops = []
                for box_info in frame_boxes_dct[frame_id]:
                    x1, y1, x2, y2 = box_info.box
                    box = (x1, y1, x2 - x1, y2 - y1)
                    frame_crops.append(self._extract_crop(frame, box))
                
                all_crops.append(torch.stack(frame_crops))
            
            crops_tensor = torch.stack(all_crops).permute(1, 0, 2, 3, 4)
            return crops_tensor, label

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
    

def get_group_loader(split='train', batch_size=4, seq=False, crops=False, pkl_path=None, **kwargs):
    
    if split == 'train':
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 256)),
            transforms.RandomCrop((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(mean=Config.NORM_MEAN, std=Config.NORM_STD)
        ])
    else:
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 256)),
            transforms.CenterCrop((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=Config.NORM_MEAN, std=Config.NORM_STD)
        ])
    
    dataset = GroupDataset(split=split, seq=seq, crops=crops, pkl_path=pkl_path, transform=transform, **kwargs)
    return DataLoader(dataset, batch_size=batch_size, shuffle=(split=='train'),
                     num_workers=Config.NUM_WORKERS, pin_memory=True)


if __name__ == "__main__":
    pkl_path = "/home/amro/Desktop/group-activity-recognition/data/annot_all.pkl"
    
    print("Testing B1 (Group, single frame)...")
    loader = get_group_loader(split='train', batch_size=2, only_target=True, pkl_path=pkl_path)
    frames, labels = next(iter(loader))
    print(f"B1: frames={frames.shape}, labels={labels.shape}\n")