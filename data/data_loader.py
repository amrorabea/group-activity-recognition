from torch.utils.data import Dataset, DataLoader
import torch
import cv2
from pathlib import Path
from torchvision import transforms

try:
    from data.helpers import load_tracking_annot
    from data.config import Config
except ImportError:
    from helpers import load_tracking_annot
    from config import Config

class VolleyDataset(Dataset):
    def __init__(self, data_root=Config.DATA_ROOT, tracking_root=Config.TRACKING_ROOT, 
                split='train', resize_dims=Config.RESIZE_DIMS, crop_size=(224, 224), 
                return_crops=False, transform=None, print_logs=True):
        
        self.data_root = Path(data_root)
        self.tracking_root = Path(tracking_root)
        self.split = split
        self.resize_dims = resize_dims
        self.crop_size = crop_size
        self.return_crops = return_crops
        self.transform = transform
        self.print_logs = print_logs

        self.splits = Config.SPLITS
        self.person_labels = Config.PERSON_LABELS
        self.frame_labels = Config.GROUP_LABELS

        self.samples = []
        self._build_clip_index()

    def _build_clip_index(self):
        # Only iterate over videos belonging to the current split
        target_video_ids = [str(x) for x in self.splits[self.split]]
        
        for video_id in target_video_ids:
            video_dir = self.data_root / video_id
            
            if not video_dir.exists():
                if self.print_logs:
                    print(f"Warning: Video {video_id} found in split definition but missing on disk.")
                continue

            annot_path = video_dir / 'annotations.txt'
            if not annot_path.exists():
                if self.print_logs:
                    print(f"Warning: Annotations path {annot_path} not found")
                continue

            with open(annot_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    clip_name, label_str = parts[:2]
                    clip_id = clip_name.replace(".jpg", "")

                    # Fix label typos if any
                    label_str = label_str.replace('-', '_')

                    clip_dir = video_dir / clip_id
                    if not clip_dir.exists():
                        continue
                    
                    tracking_file = (
                        self.tracking_root
                        / video_dir.name
                        / clip_id
                        / f"{clip_id}.txt"
                    )

                    if not tracking_file.exists():
                        continue

                    frame_boxes_dct = load_tracking_annot(tracking_file)
                    
                    # Ensure we have frames 
                    valid_frame_ids = sorted(frame_boxes_dct.keys())
                    frame_files = []
                    
                    # Verify images exist
                    all_frames_exist = True
                    for fid in valid_frame_ids:
                        img_path = clip_dir / f"{fid}.jpg"
                        if not img_path.exists():
                            all_frames_exist = False
                            break
                        frame_files.append(img_path)
                    
                    if not all_frames_exist or len(frame_files) == 0:
                        continue

                    self.samples.append({
                        "frames": frame_files,
                        "group_label": self.frame_labels[label_str],
                        "frame_boxes_dct": frame_boxes_dct,
                        "valid_frame_ids": valid_frame_ids
                    })
        if self.print_logs:
            print(f"Split '{self.split}': Loaded {len(self.samples)} clips from {len(target_video_ids)} videos.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        
        # 1. Prepare Storage
        T = len(s["valid_frame_ids"])
        
        # Determine dims for coordinates rescaling
        target_h, target_w = self.resize_dims

        # If returning crops, we prepare a 5D tensor: (T, 12, C, H, W)
        if self.return_crops:
            crop_h, crop_w = self.crop_size
            crops_tensor = torch.zeros((T, 12, 3, crop_h, crop_w))
            
            # Boxes still returned, resized to target_dims for spatial context in Stage 2
            boxes_tensor = torch.zeros((T, 12, 4))
            actions_tensor = torch.zeros((T, 12), dtype=torch.long)

            for t, img_path in enumerate(s["frames"]):
                img = cv2.imread(str(img_path))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                h_orig, w_orig = img.shape[:2]
                w_scale = target_w / w_orig
                h_scale = target_h / h_orig

                frame_id = s["valid_frame_ids"][t]
                box_list = sorted(s["frame_boxes_dct"][frame_id], key=lambda b: b.player_ID)
                
                for i, box_info in enumerate(box_list):
                    if i >= 12: 
                        break
                    
                    x1, y1, x2, y2 = box_info.box
                    
                    # 1. Fill Box Tensor (Rescaled for Stage 2 spatial awareness)
                    scaled_box = [x1 * w_scale, y1 * h_scale, x2 * w_scale, y2 * h_scale]
                    boxes_tensor[t, i] = torch.tensor(scaled_box)
                    actions_tensor[t, i] = self.person_labels.get(box_info.category, 0)
                    
                    # 2. Extract Crop (From original image)
                    # Clamp coordinates
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w_orig, x2), min(h_orig, y2)
                    
                    if x2 > x1 and y2 > y1:
                        crop = img[y1:y2, x1:x2]
                        
                        # Resize crop to standard size (e.g. 224x224)
                        crop = cv2.resize(crop, (crop_w, crop_h))
                        
                        if self.transform:
                            crop_t = self.transform(crop)
                        else:
                            crop_t = transforms.functional.to_tensor(crop)
                        
                        crops_tensor[t, i] = crop_t
            
            label = torch.tensor(s["group_label"], dtype=torch.long)
            # Return: Crops(T,12,3,224,224), Boxes(T,12,4), Actions(T,12), GroupLabel
            return crops_tensor, boxes_tensor, actions_tensor, label

        else:
            # ORIGINAL MODE: Return full frames
            frames_list = []
            
            # Read first frame to get scale (assuming consistent size)
            first_img = cv2.imread(str(s["frames"][0]))
            h_orig, w_orig = first_img.shape[:2]
            w_scale = target_w / w_orig
            h_scale = target_h / h_orig

            for img_path in s["frames"]:
                img = cv2.imread(str(img_path))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (target_w, target_h))

                if self.transform:
                    img = self.transform(img)
                else:
                    img = transforms.functional.to_tensor(img)
                frames_list.append(img)
            
            frames_tensor = torch.stack(frames_list) # (T, C, H, W)
            
            boxes_tensor = torch.zeros((T, 12, 4))
            actions_tensor = torch.zeros((T, 12), dtype=torch.long)

            for t, frame_id in enumerate(s["valid_frame_ids"]):
                box_list = sorted(s["frame_boxes_dct"][frame_id], key=lambda b: b.player_ID)
                
                for i, box_info in enumerate(box_list):
                    if i >= 12: 
                        break
                    
                    x1, y1, x2, y2 = box_info.box
                    scaled_box = [x1 * w_scale, y1 * h_scale, x2 * w_scale, y2 * h_scale]
                    
                    boxes_tensor[t, i] = torch.tensor(scaled_box)
                    actions_tensor[t, i] = self.person_labels.get(box_info.category, 0)

            label = torch.tensor(s["group_label"], dtype=torch.long)
            return frames_tensor, boxes_tensor, actions_tensor, label


def get_loader(split='train', batch_size=Config.BATCH_SIZE, return_crops=False):
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=Config.NORM_MEAN, std=Config.NORM_STD)
    ])

    dataset = VolleyDataset(
        data_root=Config.DATA_ROOT, 
        tracking_root=Config.TRACKING_ROOT, 
        split=split,
        resize_dims=Config.RESIZE_DIMS,
        return_crops=return_crops,
        transform=transform
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == 'train'),
        num_workers=Config.NUM_WORKERS
    )
    return loader

if __name__ == '__main__':
    # Test Train Split
    train_loader = get_loader(split='train')
    
    # Test Validation Split
    val_loader = get_loader(split='val')
