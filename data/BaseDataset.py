from torch.utils.data import Dataset
import cv2
import pickle
from pathlib import Path
from typing import Dict
import numpy as np

try:
    from data.config import Config
except ImportError:
    from config import Config


class BaseDataset(Dataset):
    
    def __init__(self, data_root: str, split: str,
                labels: Dict[str, int], pkl_path=None, transform=None, print_logs=True):
        self.data_root = Path(data_root)
        self.split = split
        self.labels = labels
        self.transform = transform
        self.print_logs = print_logs

        self.splits = Config.SPLITS
        
        if pkl_path is None:
            pkl_path = self.data_root.parent / "annot_all.pkl"
        
        with open(pkl_path, 'rb') as f:
            self.videos_annot = pickle.load(f)
        
        self.samples = []
        self._build_index()

    def _build_index(self):
        raise NotImplementedError("Child classes must implement _build_index")

    def _load_frame(self, frame_path: Path) -> np.ndarray:
        img = cv2.imread(str(frame_path))
        if img is None:
            raise FileNotFoundError(f"Failed to load: {frame_path}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    def __len__(self):
        return len(self.samples)