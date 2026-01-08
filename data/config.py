class Config:
    # --- Paths ---
    DATA_ROOT = "/home/amro/Desktop/group-activity-recognition/data/videos"
    PKL_PATH = "/home/amro/Desktop/group-activity-recognition/data/annot_all.pkl"
    # TRACKING_ROOT = "/home/amro/Desktop/group-activity-recognition/data/volleyball_tracking_annotation"
    
    # --- Image Processing ---
    RESIZE_DIMS = (224, 224)
    
    # ImageNet Normalization
    NORM_MEAN = [0.485, 0.456, 0.406]
    NORM_STD = [0.229, 0.224, 0.225]

    # --- Training Params ---
    BATCH_SIZE = 32
    NUM_WORKERS = 4
    EPOCHS = 20
    LR = 3e-4
    WEIGHT_DECAY = 1e-4
    DROPOUT = 0.5
    SAVE = True

    # Scheduler settings
    SCHEDULER = 'cosine'  # 'cosine', 'plateau', or 'onecycle'
    T_MAX = 20
    
    PATIENCE = 7
    MIN_DELTA = 0.005

    # --- Dataset Splits ---
    SPLITS = {
        'train': [1, 3, 6, 7, 10, 13, 15, 16, 18, 22, 23, 31, 32, 36, 38, 39, 40, 41, 42, 48, 50, 52, 53, 54],
        'val':   [0, 2, 8, 12, 17, 19, 24, 26, 27, 28, 30, 33, 46, 49, 51],
        'test':  [4, 5, 9, 11, 14, 20, 21, 25, 29, 34, 35, 37, 43, 44, 45, 47]
    }

    # --- Index to Label Maps ---
    PERSON_LABELS = {
        'blocking': 0, 'digging': 1, 'falling': 2, 'jumping': 3,
        'moving': 4, 'setting': 5, 'spiking': 6, 'standing': 7, 'waiting': 8
    }

    GROUP_LABELS = {
        'l_pass': 0, 'r_pass': 1, 'l_spike': 2, 'r_spike': 3,
        'l_set': 4, 'r_set': 5, 'l_winpoint': 6, 'r_winpoint': 7
    }