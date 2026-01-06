# Group Activity Recognition in Volleyball

This project focuses on implementing deep learning models for recognizing group activities in volleyball videos. It follows the methodology inspired by hierarchical temporal models, where individual player actions are recognized first, and then aggregated to classify the overall group activity.

## Project Goal
The main objective is to build a two-stage deep learning framework:
1.  **Person-Level Action Recognition**: Recognizing what each player is doing (e.g., Jumping, Spiking, Falling).
2.  **Group-Level Activity Recognition**: Recognizing the collective activity of the team (e.g., Left Spike, Right Set) based on the temporal dynamics of individual players.

The dataset used is the **Volleyball Dataset**, which contains annotated clips of volleyball games.

---

## Current Implementation Status

### 1. Data Configuration (`data/config.py`)
We have established a centralized configuration file to manage all project constants and hyperparameters.
-   **Paths**: Centralized management of dataset roots and annotation paths.
-   **Splits**: Implemented the **Official CVPR 2016 Splits**. Training, validation, and testing are strictly separated by Video ID to ensure valid evaluation.
-   **Scales & Norms**: Standardized image resizing (720p for full frame, 224p for classification) and ImageNet normalization stats.
-   **Labels**: Mappings for 9 individual action classes and 8 group activity classes.

### 2. Data Loading (`data/data_loader.py`)
A robust PyTorch `Dataset` and `DataLoader` pipeline has been implemented:
-   **VolleyDataset**: A custom Dataset class that parses the complex directory structure of the Volleyball dataset.
-   **Synchronization**: It automatically loads video frames and synchronizes them with tracking annotations (`.txt` files).
-   **Filtering**: Implements logic to select the relevant middle frames of a clip (temporal window) as originally proposed in the dataset benchmarks.
-   **Tensorization**: Converts raw images and box coordinates into structured PyTorch tensors:
    -   `frames`: $(B, T, C, H, W)$
    -   `boxes`: $(B, T, 12, 4)$ (12 players max)
    -   `actions`: $(B, T, 12)$
    -   `group_label`: $(B,)$

### 3. Helpers & Visualization (`data/helpers.py`, `visualize_clip.py`)
-   **Annotation Parsing**: Efficient parsing of the tracking annotation files.
-   **Visualization Tool**: A standalone script (`visualize_clip.py`) to debug and verify data integrity. It overlays bounding boxes and action labels onto video frames, allowing visual inspection of the ground truth data.

---

## Folder Structure
```
.
├── data/
│   ├── config.py           # Configuration constants (Paths, Splits, Params)
│   ├── data_loader.py      # PyTorch Dataset implementation
│   ├── helpers.py          # Annotation parsing and plotting utils
│   └── visualize_clip.py   # Script to visualize ground truth on clips
├── training/
│   └── ...                 # Training scripts
└── README.md
```
