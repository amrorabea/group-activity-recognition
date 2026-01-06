import os
from config import Config
from helpers import vis_clip

def main():
    # --- Configuration ---
    VIDEO_ID = "45"
    CLIP_ID = "19050"
    # ---------------------

    # Construct paths dynamically based on Config
    annot_path = os.path.join(
        Config.TRACKING_ROOT, 
        VIDEO_ID, 
        CLIP_ID, 
        f"{CLIP_ID}.txt"
    )

    video_dir = os.path.join(
        Config.DATA_ROOT, 
        VIDEO_ID, 
        CLIP_ID
    )

    print(f"Visualizing Clip: {CLIP_ID} from Video: {VIDEO_ID}")
    print(f"Annotation: {annot_path}")
    print(f"Video Dir:  {video_dir}")

    # Validation
    if not os.path.exists(annot_path):
        print(f"Error: Annotation file not found at {annot_path}")
        return
    if not os.path.exists(video_dir):
        print(f"Error: Video directory not found at {video_dir}")
        return

    print("Press 'q' in the window to quit")
    vis_clip(annot_path=annot_path, video_dir=video_dir)

if __name__ == "__main__":
    main()