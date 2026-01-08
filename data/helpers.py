import cv2
import os

from sklearn.metrics import confusion_matrix, classification_report, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

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


def compute_metrics(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    
    # Per-class accuracy
    per_class_acc = cm.diagonal() / cm.sum(axis=1)
    mean_class_acc = per_class_acc.mean() * 100
    
    print("\n" + "="*60)
    print("EVALUATION METRICS")
    print("="*60)
    
    print(f"\nMean Per-Class Accuracy (MCA): {mean_class_acc:.2f}%")
    print("\nPer-Class Accuracy:")
    for i, (name, acc) in enumerate(zip(class_names, per_class_acc)):
        print(f"  {name:15s}: {acc*100:.2f}%")
    
    print("\n" + classification_report(y_true, y_pred, target_names=class_names, digits=4))
    
    return cm, mean_class_acc


def plot_confusion_matrix(cm, class_names, save_path='b1_confusion_matrix.png'):
    """Plot and save confusion matrix"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('B1: Confusion Matrix (Group Activity Recognition)')
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"\nConfusion matrix saved to: {save_path}")
    plt.close()