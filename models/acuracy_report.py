import os
import csv
import cv2
import numpy as np
from mediapipe.tasks.python import vision
from mediapipe.tasks import python
from mediapipe import Image, ImageFormat

# NOTE TO SELF:
# Change this path if the dataset moves.
# The folder must contain one subfolder per gesture class.
DATASET_PATH = "/Users/kush/Documents/SFU_CMPT/FALL2025/Gesture_Project/cmpt310/cmpt310.git/cmpt310/set_data/Hagrid_data"

# NOTE TO SELF:
# All CSV files will be saved here.
OUTPUT_DIR = "/Users/kush/Documents/SFU_CMPT/FALL2025/Gesture_Project"

# NOTE TO SELF:
# The left side MUST match folder names exactly.
# The right side MUST match model label names exactly.
FOLDER_TO_LABEL = {
    "fingers_up_volume_up": "fingers_up_volume_up",
    "fist_mute": "fist_mute",
    "palm_pause_play": "palm_pause_play",
    "thumbs_down_dislike": "thumbs_down_dislike",
    "thumbs_up_like": "thumbs_up_like"
}

LABELS = list(FOLDER_TO_LABEL.values())

# NOTE TO SELF:
# Increase to test more images per class.
SAMPLES_PER_CLASS = 60

# NOTE TO SELF:
# Change if the gesture_recognizer.task file is moved.
MODEL_PATH = "/Users/kush/Documents/SFU_CMPT/FALL2025/Gesture_Project/cmpt310/cmpt310.git/cmpt310/models/gesture_recognizer.task"


def load_image_paths(folder, max_count):
    valid_ext = (".jpg", ".jpeg", ".png")
    files = sorted([f for f in os.listdir(folder) if f.lower().endswith(valid_ext)])
    return [os.path.join(folder, f) for f in files[:max_count]]


def create_matrix(n):
    return [[0 for _ in range(n)] for _ in range(n)]


def compute_metrics(cm):
    precision = []
    recall = []
    f1 = []
    num_classes = len(cm)

    for i in range(num_classes):
        tp = cm[i][i]
        fp = sum(cm[r][i] for r in range(num_classes) if r != i)
        fn = sum(cm[i][c] for c in range(num_classes) if c != i)

        p = tp / (tp + fp) if tp + fp > 0 else 0
        r = tp / (tp + fn) if tp + fn > 0 else 0
        f = 2 * p * r / (p + r) if p + r > 0 else 0

        precision.append(p)
        recall.append(r)
        f1.append(f)

    return precision, recall, f1


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)


def main():
    base = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.GestureRecognizerOptions(base_options=base)
    recognizer = vision.GestureRecognizer.create_from_options(options)

    cm = create_matrix(len(LABELS))
    total = 0
    correct = 0

    # NOTE TO SELF:
    # Each folder is tested independently.
    # If you see 0 samples counted, check folder names and paths.
    for folder, real_label in FOLDER_TO_LABEL.items():
        folder_path = os.path.join(DATASET_PATH, folder)
        paths = load_image_paths(folder_path, SAMPLES_PER_CLASS)

        for img_path in paths:
            img = cv2.imread(img_path)
            if img is None:
                continue

            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            mp_img = Image(image_format=ImageFormat.SRGB, data=rgb)
            result = recognizer.recognize(mp_img)

            predicted = result.gestures[0][0].category_name if result.gestures else "none"

            if predicted in LABELS:
                true_idx = LABELS.index(real_label)
                pred_idx = LABELS.index(predicted)

                cm[true_idx][pred_idx] += 1

                if predicted == real_label:
                    correct += 1

                total += 1

    precision, recall, f1 = compute_metrics(cm)
    overall_accuracy = correct / total if total > 0 else 0

    # SUMMARY CSV

    summary_rows = [
        ["Metric", "Value"],
        ["Total Samples", total],
        ["Correct Predictions", correct],
        ["Overall Accuracy (%)", round(overall_accuracy * 100, 2)],
    ]
    write_csv(f"{OUTPUT_DIR}/gesture_summary.csv", summary_rows)

    # CONFUSION MATRIX CSV

    cm_rows = [[""] + LABELS]
    for i, label in enumerate(LABELS):
        cm_rows.append([label] + cm[i])

    write_csv(f"{OUTPUT_DIR}/gesture_confusion_matrix.csv", cm_rows)

    # CLASS METRICS CSV

    metrics_rows = [["Class", "Precision", "Recall", "F1 Score"]]

    for i, label in enumerate(LABELS):
        metrics_rows.append([
            label,
            round(precision[i], 4),
            round(recall[i], 4),
            round(f1[i], 4)
        ])

    metrics_rows.append(["Overall Accuracy", round(overall_accuracy, 4), "", ""])

    write_csv(f"{OUTPUT_DIR}/gesture_class_metrics.csv", metrics_rows)

    print("\nSaved CSV reports:")
    print(f"- {OUTPUT_DIR}/gesture_summary.csv")
    print(f"- {OUTPUT_DIR}/gesture_confusion_matrix.csv")
    print(f"- {OUTPUT_DIR}/gesture_class_metrics.csv\n")


if __name__ == "__main__":
    main()
