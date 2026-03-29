import os
import sys
import cv2
import numpy as np
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# import your modules
from shared_processing.detector import ObjectDetector
from shared_processing.tracker import ObjectTracker
from shared_processing.optical_flow import OpticalFlowEstimator
from shared_processing.scene_state import SceneState
from shared_processing.trajectory_manager import TrajectoryManager

from agents.abnormal_activity.heatmap_generator import HeatmapGenerator
from agents.abnormal_activity.sequence_builder import SequenceBuilder


# -------------------------
# CONFIG
# -------------------------
SEQ_LEN = 16
DATASET_PATH = r"E:\Project\dataset\UCSDped2"
SAVE_PATH = r"E:\Project\dataset"

os.makedirs(f"{SAVE_PATH}/sequences", exist_ok=True)
os.makedirs(f"{SAVE_PATH}/labels", exist_ok=True)


# -------------------------
# Initialize modules
# -------------------------
detector = ObjectDetector()
tracker = ObjectTracker()
flow_estimator = OpticalFlowEstimator()
trajectory_manager = TrajectoryManager()

# Dummy init (updated after reading first frame)
heatmap_generator = None


def process_clip(frames_dir, gt_dir=None, label_mode="train"):
    global heatmap_generator

    seq_builder = SequenceBuilder(sequence_length=SEQ_LEN)

    frames = sorted(os.listdir(frames_dir))

    sequences = []
    labels = []

    for i, fname in enumerate(frames):
        frame_path = os.path.join(frames_dir, fname)

        frame = cv2.imread(frame_path)

        if frame is None:
            continue

        if len(frame.shape) == 2 or frame.shape[2] == 1:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        # Initialize heatmap generator once
        if heatmap_generator is None:
            h, w = frame.shape[:2]
            heatmap_generator = HeatmapGenerator(w, h)

        # -------------------------
        # Shared Processing
        # -------------------------
        detections = detector.detect(frame)
        tracked = tracker.update(detections, frame)
        tracked = trajectory_manager.update(tracked)
        flow = flow_estimator.compute(frame)

        scene_state = SceneState(i, 0, tracked, flow)

        # -------------------------
        # Heatmap
        # -------------------------
        heatmap = heatmap_generator.generate(scene_state)

        # -------------------------
        # Sequence
        # -------------------------
        seq = seq_builder.update(heatmap)

        if seq is None:
            continue

        # -------------------------
        # Label
        # -------------------------
        if label_mode == "train":
            label = 0
        else:
            # check GT masks
            start_idx = i - SEQ_LEN + 1
            anomaly = False

            for j in range(start_idx, i + 1):
                gt_name = frames[j].replace(".tif", ".bmp")
                gt_path = os.path.join(gt_dir, gt_name)

                if os.path.exists(gt_path):
                    gt_img = cv2.imread(gt_path, 0)
                    if gt_img is not None and np.any(gt_img > 0):
                        anomaly = True
                        break

            label = 1 if anomaly else 0

        sequences.append(seq)
        labels.append(label)

    return sequences, labels


# -------------------------
# MAIN LOOP
# -------------------------
all_sequences = []
all_labels = []

# -------- TRAIN --------
train_path = os.path.join(DATASET_PATH, "Train")

for folder in tqdm(sorted(os.listdir(train_path)), desc="Train"):
    fpath = os.path.join(train_path, folder)
    seqs, labs = process_clip(fpath, label_mode="train")

    all_sequences.extend(seqs)
    all_labels.extend(labs)


# -------- TEST --------
test_path = os.path.join(DATASET_PATH, "Test")

for folder in tqdm(sorted(os.listdir(test_path)), desc="Test"):

    # skip GT folders
    if "_gt" in folder:
        continue

    fpath = os.path.join(test_path, folder)

    # GT is inside same Test directory
    gt_folder = os.path.join(test_path, folder + "_gt")

    seqs, labs = process_clip(fpath, gt_folder, label_mode="test")

    all_sequences.extend(seqs)
    all_labels.extend(labs)


# -------------------------
# SAVE DATA
# -------------------------
all_sequences = np.array(all_sequences)
all_labels = np.array(all_labels)

np.save(f"{SAVE_PATH}/sequences/data.npy", all_sequences)
np.save(f"{SAVE_PATH}/labels/labels.npy", all_labels)

print("Saved:", all_sequences.shape, all_labels.shape)