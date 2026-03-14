import cv2
import time

from shared_processing.detector import ObjectDetector
from shared_processing.tracker import ObjectTracker
from shared_processing.optical_flow import OpticalFlowEstimator
from shared_processing.scene_state import SceneState

cap = cv2.VideoCapture("data/raw_videos/sample1.mp4")

detector = ObjectDetector()
tracker = ObjectTracker()
flow_estimator = OpticalFlowEstimator()

frame_id = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    timestamp = time.time()

    detections = detector.detect(frame)
    tracked_objects = tracker.update(detections, frame)
    flow = flow_estimator.compute(frame)

    scene_state = SceneState(
        frame_id=frame_id,
        timestamp=timestamp,
        objects=tracked_objects,
        optical_flow=flow
    )

    # TEMP VISUAL CHECK
    for obj in scene_state.objects:
        x1, y1, x2, y2 = obj["bbox"]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame,
            f'ID {obj["track_id"]}',
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1
        )

    cv2.imshow("Shared Processing Output", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

    frame_id += 1

cap.release()
cv2.destroyAllWindows()
