import cv2
import time

from shared_processing.detector import ObjectDetector
from shared_processing.tracker import ObjectTracker
from shared_processing.optical_flow import OpticalFlowEstimator
from shared_processing.scene_state import SceneState
from agents.abnormal_activity.abnormal_activity_agent import AbnormalActivityAgent
from shared_processing.trajectory_manager import TrajectoryManager

cap = cv2.VideoCapture("data/raw_videos/sample1.mp4")

# Extract video dimensions
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

detector = ObjectDetector()
tracker = ObjectTracker()
trajectory_manager = TrajectoryManager()
flow_estimator = OpticalFlowEstimator()

frame_id = 0

abnormal_agent = AbnormalActivityAgent(
    frame_width=width,
    frame_height=height,
    model_path="models/abnormal_cnn_lstm1.pth"
)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    timestamp = time.time()

    detections = detector.detect(frame)
    tracked_objects = tracker.update(detections, frame)
    tracked_objects = trajectory_manager.update(tracked_objects)
    flow = flow_estimator.compute(frame)

    scene_state = SceneState(
        frame_id=frame_id,
        timestamp=timestamp,
        objects=tracked_objects,
        optical_flow=flow
    )

    result = abnormal_agent.process(scene_state)

    if result is not None:
        print("Abnormal:", result["abnormal"], "Score:", result["score"])
        if result["abnormal"]:
            cv2.putText(
                frame,
                f"ABNORMAL ({result['score']:.2f})",
                (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),  # Red
                3
            )
        else:
            cv2.putText(
                frame,
                f"Normal ({result['score']:.2f})",
                (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),  # Green
                2
            )
    else:
        print("Warming up model sequence...")
        cv2.putText(
            frame,
            "Warming up model...",
            (50, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 255),
            2
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
        # print(obj["track_id"], obj["speed"])

    cv2.imshow("Shared Processing Output", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

    frame_id += 1

cap.release()
cv2.destroyAllWindows()
