import cv2
import time

from agents.surveillance_graph import SurveillanceGraphRunner

cap = cv2.VideoCapture("data/raw_videos/sample1.mp4")

# Extract video dimensions
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

frame_id = 0

graph_runner = SurveillanceGraphRunner(
    frame_width=width,
    frame_height=height,
    model_path="models/abnormal_cnn_lstm1.pth",
)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    timestamp = time.time()

    result = graph_runner.run_frame(frame=frame, frame_id=frame_id, timestamp=timestamp)
    scene_state = result["scene_state"]

    abnormal_result = result.get("abnormal_result")
    panic_result = result.get("panic_result", {})
    risk_result = result.get("risk_result", {})
    decision_result = result.get("decision_result", {})
    sudden_run_ids = set(panic_result.get("sudden_run_ids", []))

    if abnormal_result is not None:
        print(
            "Abnormal:", abnormal_result["abnormal"],
            "| Score:", round(abnormal_result["score"], 3),
            "| Risk:", risk_result.get("risk_level"),
            "| Alerts:", decision_result.get("alerts", [])
        )

        if decision_result.get("alert", False):
            cv2.putText(
                frame,
                f"ALERT: {risk_result.get('risk_level', 'unknown').upper()}",
                (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                3
            )
        else:
            cv2.putText(
                frame,
                f"Normal | Risk {risk_result.get('risk_level', 'low').upper()}",
                (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

        risk_score = risk_result.get("risk_score", 0.0)
        cv2.putText(
            frame,
            f"Risk Score: {risk_score:.2f}",
            (50, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 0),
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

    for obj in scene_state.objects:
        x1, y1, x2, y2 = obj["bbox"]
        is_sudden_runner = (
            obj.get("class") == "person" and obj.get("track_id") in sudden_run_ids
        )
        color = (0, 0, 255) if is_sudden_runner else (0, 255, 0)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame,
            f'ID {obj["track_id"]}' + (" RUN" if is_sudden_runner else ""),
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1
        )

    cv2.imshow("Shared Processing Output", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

    frame_id += 1

cap.release()
cv2.destroyAllWindows()
