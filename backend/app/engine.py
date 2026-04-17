import base64
import threading
import time
from typing import Any, Dict

import cv2

from agents.surveillance_graph import SurveillanceGraphRunner


class SurveillanceEngine:
    def __init__(self, video_path: str, model_path: str):
        self.video_path = video_path
        self.model_path = model_path
        self.cap = cv2.VideoCapture(video_path)

        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open video source: {video_path}")

        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.runner = SurveillanceGraphRunner(
            frame_width=width,
            frame_height=height,
            model_path=model_path,
        )

        self.frame_id = 0
        self.latest_status: Dict[str, Any] = {
            "frame_id": 0,
            "warmup": True,
            "agents": {},
            "risk": {"risk_level": "low", "risk_score": 0.0},
            "alerts": [],
            "disaster_type": "Normal",
            "updated_at": time.time(),
        }
        self.lock = threading.Lock()

    def _infer_disaster_type(self, status: Dict[str, Any]) -> str:
        agents = status.get("agents", {})
        panic = agents.get("panic", {})
        suspicious = agents.get("suspicious_object", {})
        abnormal = agents.get("abnormal_activity", {})
        crowd = agents.get("crowd_density", {})

        if panic.get("sudden_run", False):
            return "Panic / Stampede Risk"
        if suspicious.get("detected", False):
            return "Suspicious Object"
        if abnormal.get("detected", False):
            return "Abnormal Activity"
        if crowd.get("density_level") == "high":
            return "Overcrowding"
        return "Normal"

    @staticmethod
    def _draw_overlays(frame, status: Dict[str, Any]):
        color = (0, 255, 0)
        risk = status.get("risk", {})
        alerts = status.get("alerts", [])
        if alerts:
            color = (0, 0, 255)

        cv2.putText(
            frame,
            f"Risk: {risk.get('risk_level', 'low').upper()} ({risk.get('risk_score', 0.0):.2f})",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2,
        )
        cv2.putText(
            frame,
            f"Disaster: {status.get('disaster_type', 'Normal')}",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2,
        )

    @staticmethod
    def _draw_pinpoint_boxes(
        frame,
        scene_objects,
        sudden_run_ids,
        unattended_bag_ids,
        unattended_bboxes=None,
    ):
        unattended_bboxes = unattended_bboxes or []
        for obj in scene_objects:
            track_id = obj.get("track_id")
            label = obj.get("class", "")
            x1, y1, x2, y2 = obj.get("bbox", [0, 0, 0, 0])

            if label == "person" and track_id in sudden_run_ids:
                color = (0, 0, 255)  # red
                text = f"ID {track_id} RUN"
            elif label in {"backpack", "handbag", "suitcase"} and track_id in unattended_bag_ids:
                color = (0, 140, 255)  # orange
                text = f"ID {track_id} UNATTENDED"
            else:
                continue

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame,
                text,
                (x1, max(15, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
            )

        # Draw fallback unattended detections that may not have DeepSORT tracks.
        for item in unattended_bboxes:
            track_id = item.get("track_id")
            x1, y1, x2, y2 = item.get("bbox", [0, 0, 0, 0])
            color = (0, 140, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame,
                f"ID {track_id} UNATTENDED",
                (x1, max(15, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
            )

    def read_next(self):
        with self.lock:
            ok, frame = self.cap.read()
            if not ok:
                # loop video for demo continuity
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self.frame_id = 0
                ok, frame = self.cap.read()
                if not ok:
                    return None, None

            ts = time.time()
            result = self.runner.run_frame(frame=frame, frame_id=self.frame_id, timestamp=ts)
            self.frame_id += 1

            abnormal = result.get("abnormal_result")
            panic = result.get("panic_result", {})
            crowd = result.get("crowd_result", {})
            suspicious = result.get("suspicious_result", {})
            risk = result.get("risk_result", {})
            decision = result.get("decision_result", {})
            scene_state = result.get("scene_state")
            sudden_run_ids = set(panic.get("sudden_run_ids", []))
            unattended_bag_ids = set(suspicious.get("unattended_bag_ids", []))
            unattended_bboxes = suspicious.get("unattended_bboxes", [])
            suspicious_debug = suspicious.get("debug", {})

            status = {
                "frame_id": self.frame_id,
                "warmup": abnormal is None,
                "agents": {
                    "abnormal_activity": {
                        "detected": bool(abnormal and abnormal.get("abnormal", False)),
                        "score": float(abnormal.get("score", 0.0)) if abnormal else 0.0,
                    },
                    "crowd_density": {
                        "density_level": crowd.get("density_level", "low"),
                        "crowd_count": int(crowd.get("crowd_count", 0)),
                        "score": float(crowd.get("score", 0.0)),
                    },
                    "panic": {
                        "detected": bool(panic.get("panic", False)),
                        "score": float(panic.get("score", 0.0)),
                        "sudden_run": bool(panic.get("sudden_run", False)),
                        "sudden_run_count": int(panic.get("sudden_run_count", 0)),
                        "sudden_run_ids": list(sudden_run_ids),
                    },
                    "suspicious_object": {
                        "detected": bool(suspicious.get("suspicious", False)),
                        "bag_count": int(suspicious.get("bag_count", 0)),
                        "unattended_count": int(suspicious.get("suspicious_count", 0)),
                        "unattended_bag_ids": list(unattended_bag_ids),
                        "unattended_bboxes": unattended_bboxes,
                        "score": float(suspicious.get("score", 0.0)),
                        "debug": suspicious_debug,
                    },
                },
                "risk": {
                    "risk_level": risk.get("risk_level", "low"),
                    "risk_score": float(risk.get("risk_score", 0.0)),
                },
                "alerts": decision.get("alerts", []),
                "updated_at": ts,
            }
            status["disaster_type"] = self._infer_disaster_type(status)
            self.latest_status = status

            if scene_state is not None:
                self._draw_pinpoint_boxes(
                    frame=frame,
                    scene_objects=scene_state.objects,
                    sudden_run_ids=sudden_run_ids,
                    unattended_bag_ids=unattended_bag_ids,
                    unattended_bboxes=unattended_bboxes,
                )
            self._draw_overlays(frame, status)
            return frame, status

    def mjpeg_generator(self):
        while True:
            frame, _ = self.read_next()
            if frame is None:
                break
            ok, jpeg = cv2.imencode(".jpg", frame)
            if not ok:
                continue
            payload = (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n"
            )
            yield payload
            time.sleep(0.03)

    def get_latest_status(self):
        with self.lock:
            return self.latest_status

    def get_snapshot_base64(self):
        frame, status = self.read_next()
        if frame is None:
            return {"image": None, "status": status}
        ok, jpeg = cv2.imencode(".jpg", frame)
        if not ok:
            return {"image": None, "status": status}
        return {
            "image": base64.b64encode(jpeg.tobytes()).decode("utf-8"),
            "status": status,
        }
