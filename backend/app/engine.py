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

        source_fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 0.0)
        self.stream_fps = source_fps if source_fps > 1 else 25.0
        self.stream_fps = min(self.stream_fps, 30.0)
        self.inference_fps = 6.0
        self.stream_interval = 1.0 / self.stream_fps
        self.inference_interval = 1.0 / self.inference_fps

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
        self.latest_jpeg: bytes | None = None
        self.latest_stream_frame_id = -1
        self.latest_raw_frame = None
        self.latest_raw_frame_id = -1
        self.overlay_state: Dict[str, Any] = {
            "scene_objects": [],
            "sudden_run_ids": set(),
            "unattended_bag_ids": set(),
            "unattended_bboxes": [],
            "abnormal_detected": False,
        }

        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.stream_thread: threading.Thread | None = None
        self.inference_thread: threading.Thread | None = None

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
                color = (0, 0, 255)
                text = f"ID {track_id} RUN"
            elif label in {"backpack", "handbag", "suitcase"} and track_id in unattended_bag_ids:
                color = (0, 140, 255)
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

    @staticmethod
    def _draw_abnormal_boxes(frame, scene_objects, abnormal_detected, sudden_run_ids):
        if not abnormal_detected:
            return

        people = [obj for obj in scene_objects if obj.get("class") == "person"]
        if not people:
            return

        prioritized = [
            obj
            for obj in people
            if obj.get("track_id") not in sudden_run_ids and float(obj.get("speed", 0.0)) >= 2.0
        ]
        if not prioritized:
            prioritized = [obj for obj in people if obj.get("track_id") not in sudden_run_ids]
        if not prioritized:
            prioritized = people

        prioritized.sort(key=lambda obj: float(obj.get("speed", 0.0)), reverse=True)
        highlighted = prioritized[: min(3, len(prioritized))]
        if not highlighted:
            return

        color = (0, 255, 255)
        for obj in highlighted:
            track_id = obj.get("track_id")
            x1, y1, x2, y2 = obj.get("bbox", [0, 0, 0, 0])
            speed = float(obj.get("speed", 0.0))
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame,
                f"ID {track_id} ABNORMAL {speed:.1f}",
                (x1, max(15, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
            )

        x1 = min(obj["bbox"][0] for obj in highlighted)
        y1 = min(obj["bbox"][1] for obj in highlighted)
        x2 = max(obj["bbox"][2] for obj in highlighted)
        y2 = max(obj["bbox"][3] for obj in highlighted)
        pad = 14
        height, width = frame.shape[:2]
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(width - 1, x2 + pad)
        y2 = min(height - 1, y2 + pad)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame,
            "ABNORMAL ACTIVITY REGION",
            (x1, max(20, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )

    def _parse_inference_result(self, result: Dict[str, Any], frame_id: int, ts: float):
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
        abnormal_detected = bool(abnormal and abnormal.get("abnormal", False))

        status = {
            "frame_id": frame_id + 1,
            "warmup": abnormal is None,
            "agents": {
                "abnormal_activity": {
                    "detected": abnormal_detected,
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

        overlay_state = {
            "scene_objects": scene_state.objects if scene_state is not None else [],
            "sudden_run_ids": sudden_run_ids,
            "unattended_bag_ids": unattended_bag_ids,
            "unattended_bboxes": unattended_bboxes,
            "abnormal_detected": abnormal_detected,
        }
        return status, overlay_state

    def _read_next_video_frame(self):
        ok, frame = self.cap.read()
        if ok:
            return frame

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ok, frame = self.cap.read()
        if not ok:
            return None

        with self.lock:
            self.frame_id = 0
        return frame

    def _stream_loop(self):
        while not self.stop_event.is_set():
            started_at = time.time()
            frame = self._read_next_video_frame()
            if frame is None:
                time.sleep(0.05)
                continue

            with self.lock:
                current_frame_id = self.frame_id
                self.frame_id += 1
                self.latest_raw_frame = frame.copy()
                self.latest_raw_frame_id = current_frame_id
                status = dict(self.latest_status)
                overlay_state = {
                    "scene_objects": list(self.overlay_state.get("scene_objects", [])),
                    "sudden_run_ids": set(self.overlay_state.get("sudden_run_ids", set())),
                    "unattended_bag_ids": set(self.overlay_state.get("unattended_bag_ids", set())),
                    "unattended_bboxes": list(self.overlay_state.get("unattended_bboxes", [])),
                    "abnormal_detected": bool(self.overlay_state.get("abnormal_detected", False)),
                }

            self._draw_pinpoint_boxes(
                frame=frame,
                scene_objects=overlay_state["scene_objects"],
                sudden_run_ids=overlay_state["sudden_run_ids"],
                unattended_bag_ids=overlay_state["unattended_bag_ids"],
                unattended_bboxes=overlay_state["unattended_bboxes"],
            )
            self._draw_abnormal_boxes(
                frame=frame,
                scene_objects=overlay_state["scene_objects"],
                abnormal_detected=overlay_state["abnormal_detected"],
                sudden_run_ids=overlay_state["sudden_run_ids"],
            )
            self._draw_overlays(frame, status)

            ok, jpeg = cv2.imencode(
                ".jpg",
                frame,
                [int(cv2.IMWRITE_JPEG_QUALITY), 78],
            )
            if ok:
                with self.lock:
                    self.latest_jpeg = jpeg.tobytes()
                    self.latest_stream_frame_id = current_frame_id

            elapsed = time.time() - started_at
            remaining = self.stream_interval - elapsed
            if remaining > 0:
                self.stop_event.wait(remaining)

    def _inference_loop(self):
        last_inferred_frame_id = -1
        while not self.stop_event.is_set():
            started_at = time.time()
            with self.lock:
                frame_id = self.latest_raw_frame_id
                frame = None if self.latest_raw_frame is None else self.latest_raw_frame.copy()

            if frame is None or frame_id == last_inferred_frame_id:
                self.stop_event.wait(0.01)
                continue

            ts = time.time()
            result = self.runner.run_frame(frame=frame, frame_id=frame_id, timestamp=ts)
            status, overlay_state = self._parse_inference_result(
                result=result,
                frame_id=frame_id,
                ts=ts,
            )

            with self.lock:
                self.latest_status = status
                self.overlay_state = overlay_state
            last_inferred_frame_id = frame_id

            elapsed = time.time() - started_at
            remaining = self.inference_interval - elapsed
            if remaining > 0:
                self.stop_event.wait(remaining)

    def start(self):
        if (
            self.stream_thread
            and self.stream_thread.is_alive()
            and self.inference_thread
            and self.inference_thread.is_alive()
        ):
            return

        self.stop_event.clear()
        self.stream_thread = threading.Thread(
            target=self._stream_loop,
            name="surveillance-stream",
            daemon=True,
        )
        self.inference_thread = threading.Thread(
            target=self._inference_loop,
            name="surveillance-inference",
            daemon=True,
        )
        self.stream_thread.start()
        self.inference_thread.start()

    def stop(self):
        self.stop_event.set()
        if self.stream_thread and self.stream_thread.is_alive():
            self.stream_thread.join(timeout=2.0)
        if self.inference_thread and self.inference_thread.is_alive():
            self.inference_thread.join(timeout=2.0)
        if self.cap.isOpened():
            self.cap.release()

    def mjpeg_generator(self):
        last_stream_frame_id = -1
        while not self.stop_event.is_set():
            with self.lock:
                stream_frame_id = self.latest_stream_frame_id
                jpeg = self.latest_jpeg

            if jpeg is None or stream_frame_id == last_stream_frame_id:
                self.stop_event.wait(0.005)
                continue

            last_stream_frame_id = stream_frame_id
            payload = (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
            )
            yield payload

    def get_latest_status(self):
        with self.lock:
            return dict(self.latest_status)

    def get_snapshot_base64(self):
        with self.lock:
            jpeg = self.latest_jpeg
            status = dict(self.latest_status)
        if jpeg is None:
            return {"image": None, "status": status}
        return {
            "image": base64.b64encode(jpeg).decode("utf-8"),
            "status": status,
        }
