import numpy as np
import cv2

class SceneState:
    def __init__(self, frame_id, timestamp, objects, optical_flow):
        self.frame_id = frame_id
        self.timestamp = timestamp
        self.objects = objects
        self.optical_flow = optical_flow

        self.crowd_count = sum(
            1 for obj in objects if obj["class"] == "person"
        )

        self.global_motion = self._compute_global_motion()

    def _compute_global_motion(self):
        if self.optical_flow is None:
            return {
                "mean_magnitude": 0.0,
                "std_magnitude": 0.0
            }

        mag, _ = cv2.cartToPolar(
            self.optical_flow[..., 0],
            self.optical_flow[..., 1]
        )

        return {
            "mean_magnitude": float(np.mean(mag)),
            "std_magnitude": float(np.std(mag))
        }
