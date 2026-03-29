import numpy as np
from collections import deque

class TrajectoryManager:
    def __init__(self, max_history=20):
        self.history = {}  # track_id → deque of positions
        self.max_history = max_history

    def update(self, tracked_objects):
        enriched_objects = []

        for obj in tracked_objects:
            track_id = obj["track_id"]
            x1, y1, x2, y2 = obj["bbox"]

            # center point
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2

            if track_id not in self.history:
                self.history[track_id] = deque(maxlen=self.max_history)

            self.history[track_id].append((cx, cy))

            trajectory = list(self.history[track_id])

            # compute velocity
            velocity = (0.0, 0.0)
            speed = 0.0

            if len(trajectory) >= 2:
                dx = trajectory[-1][0] - trajectory[-2][0]
                dy = trajectory[-1][1] - trajectory[-2][1]

                velocity = (dx, dy)
                speed = np.sqrt(dx**2 + dy**2)

            enriched_objects.append({
                **obj,
                "center": (cx, cy),
                "trajectory": trajectory,
                "velocity": velocity,
                "speed": float(speed)
            })

        return enriched_objects