import numpy as np
import cv2

class HeatmapGenerator:
    def __init__(self, frame_width, frame_height, grid_size=64):
        self.W = frame_width
        self.H = frame_height
        self.grid = grid_size

    def _get_cell(self, x, y):
        gx = int((x / self.W) * self.grid)
        gy = int((y / self.H) * self.grid)

        gx = min(max(gx, 0), self.grid - 1)
        gy = min(max(gy, 0), self.grid - 1)

        return gx, gy

    def generate(self, scene_state):
        # Initialize channels
        occupancy = np.zeros((self.grid, self.grid), dtype=np.float32)
        speed_map = np.zeros((self.grid, self.grid), dtype=np.float32)
        motion_map = np.zeros((self.grid, self.grid), dtype=np.float32)

        # -------------------------
        # 1. Occupancy + Speed
        # -------------------------
        for obj in scene_state.objects:
            if obj["class"] != "person":
                continue

            x1, y1, x2, y2 = obj["bbox"]

            # center of bbox
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2

            gx, gy = self._get_cell(cx, cy)

            occupancy[gy, gx] += 1

            # speed (if available)
            speed = obj.get("speed", 0.0)
            speed_map[gy, gx] += speed

        # Normalize speed
        nonzero = occupancy > 0
        speed_map[nonzero] /= occupancy[nonzero]

        # -------------------------
        # 2. Motion (Optical Flow)
        # -------------------------
        if scene_state.optical_flow is not None:
            flow = scene_state.optical_flow

            mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])

            h, w = mag.shape

            for y in range(h):
                for x in range(w):
                    gx = int((x / w) * self.grid)
                    gy = int((y / h) * self.grid)

                    gx = min(gx, self.grid - 1)
                    gy = min(gy, self.grid - 1)

                    motion_map[gy, gx] += mag[y, x]

        # Normalize motion
        if np.max(motion_map) > 0:
            motion_map /= np.max(motion_map)

        # -------------------------
        # Stack channels
        # -------------------------
        heatmap = np.stack([occupancy, speed_map, motion_map], axis=0)

        return heatmap
