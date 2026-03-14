from deep_sort_realtime.deepsort_tracker import DeepSort

class ObjectTracker:
    def __init__(self):
        self.tracker = DeepSort(
            max_age=30,
            n_init=3,
            max_iou_distance=0.7,
            embedder="mobilenet"
        )

    def update(self, detections, frame):
        ds_inputs = []

        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            w, h = x2 - x1, y2 - y1
            ds_inputs.append((
                [x1, y1, w, h],
                det["confidence"],
                det["class"]
            ))

        tracks = self.tracker.update_tracks(ds_inputs, frame=frame)

        tracked_objects = []
        for track in tracks:
            if not track.is_confirmed():
                continue

            x1, y1, x2, y2 = map(int, track.to_ltrb())

            tracked_objects.append({
                "track_id": track.track_id,
                "class": track.get_det_class(),
                "bbox": [x1, y1, x2, y2]
            })

        return tracked_objects
