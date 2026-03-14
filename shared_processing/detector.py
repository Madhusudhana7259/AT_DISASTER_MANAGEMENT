from ultralytics import YOLO

class ObjectDetector:
    def __init__(self, model_path="yolov8n.pt", conf_thresh=0.4):
        self.model = YOLO(model_path)
        self.conf_thresh = conf_thresh

    def detect(self, frame):
        results = self.model(frame, verbose=False)[0]
        detections = []

        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])

            if conf < self.conf_thresh:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            label = self.model.names[cls_id]

            # We only care about people & bags
            if label in ["person", "backpack", "handbag", "suitcase"]:
                detections.append({
                    "class": label,
                    "bbox": [x1, y1, x2, y2],
                    "confidence": conf
                })

        return detections
