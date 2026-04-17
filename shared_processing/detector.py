from ultralytics import YOLO

class ObjectDetector:
    def __init__(
        self,
        model_path="yolov8n.pt",
        person_conf_thresh=0.4,
        bag_conf_thresh=0.15,
    ):
        self.model = YOLO(model_path)
        self.person_conf_thresh = person_conf_thresh
        self.bag_conf_thresh = bag_conf_thresh
        self.bag_labels = {"backpack", "handbag", "suitcase", "bag"}

    def detect(self, frame):
        results = self.model(frame, verbose=False)[0]
        detections = []

        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            label = self.model.names[cls_id]

            if label == "person":
                min_conf = self.person_conf_thresh
            elif label in self.bag_labels:
                min_conf = self.bag_conf_thresh
            else:
                continue

            if conf < min_conf:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # We only care about people & bags
            if label == "person" or label in self.bag_labels:
                detections.append({
                    "class": label,
                    "bbox": [x1, y1, x2, y2],
                    "confidence": conf
                })

        return detections
