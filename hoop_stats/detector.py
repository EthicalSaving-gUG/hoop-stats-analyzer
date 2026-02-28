from ultralytics import YOLO
import cv2

class HoopDetector:
    def __init__(self, model_variant="yolo11n.pt"):
        # Load pre-trained YOLOv11 (or fallback to v8 if v11 is not available in local env)
        try:
            self.model = YOLO(model_variant)
        except:
            self.model = YOLO("yolov8n.pt")
            
    def detect_players(self, frame):
        """
        Detects players (class 0 in COCO) and returns bounding boxes.
        """
        results = self.model(frame, classes=[0], verbose=False)
        detections = []
        for r in results:
            for box in r.boxes:
                # box.xyxy is [x1, y1, x2, y2]
                detections.append(box.xyxy[0].tolist())
        return detections

    def detect_ball(self, frame):
        """
        Detects sports balls (class 32 in COCO) and returns bounding boxes.
        """
        results = self.model(frame, classes=[32], verbose=False, conf=0.25)
        detections = []
        for r in results:
            for box in r.boxes:
                detections.append(box.xyxy[0].tolist())
        return detections

    def get_jersey_crop(self, frame, bbox):
        """
        Crops the player's torso area where the jersey number is likely.
        """
        x1, y1, x2, y2 = map(int, bbox)
        # Heuristic: Jersey number is usually in the upper-middle part of the bbox
        h = y2 - y1
        crop = frame[y1 + int(h*0.1): y1 + int(h*0.5), x1:x2]
        return crop
