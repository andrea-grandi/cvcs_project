from ultralytics import YOLO 
import cv2
import pickle
import pandas as pd
from utils import measure_distance, get_center_of_bbox, draw_bounding_boxes


class BallDetector:
    def __init__(self,model_path):
        self.model = YOLO(model_path)

    def detect(self, image_path):
        image = cv2.imread(image_path)
        results = self.model(image)

        ball_detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                class_id = box.cls.item()
                confidence = box.conf.item()
                xmin, ymin, xmax, ymax = box.xyxy[0].tolist()
                detection = {
                    'bbox': [xmin, ymin, xmax, ymax],
                    'confidence': confidence
                }
                if hasattr(box, 'id') and box.id is not None:
                    detection['track_id'] = box.id.item()
                
                # Class ID for ball 
                if class_id == 0:
                    ball_detections.append(detection)

        return ball_detections
