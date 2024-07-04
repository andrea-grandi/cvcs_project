import cv2
import torch
from ultralytics import YOLO 
import numpy as np
from utils import measure_distance, get_center_of_bbox


class PlayerDetector:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def detect(self, image_path, img):
        if image_path is not None:
            image = cv2.imread(image_path)
        else:
            image = img
        results = self.model(image)
        player_id = 1
        
        player_detections = []
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
                
                if player_id <= 2:
                    if class_id == 1:
                        player_detections.append(detection)
                        player_id+=1

        return player_detections
