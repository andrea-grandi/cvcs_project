import cv2
import torch
from ultralytics import YOLO 
import numpy as np
from utils import measure_distance, get_center_of_bbox, draw_bounding_boxes


class PlayerDetector:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def detect(self, image_path):
        image = cv2.imread(image_path)
        results = self.model(image)

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
                
                # Class ID for Player (for YOLOv8 is the first one in results)
                if class_id == 0:
                    player_detections.append(detection)

        return player_detections

    def choose_and_filter_players(self, court_keypoints, player_detections):

        # Calculate the center of the court
        court_center_x = np.mean([court_keypoints[i][0] for i in range(4)])
        court_center_y = np.mean([court_keypoints[i][1] for i in range(4)])
        court_center = (court_center_x, court_center_y)
        
        # Calculate distances from the center of the court to each pearson
        player_distances = []
        for detection in player_detections:
            bbox_center = get_center_of_bbox(detection['bbox'])
            distance = measure_distance(court_center, bbox_center)
            player_distances.append((distance, detection))

        # Sort players by distance to the court center
        player_distances.sort(key=lambda x: x[0])

        # Select the two closest players
        filtered_player_detections = [player_distances[i][1] for i in range(min(4, len(player_distances)))]

        # Select the two player with higher confidence sorted by ID
        for i in range(4):
            if i < 3 and filtered_player_detections[i]['confidence']<filtered_player_detections[i+1]['confidence']:
                    tmp = filtered_player_detections[i]
                    filtered_player_detections[i] = filtered_player_detections[i+1]
                    filtered_player_detections[i+1] = tmp

        # Return only the first and second detections
        return filtered_player_detections[:2]
