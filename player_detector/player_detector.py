import cv2
from ultralytics import YOLO 


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
              if class_id == 0:
                  xmin, ymin, xmax, ymax = box.xyxy[0].tolist()
                  player_detection = {
                      'bbox': [xmin, ymin, xmax, ymax]
                  }
                  if hasattr(box, 'id') and box.id is not None:
                      player_detection['track_id'] = box.id.item()
                  player_detections.append(player_detection)

      return player_detections

    def choose_and_filter_players(self, court_keypoints, player_detections):
        """
        Logic for Filtering Objects - TODO
        """
        return player_detections
    
    def draw_bounding_boxes(self, image_path, filtered_player_detections):
      detections_output_image = cv2.imread(image_path)
      for player_dict in filtered_player_detections:
          for track_id, bbox in player_dict.items():
              x1, y1, x2, y2 = map(int, bbox)
              cv2.rectangle(detections_output_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
              cv2.putText(detections_output_image, f'ID: {track_id}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
      
      return detections_output_image