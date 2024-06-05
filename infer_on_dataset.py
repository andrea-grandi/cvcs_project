from utils import (save_image,
                   draw_bounding_boxes
                  )
from player_detector import PlayerDetector
from ball_detector import BallDetector
from datasets import load_dataset
import os
import matplotlib.pyplot as plt
import cv2
import numpy as np


def main():
  # --- Loading Dataset --- #
  dataset = load_dataset('DinoDave/SpatialRelationsTennis')

  # --- Models Paths --- #
  yolo_player_model_path = "models/yolo_player_model.pt"
  yolo_ball_model_path = "models/yolo_ball_model.pt"

  # --- Output Directory --- #
  output_directory = "infer_on_dataset/"
  os.makedirs(output_directory, exist_ok=True)

  # --- Initialize Detectors --- #
  player_detector = PlayerDetector(yolo_player_model_path)
  ball_detector = BallDetector(yolo_ball_model_path)

  # --- Info --- #
  print("Processing images from the dataset...")

  print(len(dataset['train']))

  # --- Inference on Dataset --- #
  for i, data in enumerate(dataset['train']):
      input_image_path = None
      input_image = cv2.cvtColor(np.array(data['image']), cv2.COLOR_RGB2BGR)

      # --- Players Detection --- #
      player_detections = player_detector.detect(input_image_path, input_image)

      # --- Ball Detection --- #
      ball_detection = ball_detector.detect(input_image_path, input_image)

      # --- Draw Bounding Boxes --- #
      detections_output_image = draw_bounding_boxes(input_image_path, input_image, player_detections, ball_detection)

      # --- Save Images --- #
      output_detection_image_path = f"infer_on_dataset/output_detection_image_{i}.png"
      save_image(output_detection_image_path, detections_output_image)

      print(f"Processed and saved image {i + 1} of {len(dataset['train'])}")

  print("Processing complete.")


if __name__ == "__main__":
    main()