import cv2
import pandas as pd
import numpy as np
from copy import deepcopy
from court_line_detector import CourtLineDetector
from geometrical_transformations import GeometricalTransformations
from utils import (read_video, 
                   save_video,
                   save_image
                   )
import matplotlib.pyplot as plt
import random
from pathlib import Path
from player_detector import PlayerDetector

"""
TO-DO: 
1. Court Line Detector (Keypoints) - DONE
2. Player Detection
3. Player Tracking
4. Ball Detection 
5. Ball Tracking
6. Geometrical Trasformation of Play Field - DONE
7. CNN Component Design
8. Image Processing Operator (like SOBEL)
9. Retrieval Algorithm or Component
10. Spatial Relationship Between Objects
11. Generating Natural Language Description
"""

WIDTH, HEIGHT = 500,500


def main():

    # --- Models Paths --- #
    yolo_model_path = "models/yolov8x.pt"
    court_model_path = "models/model_tennis_court_det.pt"

    # --- Random Choice of Input Image --- #
    images_path = Path("input/input_images/")
    image_files = list(images_path.glob("*.png"))

    # --- Image or Video Paths --- #
    input_video_path = "input/input_videos/input_video6.mp4"
    input_image_path = str(random.choice(image_files))
    output_image_path = "output/output_images/output_image.png"
    output_transformed_image_path = "output/output_images/transformed_image.png"
    output_detector_path = "output/output_images/detector_image.png"

    # --- Court Line Detector --- #
    court_line_detector = CourtLineDetector(court_model_path)
    court_keypoints_img, court_keypoints = court_line_detector.predict(input_image_path, output_image_path, use_refine_kps=True)

    # --- Apply Geometry Trasformations --- #
    src_points = court_line_detector.get_court_corners(court_keypoints)
    dst_points = np.array([(0,0), (WIDTH, 0), (0, HEIGHT), (WIDTH, HEIGHT)], dtype=np.float32)

    geometric_transform = GeometricalTransformations(src_points, dst_points)
    transformed_image = geometric_transform.transform_image(court_keypoints_img)

    # --- Players Detection --- #
    player_detector = PlayerDetector(yolo_model_path)
    player_detections = player_detector.detect(input_image_path)
    filtered_player_detections = player_detector.choose_and_filter_players(court_keypoints, player_detections)
    detections_output_image = player_detector.draw_bounding_boxes(input_image_path, filtered_player_detections)
    
    # --- Save Images --- #
    save_image(output_detector_path, detections_output_image)
    save_image(output_transformed_image_path, transformed_image)
    save_image(output_image_path, court_keypoints_img)


if __name__ == "__main__":
    main()
