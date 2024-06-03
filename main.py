import cv2
import numpy as np
import torch
from tracknet import TrackerNet
from court_line_detector import CourtLineDetector
from geometrical_transformations import GeometricalTransformations
from utils import (read_video, 
                   save_video,
                   write_video,
                   read_image,
                   scene_detect,
                   save_image,
                   draw_bounding_boxes
                   )
import matplotlib.pyplot as plt
from player_detector import PlayerDetector
from ball_detector import BallDetector
from court_visualizer import CourtVisualizer
from court_reference import CourtReference
from tracking import TrackingBallDetector, TrackingCourtDetectorNet, TrackingPersonDetector, TrackingBounceDetector, tracking

"""
TO-DO: 
1. Court Line Detector (Keypoints)
2. Player Detection - Train yolo 
3. Player Tracking 
4. Ball Detection - Train yolo
5. Ball Tracking  
6. Geometrical Trasformation of Playground 
7. CNN Component Design (we can design a part of a net for ball detection)
8. Image Processing Operator
9. Retrieval Algorithm or Component (match and player recognition)
10. Spatial Relationship Between Objects - Court Visualization
11. Generating Natural Language Description
"""

"""
Width and Height for Transformed Image:
500x500 is the dimension for the
geometrical transformed image
"""

WIDTH, HEIGHT = 500,500


def main():

    # --- Models Paths --- #
    yolo_player_model_path = "models/yolo_player_model.pt"
    yolo_ball_model_path = "models/yolo_ball_model.pt"
    court_model_path = "models/court_model.pt"
    ball_track_model_path = "models/ball_tracking_model.pt"
    bounce_tracking_model_path = "models/bounce_tracking_model.cbm"

    # --- Device --- #
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # --- Input Image Paths --- #
    input_image_path = "input/input_images/input_image1.png"

    # --- Input Video Paths --- #
    input_video_path = "input/input_videos/input_video6.mp4"

    # --- Output Image Paths --- #
    output_keypoints_image_path = "output/output_images/output_keypoints_image.png"
    output_transformed_image_path = "output/output_images/output_transformed_image.png"
    output_detection_image_path = "output/output_images/output_detection_image.png"
    output_court_visualizer_image_path = "output/output_images/output_court_visualizer_image.png"

    # --- Output Video Paths --- #
    output_video_path = "output/output_videos/output_video.mp4"

    # --- Info --- #
    print("Input Image Path: " + input_image_path)
    print("Input Video Path: " + input_video_path)
    print("Device: " + device)

    # --- Court Line Detector --- #
    court_line_detector = CourtLineDetector(court_model_path)
    court_keypoints_img, court_keypoints = court_line_detector.predict(input_image_path, output_keypoints_image_path, use_refine_kps=True)

    # --- Apply Geometry Trasformations --- #
    src_points = court_line_detector.get_court_corners(court_keypoints)
    dst_points = np.array([(0,0), (WIDTH, 0), (0, HEIGHT), (WIDTH, HEIGHT)], dtype=np.float32)
    geometric_transform = GeometricalTransformations(src_points, dst_points)
    transformed_image = geometric_transform.transform_image(court_keypoints_img)
    cropped_transformed_image = geometric_transform.crop_image(transformed_image)

    # --- Players Detection --- #
    player_detector = PlayerDetector(yolo_player_model_path)
    player_detections = player_detector.detect(input_image_path)

    # Not needed anymore (couse i have trained YOLOv8 for player detection)
    # Dataset: https://universe.roboflow.com/deep-hbapi/tennis-yfcgx/dataset/1#
    # For reference .ipynb go to training/yolov8_training.ipynb
    # filtered_player_detections = player_detector.choose_and_filter_players(court_keypoints, player_detections)

    # --- Ball Detection --- #
    ball_detector = BallDetector(yolo_ball_model_path)
    ball_detection = ball_detector.detect(input_image_path)

    # --- Draw Bounding Boxes --- #
    detections_output_image = draw_bounding_boxes(input_image_path, player_detections, ball_detection)

    # --- Court Visualization and Spatial Relationships --- #
    court_visualizer = CourtVisualizer(read_image(input_image_path))
    player_court_visualizer_detections, ball_court_visualizer_detections = court_visualizer.convert_bounding_boxes_to_mini_court_coordinates(
                                                                                                            player_detections, 
                                                                                                            ball_detection,
                                                                                                            court_keypoints)
    court_visualizer_image = court_visualizer.draw_mini_court(detections_output_image)
    court_visualizer_image = court_visualizer.draw_points_players_on_mini_court(court_visualizer_image, player_court_visualizer_detections)
    #court_visualizer_image = court_visualizer.draw_points_ball_on_mini_court(court_visualizer_image, ball_court_visualizer_detections)

    # --- Tracking --- #
    frames, fps = read_video(input_video_path) 
    scenes = scene_detect(input_video_path)
    
    print('ball detection')
    ball_detector = TrackingBallDetector(ball_track_model_path, device)
    ball_track = ball_detector.infer_model(frames)

    print('court detection')
    court_detector = TrackingCourtDetectorNet(court_model_path, device)
    homography_matrices, kps_court = court_detector.infer_model(frames)

    print('person detection')
    person_detector = TrackingPersonDetector(device)
    persons_top, persons_bottom = person_detector.track_players(frames, homography_matrices, filter_players=False)

    # bounce detection
    bounce_detector = TrackingBounceDetector(bounce_tracking_model_path)
    x_ball = [x[0] for x in ball_track]
    y_ball = [x[1] for x in ball_track]
    bounces = bounce_detector.predict(x_ball, y_ball)

    # track
    imgs_res = tracking(frames, scenes, bounces, ball_track, homography_matrices, kps_court, persons_top, persons_bottom, draw_trace=True)

    # --- Save Video --- #
    write_video(imgs_res, fps, output_video_path)

    # --- Save Images --- #
    save_image(output_detection_image_path, detections_output_image)
    save_image(output_transformed_image_path, cropped_transformed_image)
    save_image(output_keypoints_image_path, court_keypoints_img)
    save_image(output_court_visualizer_image_path, court_visualizer_image)


if __name__ == "__main__":
    main()