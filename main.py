import cv2
import numpy as np
import torch
import pickle
import os
import random
from tracknet import TrackNet
from court_line_detector import CourtLineDetector, CourtLineDetectorResNet
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
from analysis import Analysis

"""
TO-DO: 
1. Court Line Detector (Keypoints)
2. Player Detection
3. Player Tracking 
4. Ball Detection
5. Ball Tracking  
6. Geometrical Trasformation of Playground 
7. CNN Component Design
8. Image Processing Operator
9. Retrieval Algorithm or Component (match and player recognition)
10. Spatial Relationship Between Objects - Court Visualization
11. Generating Natural Language Description
12. Video Analysis

Some ideas for NLD:
a. Player positions: 
    Example: "Player A is positioned near the baseline on the left side of the court, preparing to serve. 
    Player B stands at the center of the opposite baseline, ready to receive the serve."

b. Relative positions:
    Example: "Player A is approximately 1 meter from the left sideline and 2 meters behind the baseline, 
    while Player B is positioned directly opposite, about 1 meter from the right sideline."

c. Contextual details:
    Example: "The match appears to be intense, with both players focused on their respective positions. 
    The audience in the background is watching attentively."

and also integrate some quantitative description of the objects:
for example, players positions (X, Y coordinates on the court) or 
distance from specific lines (baseline, sideline)
    Example: "Player A: (X: 3.5m, Y: 0.5m from the left sideline), 
    Player B: (X: 11m, Y: 0.5m from the right sideline)"
"""

#WIDTH, HEIGHT = 350,500


def main():
    # --- Models Paths --- #
    yolo_player_model_path = "models/yolo_player_model_test.pt"
    yolo_ball_model_path = "models/yolo_ball_model.pt"
    court_model_path = "models/court_model.pt"
    court_model_resnet_path = "models/keypoints_model.pth"
    ball_track_model_path = "models/ball_tracking_model.pt"
    bounce_tracking_model_path = "models/bounce_tracking_model.cbm"

    # --- Device --- #
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # --- Random Choice of Input Image and Video --- #
    image_number = random.randint(1,24)
    video_number = random.randint(1,6)

    # --- Input Image Paths --- #
    input_image_path = f"input/input_images/input_image{image_number}.png"

    # --- Input Video Paths --- #
    input_video_path = f"input/input_videos/input_video{video_number}.mp4"

    # --- Output Image Paths --- #
    output_keypoints_image_path = f"output/output_images/output_image{image_number}"
    os.makedirs(output_keypoints_image_path, exist_ok=True)
    output_keypoints_image_path = os.path.join(output_keypoints_image_path, "output_keypoints_image.png")

    output_keypoints_image_resnet = f"output/output_images/output_image{image_number}"
    os.makedirs(output_keypoints_image_resnet, exist_ok=True)
    output_keypoints_image_resnet = os.path.join(output_keypoints_image_resnet, "output_keypoints_image_resnet.png")

    output_transformed_image_path = f"output/output_images/output_image{image_number}"
    os.makedirs(output_transformed_image_path, exist_ok=True)
    output_transformed_image_path = os.path.join(output_transformed_image_path, "output_transformed_image.png")

    output_detection_image_path = f"output/output_images/output_image{image_number}"
    os.makedirs(output_detection_image_path, exist_ok=True)
    output_detection_image_path = os.path.join(output_detection_image_path, "output_detection_image.png")

    output_court_visualizer_image_path = f"output/output_images/output_image{image_number}"
    os.makedirs(output_court_visualizer_image_path, exist_ok=True)
    output_court_visualizer_image_path = os.path.join(output_court_visualizer_image_path, "output_court_visualizer_image.png")

    # --- Output Video Paths --- #
    output_video_path = f"output/output_videos/output_video{video_number}.mp4"

    # --- Stubs Path --- #
    stub_player_path = "analysis/stubs/player_detections.pkl"
    stub_ball_path = "analysis/stubs/ball_detections.pkl"

    # --- Info --- #
    print("Input Image Path: " + input_image_path)
    print("Input Video Path: " + input_video_path)
    print("Device: " + device)

    # --- Court Line Detector --- #
    court_line_detector = CourtLineDetector(court_model_path)
    court_keypoints_img, court_keypoints = court_line_detector.predict(input_image_path, output_keypoints_image_path)

    # Keypoint with ResNet50
    court_line_detector_resnet = CourtLineDetectorResNet(court_model_resnet_path)
    court_keypoints_resnet = court_line_detector_resnet.predict(read_image(input_image_path))
    court_keypoints_resnet_img  = court_line_detector_resnet.draw_keypoints(read_image(input_image_path), court_keypoints_resnet)
    save_image(output_keypoints_image_resnet, court_keypoints_resnet_img)

    # --- Apply Geometry Trasformations --- #
    src_points = court_line_detector.get_court_corners(court_keypoints)
    dst_points = np.array([(410,175), (877,175), (200,571), (1070,571)], dtype=np.float32)
    #dst_points = np.array([(0,0), (WIDTH, 0), (0, HEIGHT), (WIDTH, HEIGHT)], dtype=np.float32)
    geometric_transform = GeometricalTransformations(1280, 720, src_points, dst_points)
    transformed_image = geometric_transform.transform_image(court_keypoints_img)
    cropped_transformed_image = geometric_transform.crop_image(transformed_image)

    # TEST
    save_image(output_transformed_image_path, cropped_transformed_image)
    input_image_path = f"output/output_images/output_image{image_number}/output_transformed_image.png"
    
    # --- Players Detection --- #
    player_detector = PlayerDetector(yolo_player_model_path)
    player_detections = player_detector.detect(input_image_path, img=None)

    # Not needed anymore (couse i have trained YOLOv8 for player detection)
    # Dataset: https://universe.roboflow.com/deep-hbapi/tennis-yfcgx/dataset/1#
    # For reference .ipynb go to training/yolov8_training.ipynb
    # filtered_player_detections = player_detector.choose_and_filter_players(court_keypoints, player_detections)

    # --- Ball Detection --- #
    ball_detector = BallDetector(yolo_ball_model_path)
    ball_detection = ball_detector.detect(input_image_path, img=None)

    # --- Draw Bounding Boxes --- #
    detections_output_image = draw_bounding_boxes(input_image_path, None, player_detections, ball_detection)

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
    if device == 'cuda':
        frames, fps = read_video(input_video_path) 
        scenes = scene_detect(input_video_path)
        
        print('ball detection')
        ball_detector = TrackingBallDetector(ball_track_model_path, device)
        ball_track = ball_detector.infer_model(frames)
        # Save the detections in the stubs folder
        with open(stub_ball_path, 'wb') as f:
            pickle.dump(player_detections, f)

        print('court detection')
        court_detector = TrackingCourtDetectorNet(court_model_path, device)
        homography_matrices, kps_court = court_detector.infer_model(frames)

        print('person detection')
        person_detector = TrackingPersonDetector(device)
        persons_top, persons_bottom = person_detector.track_players(frames, homography_matrices, filter_players=False)
        # Save the detections in the stubs folder
        with open(stub_player_path, 'wb') as f:
            pickle.dump(player_detections, f)

        # bounce detection
        bounce_detector = TrackingBounceDetector(bounce_tracking_model_path)
        x_ball = [x[0] for x in ball_track]
        y_ball = [x[1] for x in ball_track]
        bounces = bounce_detector.predict(x_ball, y_ball)

        # track (drow all players bounding boxes and ball trajectory)
        imgs_res = tracking(frames, scenes, bounces, ball_track, homography_matrices, kps_court, persons_top, persons_bottom, draw_trace=True)

        # --- Save Video --- #
        write_video(imgs_res, fps, output_video_path)

        # --- Analysis --- #
        analysis = Analysis(bounce_detector)
        bounce_df = analysis.analyze_bounces_and_trajectories(x_ball, y_ball)

    # --- Save Images --- #
    save_image(output_detection_image_path, detections_output_image)
    save_image(output_transformed_image_path, cropped_transformed_image)
    save_image(output_keypoints_image_path, court_keypoints_img)
    save_image(output_court_visualizer_image_path, court_visualizer_image)


if __name__ == "__main__":
    main()
