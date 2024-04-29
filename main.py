import constants
import cv2
import pandas as pd
from copy import deepcopy
from court_line_detector import CourtLineDetector
from utils import (read_video, 
                   save_video
                   )

"""
TO-DO: 

1. Court Line Detector
2. Player Detection
3. Player Tracking
4. Ball Detection 
5. Ball Tracking
6. Geometrical Trasformation of Play Field
7. CNN Component Design
8. Image Processing Operator
9. Retrieval Algorithm or Component
10. Spatial Relationship Between Objects
11. Generating Natural Language Description
"""

def main():
    # --- Read Video --- #
    input_video_path = "input_videos/input_video2.mp4"
    video_frames = read_video(input_video_path)

    # --- Court Line Detector model --- #
    court_model_path = "models/keypoints_model.pth"
    court_line_detector = CourtLineDetector(court_model_path)
    court_keypoints = court_line_detector.predict(video_frames[0])

    # --- Draw court Keypoints --- #
    output_video_frames  = court_line_detector.draw_keypoints_on_video(video_frames, court_keypoints)

    # --- Draw frame number on top left corner --- #
    for i, frame in enumerate(output_video_frames):
        cv2.putText(frame, f"Frame: {i}",(10,30),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    save_video(output_video_frames, "output_videos/output_video.avi")

if __name__ == "__main__":
    main()