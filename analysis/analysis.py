import cv2
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
from utils import save_image


class Analysis():
  def __init__(self, detector):

    """
    The detector is TrackingBounceDetector 
    """
    self.detector = detector

  def analyze_bounces_and_trajectories(self, x_coords, y_coords):
    bounce_frames = self.detector.predict(x_coords, y_coords)
    
    bounce_positions = [(x_coords[frame], y_coords[frame]) for frame in bounce_frames]
    bounce_df = pd.DataFrame(bounce_positions, columns=['x-coordinate', 'y-coordinate'])
 
    plt.figure(figsize=(12, 6))
    plt.plot(x_coords, y_coords, label='Ball Trajectory')
    plt.scatter(bounce_df['x-coordinate'], bounce_df['y-coordinate'], color='red', label='Bounces', zorder=5)
    plt.title('Ball trajectory and bounce positions')
    plt.xlabel('x-coordinate')
    plt.ylabel('y-coordinate')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    return bounce_df


