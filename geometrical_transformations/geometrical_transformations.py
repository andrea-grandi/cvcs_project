import cv2
import numpy as np


class GeometricalTransformations:
  def __init__(self, height, width, src_points, dst_points):
    self.crop_width = width 
    self.crop_height = height 
    self.src_points = np.array(src_points, dtype=np.float32)
    self.dst_points = np.array(dst_points, dtype=np.float32)
    self.homography_matrix, _ = cv2.findHomography(self.src_points, self.dst_points)

  def transform_image(self, image):
    height, width = image.shape[:2]
    transformed_image = cv2.warpPerspective(image, self.homography_matrix, (width, height))
    return transformed_image
  
  def crop_image(self, image):
    return image[0:self.crop_width, 0:self.crop_height]
  
