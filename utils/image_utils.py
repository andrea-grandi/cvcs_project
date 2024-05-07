import cv2

def save_image(output_image_path, image):
  cv2.imwrite(output_image_path, image)
