import cv2


def save_image(output_image_path, image):
  cv2.imwrite(output_image_path, image)

def read_image(input_image_path):
  return cv2.imread(input_image_path)