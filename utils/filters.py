import cv2
import os


def apply_gaussian_filter(input_image_path, output_image_path, kernel_size=(5, 5), sigma=0):
    image = cv2.imread(input_image_path)
    if image is None:
        raise FileNotFoundError(f"Input image not found at {input_image_path}")
    denoised_image = cv2.GaussianBlur(image, kernel_size, sigma)
    os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
    cv2.imwrite(output_image_path, denoised_image)
    
    return denoised_image