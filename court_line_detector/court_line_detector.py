import torch
import cv2
import numpy as np 
from tracknet import TrackNet
import torch.nn.functional as F
from postprocess import postprocess, refine_kps
import torchvision.transforms as transforms
from torchvision import models


OUTPUT_WIDTH = 640
OUTPUT_HEIGHT = 360

class CourtLineDetector:
    def __init__(self, model_path):
        self.model = TrackNet(out_channels=15)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = self.model.to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        
    def predict(self, input_path, output_path, use_refine_kps=False):
        # --- Read Input Image --- #
        image = cv2.imread(input_path)
        image_resized = cv2.resize(image, (OUTPUT_WIDTH, OUTPUT_HEIGHT))
        inp = (image_resized.astype(np.float32)/255.)
        inp = torch.tensor(np.rollaxis(inp,2,0))
        inp = inp.unsqueeze(0)

        # --- Keypoints Extraction using TrackNet --- #
        out = self.model(inp.float().to(self.device))[0]
        pred = F.sigmoid(out).detach().cpu().numpy()

        points = []
        for kps_num in range(14):
            heatmap = (pred[kps_num]*255).astype(np.uint8)
            x_pred, y_pred = postprocess(heatmap, low_thresh=170, max_radius=25)
            if use_refine_kps and kps_num not in [8, 12, 9] and x_pred and y_pred:
                x_pred, y_pred = refine_kps(image, int(y_pred), int(x_pred))
            points.append((x_pred, y_pred))
        
        keypoints = torch.tensor([points])
        keypoints = torch.squeeze(keypoints)

        for j in range(len(points)):
            if points[j][0] is not None:
                image = cv2.circle(image, (int(points[j][0]), int(points[j][1])),
                                   radius=0, color=(0, 0, 255), thickness=10)

        # --- Draw keypoint in center, top and bottom of playground --- #
        court_center_x = np.mean([keypoints[i][0] for i in range(4)])
        court_center_y = np.mean([keypoints[i][1] for i in range(4)])
        image = cv2.circle(image, (int(court_center_x), int(court_center_y)), radius=0, color=(0, 0, 255), thickness=10)

        top_center_x = np.mean([keypoints[i][0] for i in range(2)])
        top_center_y = np.mean([keypoints[i][1] for i in range(2)])
        image = cv2.circle(image, (int(top_center_x), int(top_center_y)), radius=0, color=(0, 0, 255), thickness=10)

        bottom_center_x = np.mean([keypoints[i][0] for i in range(2,4)])
        bottom_center_y = np.mean([keypoints[i][1] for i in range(2,4)])
        image = cv2.circle(image, (int(bottom_center_x), int(bottom_center_y)), radius=0, color=(0, 0, 255), thickness=10)

        # --- Add new keypoints to the list --- #
        additional_keypoints = [
            (court_center_x, court_center_y),
            (top_center_x, top_center_y),
            (bottom_center_x, bottom_center_y)
        ]

        keypoints = torch.cat((keypoints, torch.tensor(additional_keypoints)), dim=0)

        return image, keypoints
    
    def get_court_corners(self, keypoints):
        keypoints = np.array(keypoints)
        
        x_coords = keypoints[:, 0]
        y_coords = keypoints[:, 1]
        
        top_left = keypoints[np.argmin(x_coords + y_coords)]
        top_right = keypoints[np.argmin((1 - x_coords) + y_coords)]
        bottom_left = keypoints[np.argmin(x_coords + (1 - y_coords))]  
        bottom_right = keypoints[np.argmin((1 - x_coords) + (1 - y_coords))] 

        court_corners = np.array([top_left, top_right, bottom_left, bottom_right], dtype=np.float32)
        
        return court_corners
    

