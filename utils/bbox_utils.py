import cv2


def get_center_of_bbox(bbox):
    x1, y1, x2, y2 = bbox
    center_x = int((x1 + x2) / 2)
    center_y = int((y1 + y2) / 2)
    return (center_x, center_y)

def measure_distance(p1,p2):
    return ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)**0.5

def get_foot_position(bbox):
    x1, y1, x2, y2 = bbox
    return (int((x1 + x2) / 2), y2)

def get_closest_keypoint_index(point, keypoints, keypoint_indices):
    closest_distance = float('inf')
    key_point_ind = keypoint_indices[0]

    for keypoint_indix in keypoint_indices:
        keypoint = keypoints[keypoint_indix]#, keypoints[keypoint_indix*2+1]
        distance = abs(point[1]-keypoint[1])
        if distance<closest_distance:
            closest_distance = distance
            key_point_ind = keypoint_indix
    
    return key_point_ind

def get_height_of_bbox(bbox):
    return bbox[3]-bbox[1]

def measure_xy_distance(p1,p2):
    return abs(p1[0]-p2[0]), abs(p1[1]-p2[1])

def draw_bounding_boxes(image_path, player_detections, ball_detections):
    detections_output_image = cv2.imread(image_path)
    id = 1
    for player_dict in player_detections:
        x1, y1, x2, y2 = map(int, player_dict['bbox'])
        track_id = player_dict.get('track_id', id)
        id+=1
        confidence = player_dict['confidence']
        label = f'ID: {track_id}, Conf: {confidence:.2f}'
        cv2.rectangle(detections_output_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(detections_output_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    for ball_dict in ball_detections:
        x1, y1, x2, y2 = map(int, ball_dict['bbox'])
        track_id = ball_dict.get('track_id', 'N/A')
        confidence = ball_dict['confidence']
        label = f'ID: {track_id}, Conf: {confidence:.2f}'
        cv2.rectangle(detections_output_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(detections_output_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    return detections_output_image

"""
def get_center_of_bbox(bbox):
    return (int((bbox[0]+bbox[2])/2),int((bbox[1]+bbox[3])/2))
"""