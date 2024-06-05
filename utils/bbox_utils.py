import cv2

FONT = cv2.FONT_HERSHEY_PLAIN


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

def draw_bounding_boxes(image_path, img, player_detections, ball_detections):
    if image_path is not None:
        detections_output_image = cv2.imread(image_path)
    else:
        detections_output_image = img
    player_id = "A"
    ball_id = 1
    color_players = (255,0,255)
    color_ball = (255,0,0)

    for player_dict in player_detections:
        x1, y1, x2, y2 = map(int, player_dict['bbox'])
        track_id = player_dict.get('track_id', player_id)
        player_id="B"
        confidence = player_dict['confidence']
        label = f'PLAYER {track_id}'
        cv2.rectangle(detections_output_image, (x1, y1), (x2, y2), color_players, 2)
        (text_width, text_height), baseline = cv2.getTextSize(label, FONT, 0.5, 1)
        cv2.rectangle(detections_output_image, (x1,y1-text_height-baseline),(x1+text_width,y1), color_players, -1)
        cv2.putText(detections_output_image, label, (x1, y1 - baseline), FONT, 0.5, (0, 0, 0), 1)
    

    for ball_dict in ball_detections:
        x1, y1, x2, y2 = map(int, ball_dict['bbox'])
        center = get_center_of_bbox(ball_dict['bbox'])
        track_id = ball_dict.get('track_id', ball_id)
        confidence = ball_dict['confidence']
        label = f'BALL'
        (text_width, text_height), baseline = cv2.getTextSize(label, FONT, 0.5, 1)
        cv2.rectangle(detections_output_image, (x1,y1-text_height-baseline), (x1+text_width,y1), color_ball, -1)
        cv2.circle(detections_output_image, center, 5, color_ball, 2)
        cv2.putText(detections_output_image, label, (x1, y1 - baseline), FONT, 0.5, (0,0,0), 1)

    return detections_output_image
