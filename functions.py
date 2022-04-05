import numpy as np
import cv2
import math
EDGES = {
    (0, 1): 'm',
    (0, 2): 'c',
    (1, 3): 'm',
    (2, 4): 'c',
    (0, 5): 'm',
    (0, 6): 'c',
    (5, 7): 'm',
    (7, 9): 'm',
    (6, 8): 'c',
    (8, 10): 'c',
    (5, 6): 'y',
    (5, 11): 'm',
    (6, 12): 'c',
    (11, 12): 'y',
    (11, 13): 'm',
    (13, 15): 'm',
    (12, 14): 'c',
    (14, 16): 'c'
}

def draw_keypoints(frame , keypoints, confidence_threshold):
    y,x,c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y,x,1]))

    x_min, y_min = np.min(shaped[:,1],0), np.min(shaped[:,0],0)
    x_max, y_max = np.max(shaped[:,1],0), np.max(shaped[:,0],0)

    y_nose,x_nose,face_confidence =shaped[0]
    y_right_waist, x_right_waist, c_right_waist = shaped[-5]
    y_left_waist, x_left_waist, c_left_waist = shaped[-6]

    x_waist_center = 0.5*(x_right_waist + x_left_waist)
    y_waist_center = 0.5*(y_right_waist + y_left_waist)

    upper_body_tilt = np.arctan(abs(y_nose - y_waist_center)/abs(x_nose  -x_waist_center))

    dot_color = (0,255,0)

    if (face_confidence> confidence_threshold)&(c_right_waist>confidence_threshold)&(c_left_waist>confidence_threshold):
        if upper_body_tilt < math.pi/6:
            cv2.rectangle(frame, (int(x_min), int(y_min)), (int(x_max),int(y_max)), (0,0,255), 2)
        else:
            cv2.rectangle(frame, (int(x_min), int(y_min)), (int(x_max),int(y_max)), (0,255,0), 2)
    else:
        dot_color = (0,255,255)
        pass

    for kp in shaped:
        ky, kx, kp_conf = kp
        if (kp_conf > confidence_threshold):
            cv2.circle(frame,(int(kx), int(ky)), 4, dot_color, -1)

def draw_connections(frame, keypoints, confidence_threshold, edges=EDGES):
    y,x,c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y,x,1]))

    for edge, _ in edges.items():
        p1, p2 = edge
        y1, x1, c1 = shaped[p1]
        y2, x2, c2 = shaped[p2]

        if (c1 > confidence_threshold) & (c2 > confidence_threshold):
            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (100,100,100), 2)
    
            
# Loop Function for all person
def loop_through_people(frame, keypoints_with_scores, confidence_threshold):
    global use_camera

    for person in keypoints_with_scores:
        draw_keypoints(frame, person, confidence_threshold)
        draw_connections(frame, person, confidence_threshold)