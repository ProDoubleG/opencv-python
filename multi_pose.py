# %%
# Install Dependencies
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import cv2
from matplotlib import pyplot as plt
import sys
# %%
# parameter
use_camera = False # web cam not tested
status = 'Initial'
# %%
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu,True)
# %%
# functions..
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
    global status
    y,x,c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y,x,1]))
    face_y,_, face_confidence            =shaped[-1]
    right_knee_y,_,right_knee_confidence = shaped[0]
    left_knee_y,_,left_knee_confidence   = shaped[1]

    if (face_confidence<0.2) | (right_knee_confidence<0.2) | (left_knee_confidence<0.2):
        pass
    else:
        if (np.float(face_y-right_knee_y)<25) | (np.float(face_y-left_knee_y)<25):
        # if (np.float(face_y-right_knee_y)<5) | (np.float(face_y-left_knee_y)<5):
            status = 'Fall'
        else:
            status = 'Normal'
        
        for kp in shaped:
            ky, kx, kp_conf = kp
            if kp_conf > confidence_threshold:
                cv2.circle(frame,(int(kx), int(ky)), 4, (0,255,0), -1)

def draw_connections(frame, keypoints, edges, confidence_threshold):
    y,x,c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y,x,1]))

    for edge, color in edges.items():
        p1, p2 = edge
        y1, x1, c1 = shaped[p1]
        y2, x2, c2 = shaped[p2]

        if (c1 > confidence_threshold) & (c2 > confidence_threshold):
            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (100,100,100), 2)

# Loop Function for all person
def loop_through_people(frame, keypoints_with_scores, edges, confidence_threshold):
    for person in keypoints_with_scores:
        draw_connections(frame, person, edges, confidence_threshold)
        draw_keypoints(frame, person, confidence_threshold)

# %%
# Load Model
model = hub.load('https://tfhub.dev/google/movenet/multipose/lightning/1')
movenet = model.signatures['serving_default']
# %%
if use_camera:
    cap = cv2.VideoCapture(2)
    while cap.isOpened():
        ret, frame = cap.read()

        cv2.imshow('fall detection',frame)

        if cv2.waitKey(10) & 0xFF==ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
else:
    cap = cv2.VideoCapture('falls.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (256,256))

    while (cap.isOpened()):
        ret, frame = cap.read()
        
        # Resize
        try:
            img = frame.copy()
        except AttributeError:
            print(" --- Video ended --- ")
            sys.exit()
        img = tf.image.resize_with_pad(tf.expand_dims(img, axis=0),256,256)
        input_img = tf.cast(img, dtype=tf.int32)

        # Detect
        results = movenet(input_img)
        keypoints_with_scores = results['output_0'].numpy()[:,:,:51].reshape((6,17,3)) # 6 people, 3d points for 17 key points

        # Render keypoints
        loop_through_people(frame, keypoints_with_scores, EDGES, 0.3)

        font                   = cv2.FONT_HERSHEY_SIMPLEX
        bottomLeftCornerOfText = (10,40)
        fontScale              = 1
        fontColor              = (255,255,255)
        thickness              = 1
        lineType               = 2

        cv2.putText(frame,f'status :{status}',
            bottomLeftCornerOfText, 
            font, 
            fontScale,
            fontColor,
            thickness,
            lineType)

        out.write(frame)
        cv2.imshow('fall detection', frame)

        # cv2.imshow('Movenet Multipse', frame)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    cap.release()
    out.release()
    cv2.destroyAllWindows()