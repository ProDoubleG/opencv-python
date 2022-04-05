# %%
# Install Dependencies
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import cv2
from matplotlib import pyplot as plt
import sys
import functions
# %%
# parameter
use_camera = True
status = 'Initial'
screen_resolution = (640,480)
# %%
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu,True)
# %%
# Load Model
model = hub.load('https://tfhub.dev/google/movenet/multipose/lightning/1')
movenet = model.signatures['serving_default']
# %%
if use_camera:
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()

        try:
            img = frame.copy()
        except AttributeError:
            print(" --- Video ended --- ")
        
        img = tf.image.resize_with_pad(tf.expand_dims(img, axis=0),screen_resolution[1],screen_resolution[0])
        input_img = tf.cast(img, dtype=tf.int32)

        results = movenet(input_img)
        keypoints_with_scores = results['output_0'].numpy()[:,:,:51].reshape((6,17,3)) # 6 people, 3d points for 17 key points
        
        functions.loop_through_people(frame, keypoints_with_scores,0.3)

        cv2.imshow('fall detection',frame)

        if cv2.waitKey(1) & 0xFF==ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

else:
    cap = cv2.VideoCapture('falls.mp4')
    # fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    # out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (256,256))

    while (cap.isOpened()):
        ret, frame = cap.read()
        
        # Resize
        try:
            img = frame.copy()
        except AttributeError:
            print(" --- Video ended --- ")
            sys.exit()
        img = tf.image.resize_with_pad(tf.expand_dims(img, axis=0),screen_resolution[1],screen_resolution[0])
        input_img = tf.cast(img, dtype=tf.int32)

        
        # Detect
        results = movenet(input_img)
        keypoints_with_scores = results['output_0'].numpy()[:,:,:51].reshape((6,17,3)) # 6 people, 3d points for 17 key points
        
        # Render keypoints
        functions.loop_through_people(frame, keypoints_with_scores, 0.3)

        # font                   = cv2.FONT_HERSHEY_SIMPLEX
        # bottomLeftCornerOfText = (10,40)
        # fontScale              = 1
        # fontColor              = (255,255,255)
        # thickness              = 1
        # lineType               = 2

        # cv2.putText(frame,f'status :{status}',
        #     bottomLeftCornerOfText, 
        #     font, 
        #     fontScale,
        #     fontColor,
        #     thickness,
        #     lineType)

        # out.write(frame)
        cv2.imshow('fall detection', frame)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    # out.release()
    cv2.destroyAllWindows()