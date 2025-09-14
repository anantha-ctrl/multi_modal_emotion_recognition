import cv2
import numpy as np
import os

def preprocess_video(video_path, frame_rate=1):
    cap = cv2.VideoCapture(video_path)
    frames = []
    count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if count % frame_rate == 0:
            frame = cv2.resize(frame, (224, 224))
            frames.append(frame)

        count += 1

    cap.release()
    return np.array(frames)  # Shape: (num_frames, 224, 224, 3)
