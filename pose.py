import cv2
import mediapipe as mp
import numpy as np
from matplotlib import pyplot as plt
import os

# Function to detect poses and draw them on the image
def detect_and_draw_pose(image, visibility_threshold=0.5):
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    # Load MediaPipe Pose model
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()

    # Convert image to RGB (MediaPipe accepts RGB images)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Perform pose estimation
    results = pose.process(image_rgb)

    # Draw pose landmarks on the image
    if results.pose_landmarks:
        mp_drawing = mp.solutions.drawing_utils
        annotated_image = image.copy()
        
        landmark_dict = {}

        for i, landmark in enumerate(results.pose_landmarks.landmark):
            landmark_name = mp_pose.PoseLandmark(i).name
            x = int(landmark.x * image.shape[1])
            y = int(landmark.y * image.shape[0])
            visibility = landmark.visibility

            # Add to landmark_dict
            landmark_dict[landmark_name] = (x, y, visibility)

            # Draw on image only if visibility is above the threshold
            if visibility > visibility_threshold:
                # Draw point
                cv2.circle(annotated_image, (x, y), 5, (0, 255, 0), -1)

        # Draw connections
        for connection in mp_pose.POSE_CONNECTIONS:
            start_idx = connection[0]
            end_idx = connection[1]
            if results.pose_landmarks.landmark[start_idx].visibility > visibility_threshold and results.pose_landmarks.landmark[end_idx].visibility > visibility_threshold:
                start_point = (int(results.pose_landmarks.landmark[start_idx].x * image.shape[1]), 
                               int(results.pose_landmarks.landmark[start_idx].y * image.shape[0]))
                end_point = (int(results.pose_landmarks.landmark[end_idx].x * image.shape[1]), 
                             int(results.pose_landmarks.landmark[end_idx].y * image.shape[0]))
                cv2.line(annotated_image, start_point, end_point, (0, 255, 0), 2)

        return annotated_image, landmark_dict
    else:
        return image, None

def run(image_input):
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    image_path = image_input
    image = cv2.imread(image_path)
    annotated_image, pose_landmarks_dict = detect_and_draw_pose(image, visibility_threshold=0.6)

    # Original Pose Dictionary
    pose_dict = {}

    if pose_landmarks_dict:
        for name, (x, y, visibility) in pose_landmarks_dict.items():
            pose_dict[str(name) + '_x'] = x
            pose_dict[str(name) + '_y'] = y
            pose_dict[str(name) + '_v'] = visibility

    return pose_dict