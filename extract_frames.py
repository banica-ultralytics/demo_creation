import cv2, os
import numpy as np
from matplotlib import pyplot as plt
from skimage.measure import shannon_entropy


test_video_path = "/Users/banika/Desktop/video.mov"
output_folder = "./extracted_frames"


def frame_entropy(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return shannon_entropy(gray)

def laplacian_variance(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def extract_frames(video_path, output_folder, temperature=10.0, step_size=10, treshold=0):    
    os.makedirs(output_folder, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Start at first frame
    current_idx = 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, current_idx)
    ret, current_frame = cap.read()
    current_variance = (laplacian_variance(current_frame) + frame_entropy(current_frame) ) / 2
    
    saved_frames = []
    saved_frames.append((current_idx, current_variance))
    cv2.imwrite(f"{output_folder}/frame_{current_idx:06d}.jpg", current_frame)

    while current_idx < total_frames - step_size:
        # Look at next candidate frame
        next_idx = current_idx + step_size
        cap.set(cv2.CAP_PROP_POS_FRAMES, next_idx)
        ret, next_frame = cap.read()
        
        if not ret:
            break
            
        next_variance = laplacian_variance(next_frame)
        
        variance_diff = next_variance - current_variance
        
        prob_accept = 1 / (1 + np.exp(-variance_diff / temperature))
        
        next_variance = laplacian_variance(next_frame)
        variance_diff = next_variance - current_variance

        print(f"Current idx: {current_idx}, Next idx: {next_idx}, Current var: {current_variance:.2f}, Next var: {next_variance:.2f}, Var diff: {variance_diff:.2f}, Prob accept: {prob_accept:.4f}")
        
        if variance_diff > treshold:
            prob_accept = 1 - np.exp(-variance_diff / temperature)
            
            if np.random.random() < prob_accept:
                saved_frames.append((next_idx, next_variance))
                cv2.imwrite(f"{output_folder}/frame_{next_idx:06d}.jpg", next_frame)
                current_variance = next_variance  
        
        current_idx = next_idx
        current_variance = next_variance
    
    cap.release()


extract_frames(test_video_path, output_folder)

