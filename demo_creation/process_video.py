import cv2
import os

def input_video(path, writer_file_name='output.mp4'):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Input video file not found: {path}")
    
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError(f"Could not open video file: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    out_writer = cv2.VideoWriter(writer_file_name, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    return cap, fps, width, height, out_writer

