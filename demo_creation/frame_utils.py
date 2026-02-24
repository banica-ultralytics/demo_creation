import cv2
import numpy as np

from font_utils import draw_rounded_rectangle
import demo_creation.demo_creation.utils as utils



# round the corners and add the brand watermark to video
# return video path
def brand_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not video_path or not cap.isOpened():
        return None
    
    save_video_path = str(video_path).replace('.mp4', '_branded.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(save_video_path, fourcc, fps, (width, height))
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        branded_frame = brand_frame(frame)
        out.write(branded_frame)
        
    cap.release()
    out.release()
    return save_video_path
    

# add watermark and round corners to a single frame
def brand_frame(frame, watermark_color='white', radius=35):
    frame = _add_ultralytics_watermark(frame, watermark_color)
    frame = _round_frame_corners(frame, radius)
    return frame


# add white the ultralytics watermark to frame
def _add_ultralytics_watermark(frame, watermark_color='white'):             
        logo_resized, x_pos, y_pos, logo_width, logo_height = _get_logo_resized(frame)
    
        overlay = frame.copy()
        overlay[y_pos:y_pos + logo_height, x_pos:x_pos + logo_width] = logo_resized
        
        alpha = 0.8
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
        
        return frame
    

#  get the watermark resized 
def _get_logo_resized(frame, watermark_color='white'):
    h, w = frame.shape[:2]
    logo_height = int(h * 0.055)
    logo_aspect_ratio = 512 / 128  # Assuming original logo size is 512x128
    logo_width = int(logo_height * logo_aspect_ratio)
    
    x_pos = w - logo_width - int(w * 0.05)
    y_pos = h - logo_height - int(h * 0.075)
    
    if watermark_color == 'blue':
        logo_resized = cv2.resize(utils.get_blue_watermark(), (logo_width, logo_height))
    else:
        logo_resized = cv2.resize(utils.get_white_watermark(), (logo_width, logo_height))
    
    return logo_resized, x_pos, y_pos, logo_width, logo_height
    

# round the corners of a frame
def _round_frame_corners(frame, radius=35):
        h, w = frame.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        draw_rounded_rectangle(mask, (0, 0), (w, h), 255, -1, radius)
        
        # create 3-channel mask
        mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        mask_3ch = mask_3ch.astype(np.float32) / 255.0
        
        return (frame.astype(np.float32) * mask_3ch).astype(np.uint8)
    