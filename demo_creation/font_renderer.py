import cv2
import numpy as np
from PIL import Image, ImageDraw
from .utils import get_font, get_color_pairs

label_padding = 6

def _get_text_size(text, font_size, font_path):
    font = get_font(font_size)
    # Calculate width with letter spacing
    total_width = 0
    letter_spacing = max(1, int(font_size * 0.12))  # 12% of font size (1.5x bigger)
    
    for i, char in enumerate(text):
        char_bbox = font.getbbox(char)
        char_width = char_bbox[2] - char_bbox[0]
        total_width += char_width
        if i < len(text) - 1:  # exclude last char
            total_width += letter_spacing
    
    bbox = font.getbbox(text)
    height = bbox[3] - bbox[1]
    return (total_width, height)



def _draw_text(self, frame, text, position, font_size, color, thickness=1):
        """Draw text with Archivo font on OpenCV frame with improved spacing"""
        font = self.get_font(font_size)
        
        # Convert OpenCV frame to PIL Image
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        draw = ImageDraw.Draw(pil_image)
        
        # Convert BGR color to RGB
        if isinstance(color, tuple) and len(color) == 3:
            print('IS BGR')
            rgb_color = (color[2], color[1], color[0])  # BGR to RGB
        else:
            print('IS RGB')
            rgb_color = color
        
        # Get text metrics for proper positioning
        x, y = position
        bbox = font.getbbox(text)
    
    
        text_top = bbox[1]      # Distance from baseline to top (usually negative)
        text_bottom = bbox[3]   # Distance from baseline to bottom (usually positive)
        text_center_from_baseline = (text_top + text_bottom) / 2
        
        # Adjust y position: move up by the center offset from baseline
        adjusted_y = y - text_center_from_baseline
        
        letter_spacing = max(1, int(font_size * 0.12))  # 12% of font size (1.5x bigger)
        
        # Draw text with letter spacing
        current_x = x
        for char in text:
            draw.text((current_x, adjusted_y), char, font=font, fill=rgb_color)
            char_width = font.getbbox(char)[2] - font.getbbox(char)[0]
            current_x += char_width + letter_spacing
        
        # Convert back to OpenCV format
        frame_bgr = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        frame[:] = frame_bgr[:]
        
        return frame
    
    
    
def draw_rounded_rectangle(img, pt1, pt2, color, thickness, radius=10):
        x1, y1 = pt1
        x2, y2 = pt2
        
        # Ensure radius doesn't exceed rectangle dimensions
        radius = min(radius, abs(x2 - x1) // 2, abs(y2 - y1) // 2)
        
        if thickness == -1:  # filled rectangle
            cv2.rectangle(img, (x1 + radius, y1), (x2 - radius, y2), color, -1)
            cv2.rectangle(img, (x1, y1 + radius), (x2, y2 - radius), color, -1)
            
            cv2.circle(img, (x1 + radius, y1 + radius), radius, color, -1)
            cv2.circle(img, (x2 - radius, y1 + radius), radius, color, -1)
            cv2.circle(img, (x1 + radius, y2 - radius), radius, color, -1)
            cv2.circle(img, (x2 - radius, y2 - radius), radius, color, -1)
        else:   # bounding box
            cv2.line(img, (x1 + radius, y1), (x2 - radius, y1), color, thickness)
            cv2.line(img, (x1 + radius, y2), (x2 - radius, y2), color, thickness)
            cv2.line(img, (x1, y1 + radius), (x1, y2 - radius), color, thickness)
            cv2.line(img, (x2, y1 + radius), (x2, y2 - radius), color, thickness)
            
            cv2.ellipse(img, (x1 + radius, y1 + radius), (radius, radius), 180, 0, 90, color, thickness)
            cv2.ellipse(img, (x2 - radius, y1 + radius), (radius, radius), 270, 0, 90, color, thickness)
            cv2.ellipse(img, (x1 + radius, y2 - radius), (radius, radius), 90, 0, 90, color, thickness)
            cv2.ellipse(img, (x2 - radius, y2 - radius), (radius, radius), 0, 0, 90, color, thickness)
            
            

# draw text and colored label 
def _draw_text_label(frame, text, position, font_size, box_color = None, text_color = None, thickness=1):
    font = get_font(font_size)
        
    frame_h, frame_w = frame.shape[:2]
    text_w, text_h = _get_text_size(text, font_size, font)
    
    label_w = text_w + label_padding * 2
    label_h = text_h + label_padding * 2
    
    # keep text label inside frame boundaries
    x = max(0, min(position[0], frame_w - label_w))
    y = max(0, min(position[1], frame_h - label_h))
        
    frame = draw_rounded_rectangle(
        frame, 
        (x, y), 
        (x + label_w, y + label_h), 
        box_color,
        thickness=-1,
        radius=5
    )
    
    frame = _draw_text(
        frame, 
        text, 
        (x + label_padding, y + label_padding), 
        font_size, 
        text_color,
        thickness
    )
    
    
def draw_box_annotations(frame, boxes, labels, colors=None):
    
    if colors is None:
        brand_color_pairs = get_color_pairs()
        
        class_labels = list(set(labels))
        label_color_map = {}
        for i, label in enumerate(class_labels):
            label_color_map[label] = brand_color_pairs[i % len(brand_color_pairs)]
    
    for box, label in zip(boxes, labels):
        x1, y1, x2, y2 = box

        if colors is None:
            box_color, text_color = label_color_map[label]
        elif isinstance(colors, tuple):
            box_color = colors
            text_color = (255, 255, 255) if (0.299 * colors[2] + 0.587 * colors[1] + 0.114 * colors[0]) < 186 else (104, 31, 17)
        else:
            box_color = colors[0]
            text_color = colors[1]

        frame = draw_rounded_rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            color=box_color,
            thickness=2,
            radius=10
        )
        frame = _draw_text_label(
                frame,
                label,
                position=(x1, y1 - 25),
                font_size=16,
                box_color=box_color,
                text_color=text_color,
            )
    return frame



def draw_mask_annotations(segmentation_mask, label, color = None):
    pass
    
    
    
def draw_keypoint_annotations():
    pass



def draw_obb_annotations():
    pass
    
    
    

    
        