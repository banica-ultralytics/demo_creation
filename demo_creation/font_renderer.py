import cv2
import numpy as np
from PIL import Image, ImageDraw
from .utils import get_font, get_color_pairs

label_padding = 0
letter_spacing_ratio = 0.12

# persistent class-label -> (box_color, text_color) map, so a class keeps the
# same color across every frame it appears in
_label_color_map = {}


def _contrast_text_color(box_color):
    brightness = 0.299 * box_color[2] + 0.587 * box_color[1] + 0.114 * box_color[0]
    return (104, 31, 17) if brightness > 186 else (255, 255, 255)


def set_label_colors(mapping):
    """Pre-assign colors to class labels. Values may be a BGR tuple or a
    (box_color, text_color) pair; missing text colors are derived from brightness."""
    for label, entry in mapping.items():
        if isinstance(entry, tuple) and len(entry) == 2 and isinstance(entry[0], tuple):
            _label_color_map[label] = entry
        else:
            _label_color_map[label] = (tuple(entry), _contrast_text_color(entry))


def reset_label_colors():
    _label_color_map.clear()

def _get_text_size(text, font_size):
    font = get_font(font_size)
    total_width = 0
    letter_spacing = max(1, int(font_size * letter_spacing_ratio))
    
    for i, char in enumerate(text):
        char_bbox = font.getbbox(char)
        char_width = char_bbox[2] - char_bbox[0]
        total_width += char_width
        if i < len(text) - 1:
            total_width += letter_spacing
    
    # Use same reference as _draw_text for consistent height
    ref_bbox = font.getbbox("Ay")
    height = ref_bbox[3] - ref_bbox[1]
    return (total_width, height)



def _draw_text(frame, text, position, font_size, color, thickness=1):
    font = get_font(font_size)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(frame_rgb)
    draw = ImageDraw.Draw(pil_image)
    rgb_color = (color[2], color[1], color[0])
    
    x, y = position
    letter_spacing = max(1, int(font_size * letter_spacing_ratio))
    
    # Get the top offset from a reference string that covers full ascent
    ref_bbox = font.getbbox("Ay")
    top_offset = ref_bbox[1]  # consistent y offset for all characters
    
    for char in text:
        char_bbox = font.getbbox(char)
        draw.text((x - char_bbox[0], y - top_offset), char, font=font, fill=rgb_color)
        char_width = char_bbox[2] - char_bbox[0]
        x += char_width + letter_spacing
    
    frame_bgr = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    frame[:] = frame_bgr[:]
    return frame
    
    
def draw_rounded_rectangle(img, pt1, pt2, color, thickness, radius=2):
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
            
        return img
            
        
            
def _draw_text_label(frame, text, track_id, position, font_size, box_color=None, text_color=None, thickness=2, padding=label_padding):
    if track_id is not None:
        text = f"{text}: {track_id}"
    
    text_w, text_h = _get_text_size(text, font_size)
    frame_h, frame_w = frame.shape[:2]
    
    label_w = text_w + padding * 2
    label_h = text_h + padding * 2
    
    # Clamp position to frame bounds
    x = max(0, min(position[0], frame_w - label_w))
    y = max(0, min(position[1], frame_h - label_h))
    
    y -= int(thickness / 2)  
    
    # Draw background rectangle — fills the full label area
    frame = draw_rounded_rectangle(
        frame,
        (x - int(thickness/2), y),
        (x + label_w, y + label_h),
        box_color,
        thickness=-1,
        radius=1
    )
    
    # Draw text centered within the label
    frame = _draw_text(
        frame,
        text,
        (x + padding, y + padding),
        font_size,
        text_color,
        thickness
    )
    return frame
    
    
    
def draw_box_annotations(frame, boxes, labels, track_ids, colors=None, font_size=20, box_thickness=2, padding = label_padding, radius = 2, add_ids = False):
    if not add_ids:
        track_ids = [None] * len(labels)

    if isinstance(colors, dict):
        set_label_colors(colors)
        colors = None

    brand_color_pairs = get_color_pairs() if colors is None else None

    for box, label, track_id in zip(boxes, labels, track_ids):
        x1, y1, x2, y2 = box

        if colors is None:
            if label not in _label_color_map:
                idx = len(_label_color_map) % len(brand_color_pairs)
                _label_color_map[label] = brand_color_pairs[idx]
            box_color, text_color = _label_color_map[label]
        elif isinstance(colors, tuple):
            box_color = colors
            text_color = _contrast_text_color(colors)
        else:
            box_color = colors[0]
            text_color = colors[1]
            
        text_box_height = _get_text_size(label, font_size)[1]

        frame = draw_rounded_rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            color=box_color,
            thickness=box_thickness,
            radius=1,
        )
        frame = _draw_text_label(
                frame,
                label,
                track_id,
                position=(x1, y1 - (text_box_height + padding + box_thickness)),  # position label above the box
                font_size=font_size,
                box_color=box_color,
                text_color=text_color,
                padding = padding
            )
    return frame



def draw_mask_annotations(segmentation_mask, label, color = None):
    pass
    
    
    
def draw_keypoint_annotations():
    pass



def draw_obb_annotations():
    pass
    
    
    

    
        