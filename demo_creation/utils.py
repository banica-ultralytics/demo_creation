from pathlib import Path
from PIL import ImageFont 
import cv2


_CURRENT_DIR = Path(__file__).parent

# branding and font resource paths
_white_watermark_path = Path.joinpath(_CURRENT_DIR, 'ultralytics_branding', 'Ultralytics_full_white.png')
_blue_watermark_path = Path.joinpath(_CURRENT_DIR, 'ultralytics_branding', 'Ultralytics_full_blue.png')
_font_path = Path.joinpath(_CURRENT_DIR, 'ultralytics_branding', 'Archivo-VariableFont.ttf')




# getters for resource paths
def get_white_watermark():
    return cv2.imread(str(_white_watermark_path), cv2.IMREAD_UNCHANGED) 
            
 
def get_blue_watermark():
    return cv2.imread(str(_blue_watermark_path), cv2.IMREAD_UNCHANGED)

def get_font(size):
    return ImageFont.truetype(str(_font_path), size)    



# color palettes
_brand_colors = {
    'bright-blue': (255, 66, 4),           
    'vivid-sky-blue': (235, 219, 11),      
    'light-gray': (243, 243, 243),      
    'robin-egg-blue': (183, 223, 0),       
    'pure-midnight': (104, 31, 17),        
    'candy-pink': (221, 111, 255),        
    'sunburnt-cyclops': (79, 68, 255),    
    'electric-lime': (0, 237, 204),        
    'malachite': (68, 243, 0),             
    'electric-purple': (255, 0, 189),      
    'blue-bolt': (255, 180, 0),           
    'deep-magenta': (186, 0, 221),         
    'aqua': (255, 255, 0),                 
    'yellow-green': (0, 192, 40),          
    'medium-spring-green': (179, 255, 1),  
    'blue-violet': (255, 36, 125),         
    'philippine-violet': (136, 0, 123),    
    'electric-pink': (140, 27, 255),       
    'smashed-pumpkin': (47, 141, 252),     
    'spring-bud': (11, 255, 162),           
    'neon-yellow': (37, 255, 255),      
    'neon-pink': (218, 100, 255),       
    'white': (255, 255, 255),
    'dark-gray': (50, 50, 50),
    'black': (0, 0, 0)
}

_box_colors = list(_brand_colors.values())
_text_color = [_brand_colors['pure-midnight'] if (0.299 * rgb[2] + 0.587 * rgb[1] + 0.114 * rgb[0]) > 186 else _brand_colors['white'] for rgb in _box_colors]

def get_color_pairs():
    return list(zip(_box_colors, _text_color))