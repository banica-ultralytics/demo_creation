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


def process_yolo_results(results):
    result = results[0]
    output = {}

    if result.boxes is not None and len(result.boxes):
        output['boxes'] = result.boxes.xyxy.cpu().numpy().astype(int).reshape(-1, 4)
        output['labels'] = [result.names[int(cls)] for cls in result.boxes.cls.cpu().numpy()]
        output['confidences'] = result.boxes.conf.cpu().numpy()

        if result.boxes.id is not None:
            output['track_ids'] = result.boxes.id.cpu().numpy().astype(int)
        else:
            output['track_ids'] = [None] * len(output['labels'])
    
    if result.masks is not None:
        output['masks'] = result.masks.data.cpu().numpy()

    if result.keypoints is not None:
        output['keypoints'] = result.keypoints.data.cpu().numpy()

    if result.obb is not None:
        output['obbs'] = result.obb.xyxyxyxy.cpu().numpy()

    return output
    
    
def plot_boxes_from_folder(fpath, classes=None):
    if not os.path.isdir(fpath):
        raise NotADirectoryError(f"Provided path is not a directory: {fpath}")
    
    for file in os.listdir(fpath):
        if file.endswith('.txt'):
            with open(os.path.join(fpath, file), 'r') as f:
                lines = f.readlines()
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    cls_id, x_center, y_center, width, height = map(float, parts[:5])
                    if classes and int(cls_id) >= len(classes):
                        continue
                    print(f"Class: {classes[int(cls_id)] if classes else cls_id}, Box: ({x_center}, {y_center}, {width}, {height})")
    

