import os
import cv2
import numpy as np
from ultralytics import YOLO
from super_gradients.training import models
from super_gradients.common.object_names import  Models
from deepsort import deepSORT_Tracker 
import warnings
warnings.filterwarnings("ignore")

model = models.get("yolo_nas_l", pretrained_weights="coco")
frame_dir_path = r"datasets\MOT17\train\MOT17-05-SDP\img1"
output_txt_path = "output.txt"
video_out_path = os.path.join('.', 'output.mp4')

frame_paths = sorted([os.path.join(frame_dir_path, f) for f in os.listdir(frame_dir_path) if f.endswith('.jpg')])

first_frame = cv2.imread(frame_paths[0])
height, width, _ = first_frame.shape

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = 30
cap_out = cv2.VideoWriter(video_out_path, fourcc, fps, (width, height))
pts = {} 
tracker = deepSORT_Tracker()
detection_threshold = 0.6

with open(output_txt_path, 'w') as output_file:
    for i, frame_path in enumerate(frame_paths):
        frame = cv2.imread(frame_path)
        print(f"Processing frame {i}")
        results = model.predict(frame)

        detections = []
        for result in results:
            for i, r in enumerate(result.prediction.bboxes_xyxy):
                x1, y1, x2, y2 = r
                x1 = int(x1)
                x2 = int(x2)
                y1 = int(y1)
                y2 = int(y2)
                score = result.prediction.confidence[i]
                labels = result.prediction.labels[i]
                class_id = int(labels)
                if score > detection_threshold:
                    detections.append([x1, y1, x2, y2, score])

        tracker.update(frame, detections)
        for track in tracker.tracks:
            track_id, bbox = track
            x1, y1, x2, y2 = bbox
            output_file.write(f"{i},{track_id},{x1},{y1},{int(x2-x1)},{int(y2-y1)}\n")
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (102,0,204), 2 )
            cv2.putText(frame, str(track_id), (int(x1), int(y1)), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,0,255),2)
            if track_id not in pts:
                pts[track_id] = [(int((x1+x2)/2), int((y1+y2)/2))]
            else:
                pts[track_id].append((int((x1+x2)/2), int((y1+y2)/2)))

            # Draw lines connecting the points in pts dictionary
            color = (255, 160, 122)
            for j in range(1, len(pts[track_id])):
                if pts[track_id][j - 1] is None or pts[track_id][j] is None:
                    continue
                thickness = max(1, int(np.sqrt(64 / float(j + 1)) * 2))
                cv2.line(frame, pts[track_id][j-1], pts[track_id][j], color, thickness)
        cap_out.write(frame)

# Sort the output file by track_id
with open(output_txt_path, 'r') as output_file:
    lines = output_file.readlines()
    lines.sort(key=lambda x: int(x.split(',')[1]))

with open(output_txt_path, 'w') as output_file:
    output_file.writelines(lines)

cap_out.release()
cv2.destroyAllWindows()
