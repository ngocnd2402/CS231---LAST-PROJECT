import os
import cv2
import numpy as np
from ultralytics import YOLO
from super_gradients.training import models
from super_gradients.common.object_names import  Models
from deepsort import deepSORT_Tracker 
model = models.get("yolo_nas_l", pretrained_weights="coco")

# Load video and initialize video writer
video_path = os.path.join('.', 'data', 'people.mp4')
video_out_path = os.path.join('.', 'people_out.mp4')
cap = cv2.VideoCapture(video_path)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
cap_out = cv2.VideoWriter(video_out_path, fourcc, fps, (width, height))

# Initialize deepSORT tracker and other parameters
tracker = deepSORT_Tracker()
detection_threshold = 0.3
pts = {}

frame_id = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    results = model.predict(frame)

    for result in results:
        detections = []
        for i, r in enumerate(result.prediction.bboxes_xyxy):
            x1, y1, x2, y2 = r
            x1 = int(x1)
            x2 = int(x2)
            y1 = int(y1)
            y2 = int(y2)
            score = result.prediction.confidence[i]
            labels = result.prediction.labels[i]
            class_id = int(labels)
            if class_id == 0 and score > detection_threshold:
                detections.append([x1, y1, x2, y2, score])
        tracker.update(frame, detections)
        for track in tracker.tracks:
            track_id, bbox = track
            x1, y1, x2, y2 = bbox
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (102,0,204), 2 )
            cv2.putText(frame, str(track_id), (int(x1), int(y1)), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,0,255),2)

            if track_id not in pts:
                pts[track_id] = [(int((x1+x2)/2), int((y1+y2)/2))]
            else:
                pts[track_id].append((int((x1+x2)/2), int((y1+y2)/2)))

            color = (255, 160, 122)
            for j in range(1, len(pts[track_id])):
                if pts[track_id][j - 1] is None or pts[track_id][j] is None:
                    continue
                thickness = int(np.sqrt(64 / float(j + 1)) * 2)
                cv2.line(frame, pts[track_id][j-1], pts[track_id][j], color, thickness)

    cap_out.write(frame)
    frame_id += 1

cap.release()
cap_out.release()
cv2.destroyAllWindows()
