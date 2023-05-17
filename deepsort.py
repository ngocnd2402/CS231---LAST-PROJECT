from utils.deep_sort.deep_sort.tracker import Tracker 
from utils.deep_sort.tools import generate_detections 
from utils.deep_sort.deep_sort import nn_matching
from utils.deep_sort.deep_sort.detection import Detection
import numpy as np
 
class deepSORT_Tracker:
    tracker = None 
    encoder = None 
    tracks = None 
    
    def __init__(self):
        max_cosine_distance = 0.3
        encoder_model_filename = r'resources/networks/mars-small128.pb'
        nn_budget = None 
        metric = nn_matching.NearestNeighborDistanceMetric('cosine', max_cosine_distance, nn_budget)
        self.tracker = Tracker(metric, max_age=72)
        self.encoder = generate_detections .create_box_encoder(encoder_model_filename, batch_size=1)
        
    def update_tracks(self):
        tracks = []
        for track in self.tracker.tracks:
            if track.is_confirmed() and track.time_since_update <= 72:
                bbox = track.to_tlbr()
                id = track.track_id
                tracks.append((id, bbox))
        self.tracks = tracks

    def update(self, frame, detections):

        bboxes = np.asarray(detections)[:, :4]
        x1, y1, x2, y2 = bboxes.T
        bboxes = np.column_stack((x1, y1, x2-x1, y2-y1))
        scores = [d[-1] for d in detections]

        features = self.encoder(frame, bboxes)

        dets = []
        for bbox_id, bbox in enumerate(bboxes):
            dets.append(Detection(bbox, scores[bbox_id], features[bbox_id]))

        self.tracker.predict()
        self.tracker.update(dets)
        self.update_tracks()