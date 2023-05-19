from utils.deep_sort.deep_sort.tracker import Tracker
from utils.deep_sort.tools import generate_detections
from utils.deep_sort.deep_sort import nn_matching
from utils.deep_sort.deep_sort.detection import Detection
import numpy as np


class deepSORT_Tracker:
    def __init__(self):
        self.max_cosine_distance = 0.3
        self.encoder_model_filename = r'resources/networks/mars-small128.pb'
        self.nn_budget = 100
        self.max_age = 30
        self.metric = nn_matching.NearestNeighborDistanceMetric('cosine', self.max_cosine_distance, self.nn_budget)
        self.tracker = Tracker(self.metric, max_age=self.max_age)
        self.encoder = generate_detections.create_box_encoder(self.encoder_model_filename, batch_size=1)

    def update_tracks(self):
        self.tracks = [(track.track_id, track.to_tlbr()) for track in self.tracker.tracks
                       if track.is_confirmed() and track.time_since_update <= self.max_age]

    def update(self, frame, detections):
        bboxes = np.asarray(detections)[:, :4]
        x1, y1, x2, y2 = bboxes.T
        bboxes = np.column_stack((x1, y1, x2 - x1, y2 - y1))
        scores = np.asarray(detections)[:, -1]

        features = self.encoder(frame, bboxes)

        dets = [Detection(bbox, score, feature) for bbox, score, feature in zip(bboxes, scores, features)]

        self.tracker.predict()
        self.tracker.update(dets)
        self.update_tracks()
