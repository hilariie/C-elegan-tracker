from deep_sort.deep_sort.tracker import Tracker as DeepSortTracker
from deep_sort.tools import generate_detections as gdet
from deep_sort.deep_sort import nn_matching
from deep_sort.deep_sort.detection import Detection
import numpy as np


class Tracker:
    tracker = None
    encoder = None
    tracks = None

    def __init__(self):
        max_cosine_distance = 0.4
        nn_budget = None

        encoder_model_filename = 'model_data/mars-small128.pb'

        metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
        self.tracker = DeepSortTracker(metric, max_age=10000, max_iou_distance=0.9)
        self.encoder = gdet.create_box_encoder(encoder_model_filename, batch_size=1)

    def update(self, frame, detections, n_init=10):

        bboxes = np.asarray([d[:-2] for d in detections])
        bboxes[:, 2:] = bboxes[:, 2:] - bboxes[:, 0:2]
        scores = [d[-2] for d in detections]

        features = self.encoder(frame, bboxes)

        dets = []
        for bbox_id, bbox in enumerate(bboxes):
            class_id = detections[bbox_id][-1]
            dets.append(Detection(bbox, scores[bbox_id], features[bbox_id], class_id))

        self.tracker.predict()
        self.tracker.update(dets, n_init=n_init)
        self.update_tracks()

    def update_tracks(self):
        tracks = []
        for x, track in enumerate(self.tracker.tracks):
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox = track.to_tlbr()
            id = track.track_id
            class_id = track.class_id
            tracks.append(Track(id, bbox, class_id))

        self.tracks = tracks


class Track:
    track_id = None
    bbox = None
    class_id = None

    def __init__(self, id, bbox, class_id):
        self.track_id = id
        self.bbox = bbox
        self.class_id = class_id
