import numpy as np
from scipy.optimize import linear_sum_assignment


def iou(bb1, bb2):
    """
    Compute IoU between two boxes in [x1, y1, x2, y2] format.
    """
    x1 = max(bb1[0], bb2[0])
    y1 = max(bb1[1], bb2[1])
    x2 = min(bb1[2], bb2[2])
    y2 = min(bb1[3], bb2[3])

    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    inter = w * h

    area1 = max(0.0, bb1[2] - bb1[0]) * max(0.0, bb1[3] - bb1[1])
    area2 = max(0.0, bb2[2] - bb2[0]) * max(0.0, bb2[3] - bb2[1])

    union = area1 + area2 - inter
    if union <= 0:
        return 0.0
    return inter / union


class Track:
    def __init__(self, bbox, track_id):
        """
        bbox: [x1, y1, x2, y2]
        """
        self.bbox = np.array(bbox, dtype=float)
        self.id = track_id
        self.time_since_update = 0  # how many frames since last matched


class SORT:
    def __init__(self, max_age=10, iou_threshold=0.3):
        """
        max_age: how many frames to keep a track without seeing it
        iou_threshold: minimum IoU to consider detection and track as same object
        """
        self.max_age = max_age
        self.iou_threshold = iou_threshold
        self.tracks = []          # list of Track objects
        self.next_id = 0          # next ID to assign

    def update(self, detections):
        """
        detections: list of [x1, y1, x2, y2] for current frame
        returns: list of (bbox, track_id)
        """
        # Convert detections to numpy array
        if len(detections) == 0:
            dets = np.empty((0, 4), dtype=float)
        else:
            dets = np.array(detections, dtype=float)

        # If no existing tracks, create new ones from all detections
        if len(self.tracks) == 0:
            for d in dets:
                self.tracks.append(Track(d, self.next_id))
                self.next_id += 1
        else:
            # Build cost matrix = 1 - IoU
            if len(dets) > 0:
                cost_matrix = np.zeros((len(dets), len(self.tracks)), dtype=float)
                for i, det in enumerate(dets):
                    for j, trk in enumerate(self.tracks):
                        cost_matrix[i, j] = 1.0 - iou(det, trk.bbox)

                # Hungarian algorithm for best assignment
                row_ind, col_ind = linear_sum_assignment(cost_matrix)

                # Matched pairs
                matched_dets = set()
                matched_trks = set()

                for r, c in zip(row_ind, col_ind):
                    if 1.0 - cost_matrix[r, c] >= self.iou_threshold:
                        # Good match -> update track with this detection
                        self.tracks[c].bbox = dets[r]
                        self.tracks[c].time_since_update = 0
                        matched_dets.add(r)
                        matched_trks.add(c)

                # Unmatched detections -> new tracks
                for i in range(len(dets)):
                    if i not in matched_dets:
                        self.tracks.append(Track(dets[i], self.next_id))
                        self.next_id += 1

                # Unmatched tracks -> increase time_since_update
                for j, trk in enumerate(self.tracks):
                    if j not in matched_trks:
                        trk.time_since_update += 1
            else:
                # No detections this frame -> all existing tracks just age
                for trk in self.tracks:
                    trk.time_since_update += 1

        # Remove old tracks
        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]

        # Return all active tracks
        outputs = []
        for trk in self.tracks:
            outputs.append((trk.bbox.copy(), trk.id))

        return outputs

