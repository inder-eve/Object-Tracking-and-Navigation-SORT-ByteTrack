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
    h = max(0.0, y2 - bb1[1])
    h = max(0.0, y2 - y1)
    inter = w * h

    area1 = max(0.0, bb1[2] - bb1[0]) * max(0.0, bb1[3] - bb1[1])
    area2 = max(0.0, bb2[2] - bb2[0]) * max(0.0, bb2[3] - bb2[1])

    union = area1 + area2 - inter
    if union <= 0:
        return 0.0
    return inter / union


class Track:
    def __init__(self, bbox, score, track_id):
        """
        bbox: [x1, y1, x2, y2]
        score: detection confidence
        """
        self.bbox = np.array(bbox, dtype=float)
        self.score = float(score)
        self.id = track_id
        self.age = 1                # how many frames since created
        self.time_since_update = 0  # frames since last matched

    def update(self, bbox, score):
        self.bbox = np.array(bbox, dtype=float)
        self.score = float(score)
        self.time_since_update = 0
        self.age += 1

    def mark_missed(self):
        self.time_since_update += 1
        self.age += 1


class ByteTrackTracker:
    def __init__(self,
                 high_thresh=0.5,
                 low_thresh=0.1,
                 iou_thresh=0.3,
                 max_age=30,
                 min_hits=3):
        """
        high_thresh: conf >= this is high-conf det
        low_thresh: low_thresh <= conf < high_thresh considered low-conf
        iou_thresh: min IoU for association
        max_age: remove track if not updated for this many frames
        min_hits: minimum age before we consider track as 'reliable'
        """
        self.high_thresh = high_thresh
        self.low_thresh = low_thresh
        self.iou_thresh = iou_thresh
        self.max_age = max_age
        self.min_hits = min_hits

        self.tracks = []
        self.next_id = 0

    def _associate(self, dets, tracks):
        """
        Associate detections with tracks based on IoU using Hungarian algorithm.
        dets: N x 4
        tracks: list of Track
        returns: matches, unmatched_dets, unmatched_tracks
        """
        if len(dets) == 0 or len(tracks) == 0:
            return [], list(range(len(dets))), list(range(len(tracks)))

        cost = np.zeros((len(dets), len(tracks)), dtype=float)
        for i, d in enumerate(dets):
            for j, t in enumerate(tracks):
                cost[i, j] = 1.0 - iou(d, t.bbox)

        row_ind, col_ind = linear_sum_assignment(cost)

        matches = []
        unmatched_dets = set(range(len(dets)))
        unmatched_tracks = set(range(len(tracks)))

        for r, c in zip(row_ind, col_ind):
            if 1.0 - cost[r, c] >= self.iou_thresh:
                matches.append((r, c))
                unmatched_dets.discard(r)
                unmatched_tracks.discard(c)

        return matches, list(unmatched_dets), list(unmatched_tracks)

    def update(self, detections):
        """
        detections: list of [x1, y1, x2, y2, conf]
        returns: list of (bbox, track_id) for active tracks this frame
        """
        if len(detections) == 0:
            # No detections: mark all tracks as missed
            for t in self.tracks:
                t.mark_missed()
            # Remove old tracks
            self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]
            # Return only currently visible tracks (time_since_update == 0)
            outputs = []
            for t in self.tracks:
                if t.time_since_update == 0 and t.age >= self.min_hits:
                    outputs.append((t.bbox.copy(), t.id))
            return outputs

        dets = np.array(detections, dtype=float)
        boxes = dets[:, :4]
        scores = dets[:, 4]

        # Split detections into high and low confidence
        high_inds = np.where(scores >= self.high_thresh)[0]
        low_inds = np.where((scores >= self.low_thresh) & (scores < self.high_thresh))[0]

        high_dets = boxes[high_inds]
        high_scores = scores[high_inds]

        low_dets = boxes[low_inds]
        low_scores = scores[low_inds]

        # STEP 1: associate high-conf detections with existing tracks
        matches, unmatched_high, unmatched_tracks = self._associate(high_dets, self.tracks)

        # Update matched tracks with high-conf detections
        for det_idx, trk_idx in matches:
            self.tracks[trk_idx].update(high_dets[det_idx], high_scores[det_idx])

        # STEP 2: try to associate remaining tracks with low-conf detections (rescue)
        if len(low_dets) > 0 and len(unmatched_tracks) > 0:
            rem_tracks = [self.tracks[i] for i in unmatched_tracks]
            low_matches, unmatched_low, unmatched_tracks2 = self._associate(low_dets, rem_tracks)

            for det_idx, rem_idx in low_matches:
                trk_idx = unmatched_tracks[rem_idx]
                self.tracks[trk_idx].update(low_dets[det_idx], low_scores[det_idx])

            final_unmatched_tracks = [unmatched_tracks[i] for i in unmatched_tracks2]
        else:
            final_unmatched_tracks = unmatched_tracks

        # Mark unmatched tracks as missed
        for trk_idx in final_unmatched_tracks:
            self.tracks[trk_idx].mark_missed()

        # STEP 3: create new tracks for unmatched high-conf detections
        for idx in unmatched_high:
            bbox = high_dets[idx]
            score = high_scores[idx]
            self.tracks.append(Track(bbox, score, self.next_id))
            self.next_id += 1

        # Remove tracks that have been missed for too long
        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]

        # Prepare output: only reliable and currently visible tracks
        outputs = []
        for t in self.tracks:
            if t.time_since_update == 0 and t.age >= self.min_hits:
                outputs.append((t.bbox.copy(), t.id))

        return outputs
