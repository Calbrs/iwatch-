"""
tracking.py — Per-camera multi-object tracking and identity stabilization.

Problems this solves (frame-level jitter/flicker):
  1. Detections over time are linked into tracklets (SORT-style IoU
     association), so boxes no longer jump between independent detections.
  2. Each tracklet's box is smoothed with an EMA and gated against wild
     jumps, so the badge/presence anchor stays stable.
  3. A doctor's identity is committed by voting over a sliding window of
     recent histograms with hysteresis — one weak or noisy frame can no
     longer flip the name, and a new identity only takes over after it
     convincingly outvotes the current one.

No images are stored — only bounding boxes, track ids and match statistics.
"""
import collections

import numpy as np


def iou(a, b):
    """Intersection-over-union of two [x1, y1, x2, y2] boxes."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    a_area = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    b_area = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return inter / (a_area + b_area - inter + 1e-9)


class IdentityTracker:
    """Sliding-window identity vote with hysteresis for one tracklet.

    Every detection round the tracklet observes the ranked candidates for
    its current box. A name is committed only after it convincingly wins a
    majority of the recent window, and it is not released (or replaced)
    until it has clearly gone weak — so single-frame histogram noise never
    flips the green badge.
    """

    def __init__(self, threshold, window=10, commit_frac=0.5,
                 release_frac=0.3, switch_margin=0.2, start_gate=4):
        self.threshold = threshold
        self.window = window
        self.commit_frac = commit_frac
        self.release_frac = release_frac
        self.switch_margin = switch_margin
        self.start_gate = start_gate
        self.history = collections.deque(maxlen=window)
        self.name = None

    def observe(self, candidates):
        """candidates: top-K [(name, dist), ...] sorted ascending by dist."""
        if candidates and candidates[0][1] <= self.threshold:
            self.history.append(candidates[0])
        else:
            self.history.append(("__unknown__", 1.0))

        if len(self.history) < self.start_gate:
            return self.name

        counts = collections.Counter()
        for name, _dist in self.history:
            counts[name] += 1
        n = len(self.history)

        support = {name: cnt / n for name, cnt in counts.items()}
        best_name = max(support, key=support.get)
        best_sup = support[best_name]

        cur = self.name
        if cur is None:
            if best_name != "__unknown__" and best_sup >= self.commit_frac:
                self.name = best_name
        elif best_name == cur:
            if support[cur] < self.release_frac:
                self.name = None
        elif best_name == "__unknown__":
            if support.get(cur, 0.0) < self.release_frac:
                self.name = None
        else:
            # Only hand over when the newcomer convincingly outvotes the
            # current identity — hysteresis prevents near-tie flicker.
            if (best_sup >= self.commit_frac and
                    best_sup - support.get(cur, 0.0) >= self.switch_margin):
                self.name = best_name
        return self.name

    def reset(self):
        self.history.clear()
        self.name = None


class Track:
    """One persistent person tracklet with a smoothed, gated box."""

    def __init__(self, track_id, box, identity_params,
                 box_alpha=0.55, jump_ratio=2.2, move_ratio=1.5):
        self.id = track_id
        self.box = [float(v) for v in box]
        self.lost = 0
        self.matched = 1
        self.age = 1
        self.box_alpha = box_alpha
        self.jump_ratio = jump_ratio
        self.move_ratio = move_ratio
        self.identity = IdentityTracker(**identity_params)

    def update(self, det_box):
        """Associate a new detection with this track (IoU already matched)."""
        self.lost = 0
        self.matched += 1
        self.age += 1

        d = [float(v) for v in det_box]
        b = self.box
        dw, dh = d[2] - d[0], d[3] - d[1]
        bw, bh = b[2] - b[0], b[3] - b[1]
        if dw <= 1.0 or dh <= 1.0 or bw <= 1.0 or bh <= 1.0:
            return

        dcx = (d[0] + d[2]) / 2 - (b[0] + b[2]) / 2
        dcy = (d[1] + d[3]) / 2 - (b[1] + b[3]) / 2
        jump = np.hypot(dcx, dcy)

        big = max(bw, dw)
        small = min(bw, dw)
        med = (bw + dw) / 2
        size_ok = med > 0 and (big / small) < self.jump_ratio and \
                  (bw * bh) / (dw * dh + 1e-9) < self.jump_ratio
        move_ok = jump <= self.move_ratio * max(bw, dw)

        if not size_ok or not move_ok:
            # Likely a bad or mismatched detection — freeze the box instead
            # of smearing the smooth tracklet.
            return

        a = self.box_alpha
        for i in range(4):
            b[i] = a * b[i] + (1.0 - a) * d[i]
        b[0], b[2] = min(b[0], b[2]), max(b[0], b[2])
        b[1], b[3] = min(b[1], b[3]), max(b[1], b[3])

    @property
    def active(self):
        return self.lost == 0

    @property
    def centroid(self):
        return ((self.box[0] + self.box[2]) / 2, (self.box[1] + self.box[3]) / 2)


class TrackManager:
    """Greedy-IoU tracker linking per-frame detections into tracklets."""

    def __init__(self, identity_params, iou_threshold=0.25, max_lost=10,
                 start_max_lost=4, hard_max_lost=30, max_tracks=8,
                 box_alpha=0.55):
        self.identity_params = identity_params
        self.iou_threshold = iou_threshold
        self.max_lost = max_lost
        self.start_max_lost = start_max_lost
        self.hard_max_lost = hard_max_lost
        self.max_tracks = max_tracks
        self.box_alpha = box_alpha
        self.tracks = {}
        self.next_id = 1

    def update(self, detections):
        """Associate `detections` ([x1, y1, x2, y2] boxes) with tracklets.

        Returns the list of tracklets matched this frame (active), sorted
        by id. Stale tracklets are pruned internally.
        """
        dets = [list(map(float, d)) for d in detections]
        remaining = set(range(len(dets)))

        for t in list(self.tracks.values()):
            best_i = -1
            best_iou = self.iou_threshold
            for i in remaining:
                io = iou(t.box, dets[i])
                if io > best_iou:
                    best_iou = io
                    best_i = i
            if best_i >= 0:
                remaining.discard(best_i)
                t.update(dets[best_i])
            else:
                t.lost += 1
                t.age += 1

        for i in sorted(remaining):
            if len(self.tracks) >= self.max_tracks:
                break
            t = Track(self.next_id, dets[i], self.identity_params,
                      box_alpha=self.box_alpha)
            self.next_id += 1
            self.tracks[t.id] = t

        for tid in [tid for tid, t in list(self.tracks.items())
                    if t.lost > self.hard_max_lost or
                       (t.lost > self.max_lost and t.matched >= 5) or
                       (t.lost > self.start_max_lost and t.matched < 5)]:
            del self.tracks[tid]

        return sorted((t for t in self.tracks.values() if t.active),
                      key=lambda t: t.id)