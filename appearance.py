"""
appearance.py — Shared appearance histogram and matching module.

Used by both enroll.py and tracker.py to avoid duplicating logic.
Histograms are numeric arrays only — no images are stored or reconstructable.
"""
import re

import cv2
import numpy as np


def sanitize_filename(name):
    """Turn a doctor name into a safe filename slug."""
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", name.strip().lower())
    return slug.strip("_") or "profile"


def compute_torso_histogram(frame, box, config):
    """Extract HSV color histogram of the torso region from a person bbox.

    This frame is never persisted — used live only.
    The returned histogram is a flat numeric array (not an image).
    """
    x1, y1, x2, y2 = map(int, box)
    h_frame, w_frame = frame.shape[:2]
    x1 = max(0, min(x1, w_frame - 1))
    x2 = max(0, min(x2, w_frame))
    y1 = max(0, min(y1, h_frame - 1))
    y2 = max(0, min(y2, h_frame))

    top_ratio, bottom_ratio = config["torso_crop_ratio"]
    bbox_h = y2 - y1
    torso_y1 = y1 + int(bbox_h * top_ratio)
    torso_y2 = y1 + int(bbox_h * bottom_ratio)
    torso = frame[torso_y1:torso_y2, x1:x2]

    if torso.size == 0:
        return None

    hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
    cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
    return hist.flatten().tolist()


def match_profile_candidates(hist, profiles, top_k=3):
    """Rank all enrolled profiles for one histogram by distance (ascending).

    Returns [(name, dist), ...] for the closest `top_k` profiles. The raw
    distances let the tracker accumulate evidence over several frames
    instead of trusting a single comparison.
    """
    if hist is None or not profiles:
        return []

    query = np.array(hist, dtype=np.float32).reshape(1, -1)
    scored = []
    for name, profile_hist in profiles.items():
        stored = np.array(profile_hist, dtype=np.float32).reshape(1, -1)
        dist = cv2.compareHist(query, stored, cv2.HISTCMP_BHATTACHARYYA)
        scored.append((dist, name))
    scored.sort(key=lambda p: p[0])
    return [(name, dist) for dist, name in scored[:top_k]]


def match_profile(hist, profiles, threshold):
    """Compare a histogram against all enrolled profiles.

    Returns the doctor name if the best match is within threshold,
    otherwise returns None (unknown person).
    """
    best = match_profile_candidates(hist, profiles, top_k=1)
    if best and best[0][1] <= threshold:
        return best[0][0]
    return None
