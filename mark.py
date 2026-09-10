"""
mark.py — MARK: Adaptive Persistent Object Tracking & Identity Engine.

MARK is the persistent tracking, identity-stability, recovery and adaptive
learning layer BETWEEN detection/recognition and the UI badge.

Layer separation:
    Detector (YOLO)        -> per-frame person boxes  (observation source)
    MARK engine            -> persistent tracks + persistent identity
    Badge / UI             -> smoothed / predicted position (follows MARK)

Design tenets (from the MARK specification):
  1. Identity is NEVER decided from a single frame: it is built from a bank of
     templates and temporal evidence accumulated over the track's life.
  2. A track is a persistent object. It survives short detection failures via
     motion prediction, a RECOVERING state and re-identification, instead of
     dying and respawning as "Unknown".
  3. Adaptive learning is progressive but reversible: candidates must pass a
     quality gate, are validated over time, and are disabled/rolled back when
     they stop performing. Original enrollment is always protected.
  4. Confidence is a FUSED score (appearance + temporal support + spatial /
     motion consistency + track history) with hysteresis: high threshold to
     CONFIRM, low threshold to release. This prevents flicker.
  5. Recognition cadence is decoupled from tracking cadence: once CONFIRMED, a
     track's identity is only re-verified periodically or when something
     changes.

No images are ever persisted — only HSV histograms, boxes and match stats.
"""
import collections
import os
import threading
import time

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Track states (explicit state machine — makes behaviour easy to debug)
# ---------------------------------------------------------------------------
SEARCH = "SEARCH"
TENTATIVE = "TENTATIVE"
STABLE_UNKNOWN = "STABLE_UNKNOWN"
CONFIRMED = "CONFIRMED"
UNCERTAIN = "UNCERTAIN"
RECOVERING = "RECOVERING"
LOST = "LOST"
TERMINATED = "TERMINATED"

_TEMPLATE_STATES = ("protected", "trusted", "candidate", "disabled")

# All knobs are tunable via the "mark" block in config.json. Defaults are
# conservative; calibrate with real data as the spec requests.
MARK_DEFAULTS = {
    # association / tracking
    "iou_threshold": 0.25,
    "max_tracks": 8,
    "badge_alpha": 0.55,
    "jump_ratio": 2.2,
    "move_ratio": 1.5,
    "motion_alpha": 0.35,
    "reid_gate": 0.20,             # IoU vs a re-id track's PREDICTED box to re-attach
    "reid_distance": 120.0,        # px; centre-distance gate for recovery re-id
    # automatic discovery / registration
    "stable_frames": 12,           # matched frames before a tag is STABLE_UNKNOWN
    "max_observations": 200,       # per-unknown-track observation ceiling
    "unknown_reid_threshold": 0.42,# dist; strong match to a remembered unknown tag
    "unknown_reid_margin": 0.03,   # dist gap required before swapping to another tag
    "unknown_mint_lag": 15,        # frames to keep probing memory before minting a
                                   #   brand-new tag for a tagless track (so a
                                   #   returning person never gets a new number
                                   #   just because the first frame was poor)
    "unknown_mem_min_box_h": 48.0, # smaller unknown crops still feed re-id memory
    "unknown_mem_sharp_frac": 0.15,# memory sharpness floor as a fraction of learning
    # identity state machine
    "match_threshold": 0.35,       # Bhattacharyya; above this = unknown
    "confirm_conf": 0.55,          # fused confidence needed to CONFIRM
    "uncertain_conf": 0.45,        # below CONFIRMED => UNCERTAIN (still shown)
    "release_conf": 0.35,          # below this the identity is released
    "vote_window": 10,
    "commit_frac": 0.55,
    "release_frac": 0.3,
    "switch_margin": 0.2,
    "start_gate": 4,
    # lifecycle
    "tentative_max_lost": 6,       # un-identified track lost before it dies
    "recover_max_lost": 12,        # CONFIRMED lost before RECOVERING -> LOST
    "terminate_lost": 30,          # total lost before it is TERMINATED
    "recognition_period": 5,       # re-verify identity every N frames once confirmed
    # fused-confidence weights (sum ~ 1)
    "weight_appearance": 0.40,
    "weight_temporal": 0.25,
    "weight_spatial": 0.20,
    "weight_history": 0.15,
    # adaptive learning
    "learn_sharpness_min": 45.0,   # Laplacian variance floor on the torso crop
    "learn_quality_min": 0.45,
    "learn_identity_conf_min": 0.55,
    "learn_min_frames": 5,         # how long an appearance must be stable
    "learn_min_box_h": 28.0,       # minimum torso crop height for learning
    "learn_near_dup": 0.14,        # below this distance a template is a duplicate
    "learn_ambiguous_margin": 0.15,
    "learn_max_templates": 14,
    "learn_validate_hits": 5,
    "template_rollback_ratio": 0.45,
    # occlusion
    "occlusion_shrink": 0.55,      # box falls below this share of typical size
}


def bhattacharyya(a, b):
    query = np.asarray(a, dtype=np.float32).reshape(1, -1)
    stored = np.asarray(b, dtype=np.float32).reshape(1, -1)
    if query.shape != stored.shape:
        return 1.0
    return float(cv2.compareHist(query, stored, cv2.HISTCMP_BHATTACHARYYA))


def _scale_label(box_or_h, frame_h):
    """Classify observation distance so templates get scale-tagged."""
    box_h = box_or_h
    try:
        if len(box_or_h) >= 4:
            box_h = box_or_h[3] - box_or_h[1]
    except TypeError:
        pass
    if frame_h <= 0:
        return "medium"
    rel = box_h / frame_h
    if rel >= 0.45:
        return "close"
    if rel >= 0.22:
        return "medium"
    return "far"


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


class MotionModel:
    """EMA velocity estimator; predicts where a moving track is next.

    This is what lets the badge follow a smooth, predicted trajectory
    instead of lurching from raw detector box to raw detector box.
    """

    def __init__(self, alpha=0.35):
        self.alpha = alpha
        self.vx = 0.0
        self.vy = 0.0
        self.dt_mean = 0.033
        self._last = None
        self._last_t = None

    def update(self, box, t=None):
        cx = (box[0] + box[2]) / 2
        cy = (box[1] + box[3]) / 2
        if self._last is not None and t is not None and self._last_t is not None:
            dt = max(1e-3, t - self._last_t)
            vx = (cx - self._last[0]) / dt
            vy = (cy - self._last[1]) / dt
            a = self.alpha
            self.vx = a * self.vx + (1 - a) * vx
            self.vy = a * self.vy + (1 - a) * vy
            self.dt_mean = 0.7 * self.dt_mean + 0.3 * dt
        self._last = (cx, cy)
        self._last_t = t

    def predict(self, box, dt=None):
        dt = self.dt_mean if dt is None else dt
        return [box[0] + self.vx * dt, box[1] + self.vy * dt,
                box[2] + self.vx * dt, box[3] + self.vy * dt]


class Template:
    """One representative observation for an identity.

    State lifecycle: candidate -> trusted (after validation) -> disabled
    (on rollback). ``protected`` is the original enrollment and can never be
    disabled or evicted.
    """
    __slots__ = ("hist", "scale", "quality", "conf", "state", "hits",
                 "bad_hits", "created", "last_used")

    def __init__(self, hist, scale, quality, conf, state="candidate"):
        self.hist = hist
        self.scale = scale
        self.quality = quality
        self.conf = conf
        self.state = state
        self.hits = 0
        self.bad_hits = 0
        self.created = time.time()
        self.last_used = 0


class IdentityBank:
    """Persistent per-identity template bank.

    Stores ONLY numeric histograms (no images). Handles: multi-scale and
    multi-angle diversity, adaptive learning with a quality gate, candidate
    validation, and per-template rollback/disable. The original enrollment
    template is always protected.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.identities = {}      # name -> {"protected": [hist], "templates": [Template]}
        self.unknown_memory = {}  # "unknown_0001" -> [hist, ...] (bounded, session)
        self._tag_seq = 0
        self._lock = threading.RLock()

    # ---- automatic discovery -------------------------------------------------
    def new_unknown_tag(self):
        with self._lock:
            self._tag_seq += 1
            return f"unknown_{self._tag_seq:04d}"

    def remember_unknown(self, tag, hist, limit=60):
        if not tag or hist is None:
            return
        with self._lock:
            bucket = self.unknown_memory.setdefault(tag, [])
            if bucket and min(bhattacharyya(hist, h) for h in bucket) < 0.15:
                return
            bucket.append(hist)
            if len(bucket) > limit:
                del bucket[:len(bucket) - limit]

    def best_unknown(self, hist, limit=80):
        """Re-identification for unassigned people: find a remembered unknown
        tag whose appearance strongly matches. Returns (tag, dist) or None."""
        if hist is None:
            return None
        best_tag, best_dist = None, 2.0
        with self._lock:
            for tag, hists in self.unknown_memory.items():
                if not tag:
                    continue
                d = min(bhattacharyya(hist, h) for h in hists[:limit])
                if d < best_dist:
                    best_tag, best_dist = tag, d
        if best_tag is not None and best_dist <= self.cfg["unknown_reid_threshold"]:
            return best_tag, best_dist
        return None

    def forget_unknown(self, tag):
        with self._lock:
            self.unknown_memory.pop(tag, None)

    def create_identity(self, name, observations, top_trusted=4):
        """Build a brand-new identity profile from a track's collected
        observations (auto-discovery assignment).

        The single best observation becomes the protected enrollment template;
        the next best validated ones become trusted; the rest become candidates
        awaiting live validation. Frames are never stored — histograms only.
        Returns True on success, False if there is nothing usable.
        """
        if not observations:
            return False
        cfg = self.cfg
        scored = sorted(
            [o for o in observations
             if o.get("hist") is not None and o.get("valid", True)],
            key=lambda o: o.get("confidence", 0.0),
            reverse=True)
        if not scored:
            return False
        with self._lock:
            old = self.identities.get(name)
            if old is not None:
                return False
            protected = scored[0]["hist"]
            bucket = {"protected": [protected], "templates": []}
            for o in scored[1:]:
                state = "trusted" if len(bucket["templates"]) < top_trusted else "candidate"
                tpl = Template(o["hist"], o.get("scale", "medium"),
                               o.get("quality", 0.5), o.get("confidence", 0.5),
                               state=state)
                bucket["templates"].append(tpl)
                if len(bucket["templates"]) >= cfg["learn_max_templates"]:
                    break
            self.identities[name] = bucket
        return True

    # ---- loading / persistence -------------------------------------------------
    def load_dir(self, profiles_dir):
        if not os.path.isdir(profiles_dir):
            return
        for fname in os.listdir(profiles_dir):
            if not fname.endswith(".json"):
                continue
            try:
                with open(os.path.join(profiles_dir, fname), "r", encoding="utf-8") as f:
                    data = json_load(f)
                self._adopt_profile(data)
            except Exception as exc:
                print(f"WARNING: MARK failed to load profile {fname}: {exc}")

    def _adopt_profile(self, data):
        name = str(data.get("name") or "").strip()
        if not name:
            return
        with self._lock:
            bucket = self.identities.setdefault(
                name, {"protected": [], "templates": []})
            tpls = data.get("templates")
            if isinstance(tpls, list) and tpls:
                for t in tpls:
                    if not isinstance(t, dict) or "hist" not in t:
                        continue
                    tpl = Template(
                        t["hist"],
                        t.get("scale", "medium"),
                        t.get("quality", 0.5),
                        t.get("conf", 0.5),
                        state=t.get("state", "trusted") if t.get("state") != "protected" else "trusted",
                    )
                    tpl.hits = t.get("hits", 0)
                    tpl.bad_hits = t.get("bad_hits", 0)
                    bucket["templates"].append(tpl)
                # fall back to plain "histogram" only if templates are absent
                if data.get("histogram") and not bucket["protected"]:
                    for tpl in bucket["templates"]:
                        if bhattacharyya(tpl.hist, data["histogram"]) < 0.05:
                            tpl.state = "protected"
                            break
                return
            # legacy single-histogram profile -> the protected enrollment template
            hist = data.get("histogram")
            if hist is not None and not bucket["protected"]:
                bucket["protected"].append(hist)

    def save_dir(self, profiles_dir, names=None):
        os.makedirs(profiles_dir, exist_ok=True)
        from appearance import sanitize_filename
        with self._lock:
            for name in list(self.identities):
                if names is not None and name not in names:
                    continue
                bucket = self.identities[name]
                tpls = [
                    {
                        "hist": tpl.hist,
                        "scale": tpl.scale,
                        "quality": tpl.quality,
                        "conf": tpl.conf,
                        "state": tpl.state,
                        "hits": tpl.hits,
                        "bad_hits": tpl.bad_hits,
                    }
                    for tpl in bucket["templates"]
                    if tpl.state in ("protected", "trusted")
                ]
                data = {"name": name,
                        "histogram": bucket["protected"][0] if bucket["protected"] else None,
                        "templates": tpls}
                path = os.path.join(profiles_dir,
                                    sanitize_filename(name) + ".json")
                with open(path, "w", encoding="utf-8") as fp:
                    json_dump(data, fp)

    # ---- enrollment / deletion -------------------------------------------------
    def upsert_enrolled(self, name, hist):
        with self._lock:
            bucket = self.identities.setdefault(
                name, {"protected": [], "templates": []})
            bucket["protected"] = [hist]
            # keep learned templates; they still describe this identity

    def forget(self, name):
        with self._lock:
            self.identities.pop(name, None)

    # ---- matching --------------------------------------------------------------
    def match(self, hist, top_k=3):
        """Rank identities for one observation.

        Returns [(name, dist, tpl)] ascending by distance, where ``tpl`` is
        the best-matching Template (or None for the protected enrollment).
        An identity's score is its BEST template, so a far/rotated learned
        template lifts recognition of the same person at a new distance — the
        core of multi-scale/multi-angle adaptation.
        """
        if hist is None:
            return []
        with self._lock:
            scored = []
            for name, bucket in self.identities.items():
                best_tpl = None
                best_dist = 2.0
                for hist_pr in bucket["protected"]:
                    d = bhattacharyya(hist, hist_pr)
                    if d < best_dist:
                        best_dist, best_tpl = d, None
                for tpl in bucket["templates"]:
                    if tpl.state not in ("protected", "trusted"):
                        continue
                    d = bhattacharyya(hist, tpl.hist)
                    if d < best_dist:
                        best_dist, best_tpl = d, tpl
                if best_dist < 2.0:
                    scored.append((name, best_dist, best_tpl))
            scored.sort(key=lambda s: s[1])
            return scored[:top_k]

    # ---- adaptive learning -----------------------------------------------------
    def learn(self, name, hist, scale, quality, conf):
        """Add a valid NEW observation as a learning candidate.

        Quality gate (blur/visibility/confidence), novelty vs. duplicates and
        ambiguity (must not also match a different identity) gate admission.
        """
        with self._lock:
            bucket = self.identities.get(name)
            if bucket is None:
                return False
            cfg = self.cfg

            def _all_hists():
                for hist_pr in bucket["protected"]:
                    yield hist_pr
                for t in bucket["templates"]:
                    yield t.hist

            # novelty: near-duplicate of an existing representation -> skip
            if min(bhattacharyya(hist, h) for h in _all_hists()) <= cfg["learn_near_dup"]:
                return False

            # ambiguity: another identity also strongly matches this sample
            own_best = min(bhattacharyya(hist, h) for h in _all_hists())
            for other, ob in self.identities.items():
                if other == name:
                    continue
                other_tpls = ob["protected"] + [
                    t.hist for t in ob["templates"]
                    if t.state in ("trusted", "protected")]
                if not other_tpls:
                    continue
                other_best = min(bhattacharyya(hist, h) for h in other_tpls)
                if other_best < own_best - cfg["learn_ambiguous_margin"]:
                    return False

            if quality < cfg["learn_quality_min"] or conf < cfg["learn_identity_conf_min"]:
                return False

            capped = [t for t in bucket["templates"]
                      if t.state in ("candidate", "trusted", "disabled")]
            if len(capped) >= cfg["learn_max_templates"]:
                # drop the worst disabled template if any, otherwise refuse
                dis = [t for t in bucket["templates"] if t.state == "disabled"]
                if dis:
                    dis.sort(key=lambda t: t.quality)
                    bucket["templates"].remove(dis[0])
                else:
                    return False

            bucket["templates"].append(Template(hist, scale, quality, conf))
            return True

    def confirm_use(self, name, tpl, decided_name, correct, validate_bias=1):
        """Per-template performance accounting, used ON DECISION.

        - Successful matches raise a candidate towards 'trusted'.
        - A trusted template that keeps winning while the DECISION goes to
          another identity accumulates 'bad_hits' and is disabled (rolled
          back). Protected templates are never disabled.
        """
        if tpl is None:
            return
        with self._lock:
            if correct:
                tpl.hits += 1
            elif decided_name is not None and correct is False:
                tpl.bad_hits += 1
            if tpl.state == "protected":
                return
            if tpl.state == "candidate":
                if tpl.hits >= self.cfg["learn_validate_hits"] and \
                        tpl.bad_hits == 0:
                    tpl.state = "trusted"
                    tpl.last_used = time.time()
            elif tpl.state == "trusted":
                total = tpl.hits + tpl.bad_hits
                if total >= 3 and tpl.bad_hits / total > self.cfg["template_rollback_ratio"]:
                    tpl.state = "disabled"
                    print(f"MARK: rolled back template for '{name}' "
                          f"({tpl.scale}, quality={tpl.quality:.2f}) after "
                          f"{tpl.bad_hits}/{total} bad matches")
            tpl.last_used = time.time()


class MarkTrack:
    """One persistent tracked object.

    Survives individual bad frames: when the detector misses it, the track
    goes RECOVERING with a motion-predicted position and is re-attached or
    re-identified when the person reappears — never silently forgotten.
    """

    def __init__(self, track_id, box, cfg, t=None):
        self.id = track_id
        self.cfg = cfg
        self.box = [float(v) for v in box]
        self.lost = 0
        self.matched = 1
        self.age = 1
        self.state = TENTATIVE
        self.identity = None
        self.identity_conf = 0.0
        self.votes = collections.deque(maxlen=int(cfg["vote_window"]))
        self.conf_history = collections.deque(maxlen=20)
        self.motion = MotionModel(alpha=cfg["motion_alpha"])
        self.last_best = None      # (name, dist, tpl) from bank.match
        self.last_hist = None
        self.in_zone = False
        self.occluded = False
        self.typical_h = None      # EMA of historical box height (for occlusion)
        self._learn_seen = collections.Counter()  # name -> stable frame count
        self.last_recognized = 0
        # automatic discovery / registration
        self.tag = None                     # "unknown_0001" until assigned
        self.tag_settled = False    # locked to a remembered tag (no more swaps)
        self.observations = collections.deque(
            maxlen=int(cfg.get("max_observations", 200)))
        self.views = collections.Counter()  # (scale, view) -> count
        self.obs_best_q = 0.0
        self.obs_best_conf = 0.0
        if t is not None:
            self.motion.update(self.box, t)

    # ---- observation collection (auto-discovery) ---------------------------
    def view_label(self):
        """Coarse pose/view guess from the torso aspect ratio."""
        w = self.box[2] - self.box[0]
        h = self.box[3] - self.box[1]
        if h <= 0:
            return "frontal"
        r = w / max(1.0, h)
        if r < 0.35:
            return "profile"
        if r >= 0.6:
            return "frontal"
        return "threequarter"

    def add_observation(self, hist, scale, quality, confidence, t, frame_count):
        """Quality + novelty-gated automatic observation for this unknown track.

        No near-duplicates are stored, so the admin ends up with diverse
        views/scales instead of hundreds of identical frames. Returns True if
        a new observation was accepted.
        """
        if hist is None or not scale:
            return False
        for o in self.observations:
            if bhattacharyya(hist, o["hist"]) < 0.12:
                return False
        self.observations.append({
            "ts": t, "frame": frame_count,
            "bbox": [float(v) for v in self.box],
            "hist": hist, "scale": scale,
            "view": self.view_label(),
            "quality": quality, "confidence": confidence, "valid": True,
        })
        self.views[(scale, self.view_label())] += 1
        self.obs_best_q = max(self.obs_best_q, quality)
        self.obs_best_conf = max(self.obs_best_conf, confidence)
        return True

    # ---- geometry -----------------------------------------------------------
    @property
    def centroid(self):
        return ((self.box[0] + self.box[2]) / 2, (self.box[1] + self.box[3]) / 2)

    @property
    def predicted_centroid(self):
        return ((self.box[0] + self.box[2] + self.motion.vx * self.motion.dt_mean * 2) / 2,
                (self.box[1] + self.box[3] + self.motion.vy * self.motion.dt_mean * 2) / 2)

    @property
    def badge_box(self):
        """Smoothed/predicted box the UI badge follows (MARK, not the detector)."""
        if self.lost > 0:
            return self.motion.predict(self.box, 2 * self.motion.dt_mean)
        return list(self.box)

    def _smooth_box(self, det_box):
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
        size_ok = med > 0 and (big / small) < self.cfg["jump_ratio"] and \
                  (bw * bh) / (dw * dh + 1e-9) < self.cfg["jump_ratio"]
        move_ok = jump <= self.cfg["move_ratio"] * max(bw, dw)
        if not size_ok or not move_ok:
            return
        a = self.cfg["badge_alpha"]
        for i in range(4):
            b[i] = a * b[i] + (1 - a) * d[i]
        b[0], b[2] = min(b[0], b[2]), max(b[0], b[2])
        b[1], b[3] = min(b[1], b[3]), max(b[1], b[3])

    def associate(self, det_box, t):
        """Attach a real detection. Returns True if the geometry was accepted."""
        before = list(self.box)
        self._smooth_box(det_box)
        accepted = self.box != before
        self.lost = 0
        self.matched += 1
        self.age += 1
        self.motion.update(det_box, t)
        bh = self.box[3] - self.box[1]
        if self.typical_h is None:
            self.typical_h = bh
        else:
            self.typical_h = 0.9 * self.typical_h + 0.1 * bh
        return accepted

    def mark_lost(self):
        self.lost += 1
        self.age += 1
        cfg = self.cfg
        if self.state == CONFIRMED and self.lost >= cfg["recover_max_lost"]:
            self.state = LOST
        elif self.state == RECOVERING and self.lost >= cfg["recover_max_lost"]:
            self.state = LOST
        elif self.state in (STABLE_UNKNOWN, TENTATIVE, SEARCH):
            limit = (cfg["tentative_max_lost"] if self.state in (TENTATIVE, SEARCH)
                     else cfg["recover_max_lost"])
            if self.lost >= limit:
                self.state = LOST
        elif self.state == LOST and self.lost >= cfg["terminate_lost"]:
            self.state = TERMINATED
        elif self.state == CONFIRMED and self.lost == 1:
            self.state = RECOVERING

    def should_recognize(self, frame_count, force=False):
        """Tracking cadence != recognition cadence (spec 16)."""
        if force or self.identity is None:
            return True
        if self.state in (UNCERTAIN, RECOVERING):
            return True
        return (frame_count - self.last_recognized) >= int(self.cfg["recognition_period"])

    # ---- identity observation -----------------------------------------------
    def observe(self, candidates, hist, frame_count):
        """Consume bank.match() output; temporal vote with hysteresis.

        Returns the current (possibly updated) identity name. The identity
        survives single bad frames and only switches after evidence is both
        strong AND persistent (switch_margin + identity locking). Sets fused
        identity_conf. Learning is gated by the caller, not here.
        """
        cfg = self.cfg
        th = cfg["match_threshold"]
        best = candidates[0] if candidates else None
        self.last_best = best
        if hist is not None:
            self.last_hist = hist

        if best is not None and best[1] <= th:
            self.votes.append((best[0], 1.0 - best[1]))
        else:
            self.votes.append((None, 0.0))
        self.last_recognized = frame_count

        n = len(self.votes)
        support = collections.Counter()
        mean_conf = collections.defaultdict(float)
        for nm, cf in self.votes:
            support[nm] += 1
            if nm is not None:
                mean_conf[nm] += cf
        for nm in list(mean_conf):
            mean_conf[nm] /= max(1, support[nm])

        cur = self.identity
        top = [(nm, support[nm] / max(1, n))
               for nm in support if nm is not None]
        top.sort(key=lambda x: x[1], reverse=True)

        if cur is None:
            if n >= cfg["start_gate"] and top:
                nm, rat = top[0]
                if rat >= cfg["commit_frac"] and \
                        mean_conf[nm] >= cfg["confirm_conf"]:
                    self.identity = nm
                    self._set_conf(nm, rat, mean_conf[nm])
                    self.state = CONFIRMED
        else:
            cur_rat = support.get(cur, 0) / max(1, n)
            cur_conf = self.identity_conf

            switch = False
            if top and top[0][0] != cur:
                nm, rat = top[0]
                if (rat >= cfg["commit_frac"] and
                        rat - cur_rat >= cfg["switch_margin"] and
                        mean_conf[nm] >= cfg["confirm_conf"] and
                        cur_conf < cfg["confirm_conf"] * 0.9):
                    # identity locking: a CONFIRMED track is not hijacked by a
                    # single strong alternative; evidence must be persistent AND
                    # the current identity already weak/holding by a cliff.
                    switch = True
            if switch:
                self.identity = top[0][0]
                self._set_conf(self.identity, top[0][1],
                               mean_conf[self.identity])
                self.state = CONFIRMED
            else:
                # hysteresis boundary: only release when the FUSED confidence
                # AND the vote support both fall below their release floors.
                fused = self._fused({
                    "appearance": cur_conf,
                    "temporal": cur_rat,
                    "spatial": 0.5 if support.get(cur, 0) else 0.2,
                    "history": min(1.0, self.matched / 20),
                })
                if fused < cfg["release_conf"] and \
                        cur_rat < cfg["release_frac"]:
                    self.identity = None
                    if self.state == CONFIRMED:
                        self.state = UNCERTAIN
                self.identity_conf = max(self.identity_conf, fused)

        occluded = False
        if self.typical_h and self.box[3] - self.box[1] > 1:
            if (self.box[3] - self.box[1]) < cfg["occlusion_shrink"] * self.typical_h:
                occluded = True
        self.occluded = occluded
        return self.identity

    def _set_conf(self, name, ratio, mean_app_conf):
        fused = self._fused({
            "appearance": mean_app_conf,
            "temporal": ratio,
            "spatial": 0.6,
            "history": min(1.0, self.matched / 20),
        })
        self.identity_conf = fused
        self.conf_history.append(fused)

    def _fused(self, parts):
        cfg = self.cfg
        w = (cfg["weight_appearance"], cfg["weight_temporal"],
             cfg["weight_spatial"], cfg["weight_history"])
        total = sum(w)
        if total <= 0:
            return parts.get("appearance", 0.0)
        return (w[0] * parts["appearance"] +
                w[1] * parts["temporal"] +
                w[2] * parts["spatial"] +
                w[3] * parts["history"]) / total

    def stable_learning_window(self, name):
        """Count consecutive stable frames for this identity (learning needs it)."""
        self._learn_seen[name] += 1
        for other in list(self._learn_seen):
            if other != name:
                self._learn_seen[other] = max(0, self._learn_seen[other] - 1)
        return self._learn_seen[name]


class MarkManager:
    """Orchestrates everything MARK does per camera per frame."""

    def __init__(self, cfg, bank):
        self.cfg = cfg
        self.bank = bank
        self.tracks = {}
        self.next_id = 1

    # ---- association --------------------------------------------------------
    def update(self, dets, t=None):
        """Associate detections with tracks; predict/recover lost ones.

        Returns the list of tracks that were matched this frame (active).
        """
        dets = [list(map(float, d)) for d in dets]
        cfg = self.cfg
        remaining = set(range(len(dets)))

        active = []
        for trk in list(self.tracks.values()):
            pred = trk.motion.predict(trk.box, trk.motion.dt_mean)
            best_i, best_score = -1, cfg["iou_threshold"]
            for i in remaining:
                det = dets[i]
                score = iou(pred, det)
                # recovery: also allow centre-distance re-id near the prediction
                if trk.lost > 0 and trk.identity is not None:
                    pc = trk.predicted_centroid
                    dc = ((det[0] + det[2]) / 2, (det[1] + det[3]) / 2)
                    ddist = np.hypot(pc[0] - dc[0], pc[1] - dc[1])
                    if score < cfg["reid_gate"] and ddist <= cfg["reid_distance"]:
                        score = max(score, cfg["reid_gate"])
                        # spatial-only re-id; identity re-affirms in observe()
                if score > best_score:
                    best_score, best_i = score, i
            if best_i >= 0:
                remaining.discard(best_i)
                trk.associate(dets[best_i], t)
                # recovered from a detection gap: restore a live state
                if trk.state in (LOST, RECOVERING):
                    trk.state = CONFIRMED if trk.identity else TENTATIVE
                if trk.state == TENTATIVE and trk.identity:
                    trk.state = CONFIRMED
                # automatic discovery: promote to STABLE_UNKNOWN after enough
                # consistent matched frames (still never auto-assigns identity)
                if (trk.state == TENTATIVE and trk.identity is None and
                        trk.matched >= cfg["stable_frames"]):
                    trk.state = STABLE_UNKNOWN
                active.append(trk)
            else:
                trk.mark_lost()

        for i in sorted(remaining):
            if len(self.tracks) >= cfg["max_tracks"]:
                break
            trk = MarkTrack(self.next_id, dets[i], cfg, t)
            self.next_id += 1
            self.tracks[trk.id] = trk
            # No tag yet: the camera loop assigns one after the re-id probe,
            # so a returning person reuses a remembered tag instead of burning
            # a brand-new unknown_XXXX number.
            active.append(trk)

        for tid in [tid for tid, trk in list(self.tracks.items())
                    if trk.state == TERMINATED]:
            del self.tracks[tid]

        return sorted((t for t in active if t.state not in (TERMINATED, LOST)),
                      key=lambda t: t.id)

    def forget_track_identity(self, name):
        for trk in self.tracks.values():
            if trk.identity == name:
                trk.identity = None
                trk.identity_conf = 0.0
                trk.votes.clear()

    def maybe_reid_unknown(self, trk, hist):
        """Re-identification for UNASSIGNED people: when a fresh track's
        appearance strongly matches a remembered unknown tag, reuse that tag
        instead of minting a new unknown_XXXX (no duplicate unknowns).

        A tag adopted from memory is SETTLED for the life of the track, so a
        person who reappears after leaving keeps their old tag and the badge
        never flips between two tags frame-to-frame. A freshly MINTED tag stays
        unsettled so it can still converge to a remembered tag if a clearly
        better match appears (camera-crossing / far-first reappearances)."""
        if trk.identity is not None or hist is None:
            return trk.tag
        if getattr(trk, "tag_settled", False):
            return trk.tag
        res = self.bank.best_unknown(hist)
        if res is None:
            return trk.tag
        tag, dist = res
        if trk.tag is None:
            trk.tag = tag
            trk.tag_settled = True
            return trk.tag
        if tag == trk.tag:
            return trk.tag
        # swap only when the candidate is clearly better than the current tag's
        # own best match, otherwise keep the current label (no flapping).
        cur_dist = 2.0
        bucket = self.bank.unknown_memory.get(trk.tag)
        if bucket:
            cur_dist = min(bhattacharyya(hist, h) for h in bucket[:80])
        if dist <= cur_dist - self.cfg["unknown_reid_margin"]:
            self.bank.forget_unknown(trk.tag)
            trk.tag = tag
            trk.tag_settled = True
        return trk.tag


def quality_gate(frame, box, cfg):
    """Evaluate one observation for learning safety (sharpness/size coverage).

    Returns (quality 0..1, ok: bool).
    """
    x1, y1, x2, y2 = map(int, box)
    hf, wf = frame.shape[:2]
    x1 = max(0, min(x1, wf - 1))
    x2 = max(0, min(x2, wf))
    y1 = max(0, min(y1, hf - 1))
    y2 = max(0, min(y2, hf))
    crop_h = y2 - y1
    crop_w = x2 - x1
    if crop_h <= 2 or crop_w <= 2:
        return 0.0, False
    if crop_h < cfg["learn_min_box_h"]:
        return 0.0, False

    crop = frame[y1:y2, x1:x2].astype(np.float32)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    lap = cv2.Laplacian(gray, cv2.CV_32F)
    sharpness = float(np.mean(lap * lap))
    size_frac = min(1.0, crop_h / 120.0)
    sharp_frac = min(1.0, sharpness / cfg["learn_sharpness_min"])
    quality = 0.45 * size_frac + 0.40 * sharp_frac + 0.15 * min(1.0, crop_w / 80.0)
    return quality, quality >= cfg["learn_quality_min"]


def quality_gate_lite(frame, box, cfg):
    """Relaxed gate for the unknown re-id memory only (never for learning).

    Re-id matching must still see a person that reappears FAR away, so far
    crops are allowed as long as they are recognisably a person (minimum box
    size + a sharpness floor) rather than noise/garbage. Detections are YOLO
    person boxes, so a small-but-sharp crop is a usable appearance sample."""
    x1, y1, x2, y2 = map(int, box)
    hf, wf = frame.shape[:2]
    x1 = max(0, min(x1, wf - 1))
    x2 = max(0, min(x2, wf))
    y1 = max(0, min(y1, hf - 1))
    y2 = max(0, min(y2, hf))
    crop_h = y2 - y1
    crop_w = x2 - x1
    if crop_h <= 2 or crop_w <= 2:
        return False
    if crop_h < cfg["unknown_mem_min_box_h"]:
        return False
    crop = frame[y1:y2, x1:x2].astype(np.float32)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    lap = cv2.Laplacian(gray, cv2.CV_32F)
    sharpness = float(np.mean(lap * lap))
    floor = cfg["learn_sharpness_min"] * cfg["unknown_mem_sharp_frac"]
    return sharpness >= floor


def mark_config(config):
    """Merged MARK config. 'mark' block wins; legacy tracking_* keys are only
    a fallback so an existing install upgrades without editing config.json."""
    cfg = dict(MARK_DEFAULTS)
    block = config.get("mark")
    if isinstance(block, dict):
        cfg.update({k: v for k, v in block.items() if v is not None})
    # honour existing config.json keys ONLY when the mark block omits them
    legacy = {
        "match_threshold": "appearance_match_threshold",
        "vote_window": "tracking_window",
        "commit_frac": "tracking_commit_frac",
        "release_frac": "tracking_release_frac",
        "start_gate": "tracking_start_gate",
        "iou_threshold": "mark_iou_threshold",
    }
    for key, cfgkey in legacy.items():
        if key not in cfg and cfgkey in config:
            cfg[key] = config[cfgkey]
    return cfg


def json_load(f):
    """Small helpers so mark.py has no hard dependency on tracker globals."""
    import json
    return json.load(f)


def json_dump(obj, fp):
    import json
    json.dump(obj, fp, indent=4)