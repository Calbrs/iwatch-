"""
tracker.py — Multi-camera presence tracking, zone logic, SQLite logging,
Flask API, live previews, and browser-device camera linking.

Camera sources are either:
  - wired: a webcam index (int) or RTSP URL (str) captured with OpenCV, or
  - remote: any device that opens /remotecam in its browser, grants camera
    access (getUserMedia), and streams JPEG frames to this server over a
    WebSocket. The server registers each device as a virtual camera and
    runs the same detection/zone pipeline on its feed (up to 50 devices).

Design:
  - One background thread per camera publishes which enrolled doctors are
    present (matched + inside that camera's chair zone) into a shared
    presence table, plus a rendered preview frame.
  - A single state-machine thread union-aggregates presence across ALL
    cameras (a doctor moving from one camera's zone to another's keeps one
    continuous session) and runs the confirm/debounce logic + DB writes.
  - Temporal smoothing (PresenceFilter) stabilizes detections so single
    missed or jittery frames don't flip status.

No images/frames are ever written to disk.
"""
import json
import os
import secrets
import sqlite3
import threading
import time
import traceback
from datetime import datetime

import cv2
import numpy as np
from flask import Flask, Response, abort, jsonify, render_template, request
from simple_websocket import ConnectionClosed, Server
from ultralytics import YOLO

from appearance import compute_torso_histogram, match_profile_candidates, sanitize_filename
from tracking import TrackManager

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(BASE_DIR, "config.json")
PROFILES_DIR = os.path.join(BASE_DIR, "profiles")
DB_PATH = os.path.join(BASE_DIR, "logs.db")
SCHEMA_PATH = os.path.join(BASE_DIR, "db_schema.sql")
LINKS_PATH = os.path.join(BASE_DIR, "links.json")

app = Flask(__name__, template_folder=os.path.join(BASE_DIR, "templates"),
            static_folder=os.path.join(BASE_DIR, "static"))

profile_lock = threading.Lock()
PROFILES = {}

presence_lock = threading.Lock()
camera_presence = {}

detection_log_lock = threading.Lock()
_last_detection_log = {}
DETECTION_LOG_INTERVAL = 10.0

state_lock = threading.Lock()
states = {}

live_lock = threading.Lock()
live_status = {}

preview_lock = threading.Lock()
preview_frames = {}
preview_times = {}
preview_versions = {}

LIVE_AFTER_SECONDS = 4.0

# --- Remote-stream telemetry logging ---------------------------------------
# Every frame the linked phone SENDS and every frame the server RECEIVES is
# appended (with timestamps) to a JSONL file so the two sides can be compared:
#   evt=t              single-frame / session event with arbitrary fields
#   evt=recv           a frame arrived at the server (server clock)
#   evt=send_side      the phone's own send log batch for the last second
#                      (phone clock, ms since page load) with its seq numbers
#   evt=recv_summary   rolling receive rate printed every 10 s
# Set STREAM_LOG=0 to disable (writes are tiny: ~1 line per frame ~60 B).
STREAM_LOG = os.environ.get("STREAM_LOG", "1").strip().lower() not in ("0", "false", "no", "off")
STREAM_LOG_PATH = os.path.join(BASE_DIR, os.environ.get("STREAM_LOG_PATH", "stream.log.jsonl"))
_stream_log_lock = threading.Lock()
_stream_codes = {}


def _log_stream(evt, **kwargs):
    if not STREAM_LOG:
        return
    rec = {"evt": evt, "t": round(time.time(), 4)}
    rec.update(kwargs)
    try:
        with _stream_log_lock:
            with open(STREAM_LOG_PATH, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec) + "\n")
    except Exception:
        pass

enroll_lock = threading.Lock()
enroll_jobs = {}

cam_lock = threading.Lock()
cameras = []
camera_labels = {}

inference_lock = threading.Lock()

remote_source_lock = threading.Lock()
remote_sources = {}
remote_stop_events = {}

link_lock = threading.Lock()
link_store = {}


def _persist_links():
    """Save device links to disk so they survive a tracker restart.

    Runtime-only fields (the live websocket, camera id, frame counts) are
    intentionally not saved; they are reset on every boot.
    """
    try:
        with open(LINKS_PATH, "w", encoding="utf-8") as f:
            json.dump([
                {
                    "code": code,
                    "label": lk.get("label", ""),
                    "status": "pending",
                    "created": lk.get("created", time.time()),
                }
                for code, lk in link_store.items()
            ], f, indent=2)
    except Exception as exc:
        print(f"WARNING: failed to save links.json: {exc}")


def _load_links():
    """Restore links created before a restart, reset to a clean pending state."""
    global link_store
    if not os.path.exists(LINKS_PATH):
        return
    try:
        with open(LINKS_PATH, "r", encoding="utf-8") as f:
            saved = json.load(f)
        for entry in saved:
            code = str(entry.get("code", "")).upper()
            if not code:
                continue
            link_store.setdefault(code, {
                "label": entry.get("label", "Remote Camera"),
                "camera_id": None,
                "status": "pending",
                "frames": 0,
                "last_frame": None,
                "active_socket": False,
                "socket": None,
                "created": entry.get("created", time.time()),
            })
        if link_store:
            print(f"RESTORED {len(link_store)} device link(s) from links.json")
    except Exception as exc:
        print(f"WARNING: failed to load links.json: {exc}")

CONFIG = {}
MODEL = None

STATE_OUT = "OUT_OF_ZONE"
STATE_ENTERING = "ENTERING"
STATE_IN = "IN_ZONE"
STATE_EXITING = "EXITING"

PRESENCE_STALE_SECONDS = 1.5
MAX_REMOTE_CAMERAS = 50
ENROLL_FRAMES_DEFAULT = 25

LINK_CODE_ALPHABET = "ABCDEFGHJKMNPQRSTUVWXYZ23456789"

DEFAULT_ZONE = [[0, 0], [640, 0], [640, 480], [0, 480]]


def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def persist_config(config):
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4)


def load_profiles_from_disk():
    """Load all enrolled appearance profiles into {name: histogram} dict."""
    profiles = {}
    if os.path.isdir(PROFILES_DIR):
        for fname in os.listdir(PROFILES_DIR):
            if fname.endswith(".json"):
                path = os.path.join(PROFILES_DIR, fname)
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    profiles[data["name"]] = data["histogram"]
                except Exception as exc:
                    print(f"WARNING: failed to load profile {fname}: {exc}")
    return profiles


def init_db():
    """Create the database and sessions table if they don't already exist."""
    conn = sqlite3.connect(DB_PATH)
    with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
        schema = f.read()
    conn.executescript(schema)
    conn.commit()
    conn.close()


def point_in_polygon(point, polygon):
    """Ray-casting point-in-polygon test."""
    x, y = point
    n = len(polygon)
    inside = False
    px, py = polygon[0]
    for i in range(1, n + 1):
        cx, cy = polygon[i % n]
        if y > min(py, cy):
            if y <= max(py, cy):
                if x <= max(px, cx):
                    if py != cy:
                        xinters = (y - py) * (cx - px) / (cy - py) + px
                    else:
                        xinters = px
                    if px == cx or x <= xinters:
                        inside = not inside
        px, py = cx, cy
    return inside


def new_state():
    return {
        "state": STATE_OUT,
        "session_id": None,
        "clock_in": None,
        "enter_timer_start": None,
        "exit_timer_start": None,
    }


class RemoteFrameSource:
    """Thread-safe single-slot frame queue fed by a browser over WebSocket."""

    def __init__(self):
        self._frame = None
        self._lock = threading.Lock()
        self._event = threading.Event()

    def push(self, bgr):
        with self._lock:
            self._frame = bgr
        self._event.set()

    def read(self):
        if not self._event.wait(timeout=2.0):
            return False, None
        self._event.clear()
        with self._lock:
            frame = self._frame
        return True, frame


class PresenceFilter:
    """Smooths per-doctor presence so single missed frames don't cause flicker.

    A name is published as present only after `min_hits` consecutive
    detections, and stays present until `max_misses` consecutive misses.
    """

    def __init__(self, min_hits=2, max_misses=3):
        self.min_hits = min_hits
        self.max_misses = max_misses
        self.hits = {}
        self.misses = {}
        self.present = set()

    def update(self, detected_names):
        result = set(self.present)

        for name in detected_names:
            self.misses[name] = 0
            self.hits[name] = self.hits.get(name, 0) + 1
            if name not in self.present and self.hits[name] >= self.min_hits:
                result.add(name)

        considered = set(self.present) | set(self.hits)
        for name in considered:
            if name not in detected_names:
                self.misses[name] = self.misses.get(name, 0) + 1
                self.hits[name] = 0
                if name in self.present and self.misses[name] >= self.max_misses:
                    result.discard(name)

        self.present = result
        return result


# Live-preview pacing.
# The linked phone may capture/send faster than the receiving browser can play
# (e.g. 35 frames in vs 30 frames out). Queueing those surplus frames makes the
# feed lag further and further behind real time. Instead we publish and stream
# at a fixed PREVIEW_FPS and DROP any frames published in between, so playback
# always stays at the consumer's pace (same principle WhatsApp/WebRTC use).
PREVIEW_FPS = max(1.0, float(os.environ.get("PREVIEW_FPS", "30")))
_MIN_PREVIEW_INTERVAL = 1.0 / PREVIEW_FPS
_RAW_PREVIEW_QUALITY = float(os.environ.get("CLOUD_PREVIEW_QUALITY", "0.82"))
PREVIEW_JPEG_QUALITY = max(1, min(95, int(_RAW_PREVIEW_QUALITY * 100 + 0.5)))


def publish_preview(camera_id, frame):
    """Share a REAL rendered frame for a camera with the preview stream.

    This frame is never persisted — used live only. A new frame simply
    overwrites the previous one in memory, and marks the camera as live/connected.
    The version counter lets the MJPEG generator skip work when nothing changed.

    Publishing stores the newest frame into a single latest-wins slot — a new
    frame simply overwrites the previous one, and the generator reads whatever is
    there when it is ready to emit. Intermediate frames are consequently dropped
    at the consumer side while NO queue ever builds, so playback stays live even
    when the phone captures faster than the browser can play (e.g. 35 vs 30 fps).
    """
    with preview_lock:
        preview_frames[camera_id] = frame.copy()
        preview_times[camera_id] = time.time()
        preview_versions[camera_id] = preview_versions.get(camera_id, 0) + 1


def publish_presence(camera_id, doctor_names):
    """Publish which enrolled doctors this camera currently sees in its zone."""
    clean = {name for name in doctor_names if name}
    with presence_lock:
        camera_presence[camera_id] = {"names": clean, "ts": time.time()}


def camera_loop(cam, stop_event):
    """Per-camera thread: capture, detect, stabilize, publish presence + preview.

    This frame is never persisted — used live only.
    """
    camera_id = cam["id"]
    polygon = np.array(cam["chair_zone_polygon"], dtype=np.int32)
    zone_reflected = False
    frame_skip = cam["frame_skip"]
    confidence = cam["detection_confidence"]

    src = remote_sources.get(camera_id)
    if src is None:
        print(f"ERROR: remote source '{camera_id}' is missing.")
        return

    identity_params = dict(
        threshold=CONFIG["appearance_match_threshold"],
        window=CONFIG.get("tracking_window", 10),
        commit_frac=CONFIG.get("tracking_commit_frac", 0.5),
        release_frac=CONFIG.get("tracking_release_frac", 0.3),
        start_gate=CONFIG.get("tracking_start_gate", 4),
    )
    tracker = TrackManager(
        identity_params=identity_params,
        max_lost=CONFIG.get("tracking_max_lost", 10),
    )
    presence_filter = PresenceFilter(min_hits=2, max_misses=3)
    current_badges = {}
    frame_count = 0
    det_errors = 0

    consecutive_errors = 0
    while not stop_event.is_set():
        try:
            ok, frame = src.read()

            if not ok:
                time.sleep(0.02)
                continue

            frame_count += 1

            # Remote frames were flipped at the source (front camera is a
            # mirror of reality). Reflect the configured chair zone along x so
            # it still lines up with the physical chairs in the mirrored view.
            if cam.get("remote") and not zone_reflected:
                polygon[:, 0] = frame.shape[1] - polygon[:, 0]
                zone_reflected = True

            # Redraw badges from the last detection round, then publish the
            # frame IMMEDIATELY. Detection (YOLO) is the slowest step on the
            # box; if we published only after it finished, the preview would be
            # a full detection-cycle (often 150-400 ms) behind the live scene.
            # Publishing first means the viewer always gets the freshest frame
            # and badges simply lag one detection round behind.
            for name, box in current_badges.items():
                _draw_name_badge(frame, name, box)
            publish_preview(camera_id, frame)

            if frame_count % frame_skip == 0:
                try:
                    results = None
                    with inference_lock:
                        results = MODEL.predict(frame, conf=confidence, verbose=False)
                    persons = [r for r in results[0].boxes if int(r.cls) == 0]

                    with profile_lock:
                        profs = PROFILES

                    # Link frames into persistent tracklets first, so the boxes the
                    # identities run on are stable across time and do not jump.
                    active = tracker.update([b.xyxy[0].cpu().numpy() for b in persons])

                    in_zone_names = set()
                    for t in active:
                        cx, cy = t.centroid
                        if not point_in_polygon((cx, cy), polygon):
                            continue
                        hist = compute_torso_histogram(frame, t.box, CONFIG)
                        candidates = (match_profile_candidates(hist, profs, top_k=3)
                                      if hist is not None else [])
                        name = t.identity.observe(candidates)
                        if name:
                            in_zone_names.add(name)

                    # Presence is only confirmed after consecutive votes, so the
                    # green badge below does not flicker on single missed frames.
                    presence = presence_filter.update(in_zone_names)
                    publish_presence(camera_id, presence)

                    # Badge boxes come from smooth tracklets and persist through
                    # the short debounce window above even if a frame is dropped.
                    current_badges = {
                        t.identity.name: t.box
                        for t in tracker.tracks.values()
                        if t.identity.name in presence
                    }

                    # Web self-enrollment: when a job targets this camera, collect
                    # samples whenever exactly one stable person can be tracked.
                    # Frames used here are still live-only, never persisted.
                    with enroll_lock:
                        job = enroll_jobs.get(camera_id)

                    if job is not None and job["status"] == "collecting":
                        if len(active) == 1 and active[0].matched >= 2:
                            box = active[0].box
                            hist = compute_torso_histogram(frame, box, CONFIG)
                            if hist is not None:
                                job["collected"].append(hist)
                                job["progress"] = len(job["collected"])
                                if job["progress"] >= job["frames_needed"]:
                                    with enroll_lock:
                                        job["status"] = "finalizing"
                                    finalize_enrollment(job)
                                else:
                                    job["message"] = (
                                        f"Collected {job['progress']}/{job['frames_needed']}"
                                    )
                        elif len(active) > 1:
                            job["message"] = ("More than one person detected — only "
                                              "the doctor should be in view.")
                        else:
                            job["message"] = ("No person detected — stand in view "
                                              "of the camera.")
                except Exception as exc:
                    # Detection/pipeline failure must NEVER freeze the live
                    # preview: keep the frame flowing so the feed stays live
                    # even if YOLO (or its compiled ops) is misbehaving.
                    det_errors += 1
                    if det_errors <= 3 or det_errors % 100 == 0:
                        print(f"ERROR camera '{camera_id}' detection: {exc}")
                        traceback.print_exc()
                    if det_errors >= 3:
                        current_badges = {}
                    time.sleep(0.02)
            else:
                time.sleep(0.01)

            consecutive_errors = 0
        except Exception as exc:
            consecutive_errors += 1
            print(f"ERROR camera '{camera_id}' loop: {exc}")
            traceback.print_exc()
            time.sleep(0.2)
            if consecutive_errors >= 30:
                print(f"FATAL camera '{camera_id}': too many errors, stopping thread.")
                break


def _draw_name_badge(frame, name, box):
    """Draw a stable green name badge for a tracked doctor on the live feed."""
    sx1, sy1, sx2, sy2 = map(int, box)
    (tw, th), _ = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    bx1 = max(0, sx1)
    by1 = max(16, sy1 - th - 16)
    bx2 = bx1 + tw + 12
    by2 = by1 + th + 10
    cv2.rectangle(frame, (bx1, by1), (bx2, by2), (34, 197, 94), -1)
    cv2.putText(frame, name, (bx1 + 6, by2 - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (5, 46, 15), 2)


def finalize_enrollment(job):
    """Average collected samples, write the profile JSON, and activate it live."""
    global PROFILES
    if not job["collected"]:
        with enroll_lock:
            job["status"] = "error"
            job["message"] = "No valid frames captured — please try again."
        return

    mean_hist = np.mean(np.array(job["collected"], dtype=np.float32), axis=0).tolist()
    filename = sanitize_filename(job["name"]) + ".json"
    path = os.path.join(PROFILES_DIR, filename)

    with open(path, "w", encoding="utf-8") as f:
        json.dump({"name": job["name"], "histogram": mean_hist}, f)

    with profile_lock:
        PROFILES = load_profiles_from_disk()

    with state_lock:
        if job["name"] not in states:
            states[job["name"]] = new_state()

    with live_lock:
        if job["name"] not in live_status:
            live_status[job["name"]] = {
                "in_zone": False,
                "clock_in": None,
                "cameras": [],
            }

    with enroll_lock:
        job["status"] = "complete"
        job["message"] = (f"Enrolled {job['name']}. "
                          f"Profile saved to profiles/{filename}")

    print(f"ENROLLED: {job['name']} -> profiles/{filename}")


def log_detections(snap, now):
    """Record which doctors are seen in-zone on which cameras.

    Logs a row per (doctor, camera) but throttles repeats so a continuous
    presence does not flood the table — it records while a doctor is being
    seen on a camera, including when the same doctor is on several cameras
    at once.
    """
    interval = CONFIG.get("detection_log_interval", DETECTION_LOG_INTERVAL)
    seen_at = datetime.now().isoformat(timespec="seconds")
    batch = []
    with detection_log_lock:
        for cid, info in snap.items():
            if now - info["ts"] > PRESENCE_STALE_SECONDS:
                continue
            label = camera_labels.get(cid, cid)
            for name in info["names"]:
                key = (name, cid)
                last = _last_detection_log.get(key)
                if last is None or now - last >= interval:
                    _last_detection_log[key] = now
                    batch.append((name, cid, label, seen_at))
    if not batch:
        return
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.executemany(
            "INSERT INTO detections (doctor_name, camera_id, camera_label, seen_at) "
            "VALUES (?, ?, ?, ?)",
            batch,
        )
        conn.commit()
        conn.close()
    except Exception as exc:
        print(f"ERROR writing detection log: {exc}")


def state_machine_loop(config):
    """Union presence across all cameras and drive per-doctor debounce + DB writes."""
    while True:
        time.sleep(0.2)
        now = time.time()

        with presence_lock:
            snap = {
                cid: {"names": set(info["names"]), "ts": info["ts"]}
                for cid, info in camera_presence.items()
            }

        log_detections(snap, now)

        # A doctor is "present" if ANY camera sees them in zone right now.
        union = {}
        for cid, info in snap.items():
            if now - info["ts"] > PRESENCE_STALE_SECONDS:
                continue
            for name in info["names"]:
                union.setdefault(name, []).append(cid)

        with state_lock:
            for name, state in states.items():
                cams_now = union.get(name, [])
                in_zone_now = bool(cams_now)

                if state["state"] == STATE_IN:
                    if not in_zone_now:
                        if state["exit_timer_start"] is None:
                            state["exit_timer_start"] = now
                        elif now - state["exit_timer_start"] >= config["exit_confirm_seconds"]:
                            _clock_out(name, state)
                            state["state"] = STATE_OUT
                            state["exit_timer_start"] = None
                    else:
                        state["exit_timer_start"] = None
                else:
                    if in_zone_now:
                        if state["enter_timer_start"] is None:
                            state["enter_timer_start"] = now
                        elif now - state["enter_timer_start"] >= config["enter_confirm_seconds"]:
                            _clock_in(name, state)
                            state["state"] = STATE_IN
                            state["enter_timer_start"] = None
                    else:
                        state["enter_timer_start"] = None

                # Keep the live "where are they now" view current.
                with live_lock:
                    entry = live_status.setdefault(name, {
                        "in_zone": False, "clock_in": None, "cameras": [],
                    })
                    entry["in_zone"] = state["state"] == STATE_IN
                    entry["cameras"] = [camera_labels.get(c, c) for c in cams_now]


def _clock_in(name, state):
    clock_in = datetime.now().isoformat(timespec="seconds")
    conn = sqlite3.connect(DB_PATH)
    cur = conn.execute(
        "INSERT INTO sessions (doctor_name, clock_in, clock_out, duration_seconds) "
        "VALUES (?, ?, ?, ?)",
        (name, clock_in, None, None),
    )
    conn.commit()
    state["session_id"] = cur.lastrowid
    state["clock_in"] = clock_in
    conn.close()
    print(f"CLOCK IN: {name} at {clock_in}")

    with live_lock:
        live_status[name]["in_zone"] = True
        live_status[name]["clock_in"] = clock_in


def _clock_out(name, state):
    clock_out = datetime.now().isoformat(timespec="seconds")
    clock_in_dt = datetime.fromisoformat(state["clock_in"]) if state["clock_in"] else None
    duration = None
    if clock_in_dt:
        duration = int((datetime.fromisoformat(clock_out) - clock_in_dt).total_seconds())

    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "UPDATE sessions SET clock_out = ?, duration_seconds = ? WHERE id = ?",
        (clock_out, duration, state["session_id"]),
    )
    conn.commit()
    conn.close()
    state["session_id"] = None
    state["clock_in"] = None
    print(f"CLOCK OUT: {name} at {clock_out} ({duration}s)")

    with live_lock:
        live_status[name]["in_zone"] = False
        live_status[name]["clock_in"] = None


@app.route("/")
def index():
    return render_template("landing.html")


@app.route("/feeds")
def feeds_page():
    return render_template("feeds.html")


@app.route("/enroll")
def enroll_page():
    return render_template("enroll.html")


def _new_link_code():
    while True:
        code = "".join(secrets.choice(LINK_CODE_ALPHABET) for _ in range(6))
        with link_lock:
            if code not in link_store:
                return code


def _activate_link(link):
    """Activate a link: register a live remote camera and start its thread.

    Idempotent: if the link is already active its existing camera id is
    returned. Called by the HOST's Approve action (never by the device).
    """
    if link.get("camera_id"):
        return link["camera_id"]

    with cam_lock:
        used = {c["id"] for c in cameras}
        idx = 1
        while f"remote{idx:02d}" in used:
            idx += 1
        camera_id = f"remote{idx:02d}"

    zone = CONFIG.get("chair_zone_polygon") or list(DEFAULT_ZONE)
    cam = {
        "id": camera_id,
        "label": link["label"],
        "source": None,
        "remote": True,
        "chair_zone_polygon": list(zone),
        "detection_confidence": CONFIG.get("detection_confidence", 0.5),
        "frame_skip": CONFIG.get("frame_skip", 2),
    }

    with cam_lock:
        cameras.append(cam)
        camera_labels[camera_id] = link["label"]

    src = RemoteFrameSource()
    stop = threading.Event()
    with remote_source_lock:
        remote_sources[camera_id] = src
    remote_stop_events[camera_id] = stop
    threading.Thread(target=camera_loop, args=(cam, stop), daemon=True).start()

    link["camera_id"] = camera_id
    link["status"] = "active"
    link["frames"] = 0
    link["last_frame"] = None

    print(f"REMOTE CAMERA ACTIVATED: {link['label']} ({camera_id})")
    return camera_id


@app.route("/api/link/create", methods=["POST"])
def api_link_create():
    """Generate a copyable link that a device opens to add its camera."""
    data = request.get_json(silent=True) or {}
    label = (data.get("label") or "Remote Camera").strip() or "Remote Camera"

    with cam_lock:
        remote_count = sum(1 for c in cameras if c.get("remote"))
        if remote_count >= MAX_REMOTE_CAMERAS:
            return jsonify({
                "ok": False,
                "message": f"Device camera limit ({MAX_REMOTE_CAMERAS}) reached.",
            }), 400

    code = _new_link_code()
    with link_lock:
        link_store[code] = {
            "label": label,
            "camera_id": None,
            "status": "pending",
            "frames": 0,
            "last_frame": None,
            "active_socket": False,
            "socket": None,
            "created": time.time(),
        }
        _persist_links()

    print(f"LINK CREATED: {label} code={code}")
    return jsonify({"ok": True, "code": code, "link_url": f"/link/{code}"})


@app.route("/api/link/<code>")
def api_link_status(code):
    with link_lock:
        link = link_store.get(code.upper())
    if link is None:
        return jsonify({"ok": False, "message": "Unknown link code."}), 404
    return jsonify({
        "ok": True,
        "label": link["label"],
        "status": link["status"],
        "camera_id": link["camera_id"],
        "frames": link.get("frames", 0),
        "last_frame": link.get("last_frame"),
    })


@app.route("/api/links")
def api_links():
    """List all device links (for the host /devices page)."""
    with link_lock:
        return jsonify([
            {
                "code": code,
                "label": lk["label"],
                "status": lk["status"],
                "camera_id": lk["camera_id"],
                "frames": lk.get("frames", 0),
                "created": lk.get("created"),
            }
            for code, lk in sorted(link_store.items(),
                                   key=lambda kv: kv[1].get("created", 0))
        ])


@app.route("/api/link/<code>/confirm", methods=["POST"])
def api_link_confirm(code):
    """Host approves a device link: register the camera for tracking."""
    code = code.upper()
    with link_lock:
        link = link_store.get(code)
    if link is None:
        return jsonify({"ok": False, "message": "Unknown link code."}), 404

    with link_lock:
        camera_id = link.get("camera_id") if link else None
    if camera_id is None:
        with link_lock:
            link = link_store.get(code)
            if link is None:
                return jsonify({"ok": False, "message": "Unknown link code."}), 404
            camera_id = _activate_link(link)
            _persist_links()

    with link_lock:
        link = link_store.get(code)
    label = link["label"] if link else code
    socket = link.get("socket") if link else None
    if socket is not None:
        try:
            socket.send(json.dumps({"type": "active", "camera_id": camera_id}))
        except Exception:
            pass
    print(f"HOST APPROVED: <{code}> -> {label} ({camera_id})")
    return jsonify({"ok": True, "camera_id": camera_id, "status": "active"})


@app.route("/api/link/<code>/delete", methods=["POST"])
def api_link_delete(code):
    """Discard a link that is not yet active."""
    code = code.upper()
    with link_lock:
        link = link_store.get(code)
        if link is None:
            return jsonify({"ok": False, "message": "Unknown link code."}), 404
        if link.get("socket") is not None:
            return jsonify({
                "ok": False,
                "message": "Device still connected — remove it from the cameras table first."
            }), 400
        del link_store[code]
        _persist_links()
    print(f"LINK DISCARDED: <{code}>")
    return jsonify({"ok": True})


@app.route("/link/<code>")
def link_page(code):
    """Device page: grant camera access and stream back to the tracker."""
    with link_lock:
        link = link_store.get(code.upper())
    if link is None:
        return "This link is not valid or has already been used.", 404
    return render_template("link.html", link_code=code.upper())


@app.route("/devices")
def devices_page():
    return render_template("devices.html")


@app.route("/logs")
def logs_page():
    return render_template("logs.html")


@app.route("/api/cameras")
def api_cameras():
    now = time.time()
    with cam_lock, preview_lock:
        return jsonify([
            {
                "id": c["id"],
                "label": c["label"],
                "remote": bool(c.get("remote")),
                "live": preview_times.get(c["id"], 0) and
                        (now - preview_times.get(c["id"], 0)) < LIVE_AFTER_SECONDS,
            }
            for c in cameras
        ])


@app.route("/api/unregister_remote_camera", methods=["POST"])
def api_unregister_remote_camera():
    data = request.get_json(silent=True) or {}
    camera_id = data.get("camera_id") or ""

    with cam_lock:
        match = [c for c in cameras if c["id"] == camera_id]
    if not match or not match[0].get("remote"):
        return jsonify({"ok": False, "message": "Not a remote camera."}), 400

    with cam_lock:
        cameras[:] = [c for c in cameras if c["id"] != camera_id]
        camera_labels.pop(camera_id, None)

    with link_lock:
        for code, lk in list(link_store.items()):
            if lk["camera_id"] == camera_id:
                del link_store[code]
        _persist_links()

    stop = remote_stop_events.pop(camera_id, None)
    if stop is not None:
        stop.set()
    with remote_source_lock:
        remote_sources.pop(camera_id, None)

    print(f"REMOTE CAMERA REMOVED: {camera_id}")
    return jsonify({"ok": True})


@app.route("/ws/cam/<code>", websocket=True)
def ws_cam(code):
    """WebSocket endpoint for a linked device's camera frames.

    The device streams frames immediately when its page opens — there is no
    confirm step on the device side. Activation for TRACKING is gated by the
    HOST: the admin's "Approve & Start Tracking" button on /devices calls
    POST /api/link/<code>/confirm, which registers the remote camera. Until
    then frames are received and counted but not consumed for tracking.
    """
    code = code.upper()
    with link_lock:
        link = link_store.get(code)
    if link is None:
        abort(404)

    try:
        ws = Server.accept(request.environ)
    except Exception as exc:
        print(f"WS HANDSHAKE FAILED <{code}>: {exc}")
        traceback.print_exc()
        abort(400)

    try:
        with link_lock:
            link = link_store.get(code)
            if link is None:
                return ""
            if link.get("socket") is not None:
                ws.send(json.dumps({"type": "error",
                                    "message": "already streaming from another tab"}))
                ws.close()
                return ""
            link["socket"] = ws
            if not link.get("camera_id"):
                link["status"] = "awaiting_confirm"
                _persist_links()
        print(f"WS CONNECTED: <{code}> streaming, waiting for host approval...")
        _log_stream("ws_open", code=code)

        while True:
            try:
                data = ws.receive(timeout=1.0)
            except TimeoutError:
                if ws.closed:
                    break
                continue
            except ConnectionClosed:
                break
            if data is None:
                break

            if isinstance(data, str):
                try:
                    msg = json.loads(data)
                except Exception:
                    continue
                if msg.get("type") == "telemetry":
                    _log_stream("send_side", code=code, data=msg)
                continue
            if not isinstance(data, (bytes, bytearray)):
                continue

            st = _stream_codes.setdefault(code, {"recv": 0, "t0": time.time()})
            st["recv"] += 1
            _log_stream("recv", code=code, n=st["recv"], size=len(data))
            if st["recv"] % 300 == 0:
                fps = st["recv"] / max(1e-6, time.time() - st["t0"])
                print(f"STREAM RECV <{code}>: {st['recv']} frames, "
                      f"{fps:.1f} fps since start")
                _log_stream("recv_summary", code=code, total=st["recv"],
                            elapsed_s=round(time.time() - st["t0"], 3),
                            fps=round(fps, 2))

            arr = np.frombuffer(data, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is None:
                continue

            # A phone's front camera records a mirror of reality: raise your
            # right hand and it appears on the LEFT of the raw image. Flip
            # once at the source so the preview (feeds / enroll) and tracking
            # frames show the world as the person facing the camera sees it
            # (right hand on the right). Badges drawn later stay readable and
            # enrollment histograms match detection frames.
            img = cv2.flip(img, 1)

            cam_id = None
            with link_lock:
                link = link_store.get(code)
                if link is not None:
                    cam_id = link.get("camera_id")
            src = None
            if cam_id:
                with remote_source_lock:
                    src = remote_sources.get(cam_id)
            if src is not None:
                src.push(img)
                # Keep the live preview fresh at the FULL inbound frame rate.
                # The camera thread only gets to publish when a detection cycle
                # finishes (YOLO ~200-400ms every few frames on this box), so if
                # publishing were left to it alone, viewers would see a low-fps
                # slideshow that looks like a bad call. Publishing here, on the
                # socket thread, decouples preview freshness from detection
                # latency. The camera thread still publishes the badged version
                # whenever one is ready, which simply overwrites this latest-wins
                # slot a moment later.
                publish_preview(cam_id, img)

            with link_lock:
                link = link_store.get(code)
                if link is not None:
                    link["frames"] = link.get("frames", 0) + 1
                    if link["frames"] == 1:
                        print(f"LINK FIRST FRAME RECEIVED: <{code}>")
                    link["last_frame"] = time.time()
    except Exception as exc:
        print(f"WS CAM ERROR <{code}>: {exc}")
        traceback.print_exc()
    finally:
        try:
            ws.close()
        except Exception:
            pass
        with link_lock:
            link = link_store.get(code)
            if link is not None:
                if link.get("socket") is ws:
                    link["socket"] = None
                if not link.get("camera_id") and link["status"] != "pending":
                    link["status"] = "pending"
                    _persist_links()
    _log_stream("ws_close", code=code)
    return ""


@app.route("/api/live_status")
def api_live_status():
    with live_lock:
        snapshot = {
            name: {
                "in_zone": info["in_zone"],
                "clock_in": info["clock_in"],
                "cameras": list(info["cameras"]),
            }
            for name, info in live_status.items()
        }
    return jsonify(snapshot)


@app.route("/api/logs")
def api_logs():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT id, doctor_name, clock_in, clock_out, duration_seconds "
        "FROM sessions ORDER BY id DESC LIMIT 50"
    ).fetchall()
    conn.close()
    return jsonify([dict(r) for r in rows])


@app.route("/api/detections")
def api_detections():
    limit = min(int(request.args.get("limit", 500)), 2000)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT id, doctor_name, camera_id, camera_label, seen_at "
        "FROM detections ORDER BY id DESC LIMIT ?",
        (limit,),
    ).fetchall()
    conn.close()
    return jsonify([dict(r) for r in rows])


def preview_generator(camera_id):
    """Yield MJPEG frames of one camera's latest rendered preview frame.

    This frame is never persisted — encoded to JPEG in memory and streamed
    live to the browser. Output is paced to PREVIEW_FPS (the consumer's rate):
    a newer frame is only served when one is actually available and the pacing
    interval has passed, and intermediate frames are dropped rather than
    buffered — so playback stays smooth/live even when the phone captures
    faster than the browser can play (e.g. 35 vs 30 fps). When the camera has
    no feed at all, a "No feed" placeholder is served once, not in a hot loop.
    """
    last_version = -1
    last_sent = 0.0
    showed_placeholder = False
    placeholder = None

    while True:
        now = time.time()
        wait = (last_sent + _MIN_PREVIEW_INTERVAL) - now
        if wait > 0:
            time.sleep(min(wait, 0.1))

        with preview_lock:
            version = preview_versions.get(camera_id, 0)
            live_frame = preview_frames.get(camera_id)
            if live_frame is not None:
                live_frame = live_frame.copy()

        if live_frame is not None and version != last_version:
            frame = live_frame
            last_version = version
            showed_placeholder = False
            placeholder = None
        elif live_frame is None and not showed_placeholder:
            if placeholder is None:
                placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(placeholder, "No feed", (20, 240),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            frame = placeholder
            showed_placeholder = True
        else:
            time.sleep(0.02)
            continue

        ok, jpeg = cv2.imencode(
            ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, PREVIEW_JPEG_QUALITY])
        if not ok:
            time.sleep(0.02)
            continue

        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" +
               jpeg.tobytes() + b"\r\n")
        last_sent = time.time()


@app.route("/api/video_preview/<camera_id>")
def api_video_preview(camera_id):
    return Response(
        preview_generator(camera_id),
        mimetype="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-store"},
    )


@app.route("/api/enroll", methods=["POST"])
def api_enroll():
    data = request.get_json(silent=True) or {}
    name = (data.get("name") or "").strip()
    camera_id = data.get("camera_id") or ""

    if not name:
        return jsonify({"ok": False, "message": "A doctor name is required."}), 400
    if camera_id not in camera_labels:
        return jsonify({"ok": False, "message": "Invalid camera selected."}), 400

    with enroll_lock:
        existing = enroll_jobs.get(camera_id)
        if existing and existing["status"] in ("collecting", "finalizing"):
            return jsonify({
                "ok": False,
                "message": f"An enrollment for this camera is already in progress "
                           f"({existing['name']}).",
            }), 409

        enroll_jobs[camera_id] = {
            "name": name,
            "camera_id": camera_id,
            "frames_needed": ENROLL_FRAMES_DEFAULT,
            "collected": [],
            "progress": 0,
            "message": "",
            "status": "collecting",
        }

    print(f"ENROLLMENT STARTED: {name} on camera '{camera_id}'")
    return jsonify({"ok": True, "message": "Enrollment started. Stand in view now."})


@app.route("/api/enroll_status")
def api_enroll_status():
    with enroll_lock:
        jobs = []
        for job in enroll_jobs.values():
            jobs.append({
                "name": job["name"],
                "camera_id": job["camera_id"],
                "progress": job["progress"],
                "frames_needed": job["frames_needed"],
                "message": job["message"],
                "status": job["status"],
            })
    return jsonify(jobs)


def main():
    init_db()
    with link_lock:
        _load_links()

    global CONFIG, MODEL, cameras, camera_labels
    config = load_config()
    CONFIG = config
    # Apply cloud environment overrides
    if os.environ.get("CLOUD_FRAME_SKIP"):
        config["frame_skip"] = int(os.environ["CLOUD_FRAME_SKIP"])
    if os.environ.get("CLOUD_DETECTION_CONFIDENCE"):
        config["detection_confidence"] = float(os.environ["CLOUD_DETECTION_CONFIDENCE"])
    # Cameras are attached at startup from config.json (wired) or later via the
    # device-link / approval flow (remote). See _boot_wired_cameras below.
    global cameras, camera_labels
    cameras = []
    camera_labels = {}

    global PROFILES
    with profile_lock:
        PROFILES = load_profiles_from_disk()
    for name in PROFILES:
        with state_lock:
            states[name] = new_state()
        with live_lock:
            live_status[name] = {"in_zone": False, "clock_in": None, "cameras": []}
    if not PROFILES:
        print("WARNING: no doctor profiles enrolled yet. "
              "Use the web UI (/enroll) or enroll.py to add doctors.")

    MODEL = YOLO("yolov8n.pt")

    # Boot the wired (webcam / RTSP) cameras declared in config.json so they
    # appear in the feeds/enroll lists and run the tracking pipeline from the
    # start. Remote ("link") cameras are NOT started here — they attach later
    # through the device-link / approval flow (_activate_link).
    for ent in config.get("cameras", []):
        cid = ent.get("id")
        src_cfg = ent.get("source")
        if isinstance(src_cfg, dict):
            continue
        cam = {
            "id": cid,
            "label": ent.get("label", cid),
            "source": src_cfg,
            "remote": False,
            "chair_zone_polygon": (ent.get("chair_zone_polygon")
                                   or config.get("chair_zone_polygon")
                                   or list(DEFAULT_ZONE)),
            "detection_confidence": ent.get(
                "detection_confidence",
                config.get("detection_confidence", 0.5)),
            "frame_skip": ent.get("frame_skip", config.get("frame_skip", 2)),
        }
        cap = cv2.VideoCapture(src_cfg)
        if not cap.isOpened():
            print(f"WARNING: could not open configured camera '{cid}' "
                  f"(source {src_cfg!r}) - skipped.")
            continue
        stop = threading.Event()
        with cam_lock:
            cameras.append(cam)
            camera_labels[cid] = cam["label"]
        with remote_source_lock:
            remote_sources[cid] = cap
            remote_stop_events[cid] = stop
        threading.Thread(target=camera_loop, args=(cam, stop), daemon=True).start()
        print(f"WIRED CAMERA STARTED: {cam['label']} ({cid}) source={src_cfg!r}")

    threading.Thread(target=state_machine_loop, args=(config,), daemon=True).start()

    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), threaded=True)


if __name__ == "__main__":
    main()
