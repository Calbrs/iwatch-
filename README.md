---
title: Clinic Doctor Time Tracker
emoji: 🕒
colorFrom: green
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# Clinic Doctor Time Tracker

A system that tracks how much time a doctor spends at the treatment chair
using cameras on any number of linked mobile devices/phones. The doctor is
identified by **physical appearance** (clothing/uniform color histogram)
rather than facial recognition — the camera is positioned too far away for
a face to be reliably readable.

A dashboard (landing page) shows who is currently at the chair (live
status) plus a historical report of time per doctor (clock_in, clock_out,
duration). Live camera feeds live on their own page (`/feeds`). No video or
images of any person are stored persistently — the only data retained is
name + timestamps.

## How It Works

YOLOv8n detects people in each live frame (person class only, no facial
recognition). Detections are first linked into **persistent tracklets**
(SORT-style IoU association in `tracking.py`) so boxes stay stable between
detections. For each tracked person inside the configured chair-zone ROI
polygon, an HSV color histogram of the chest/torso region is computed and
compared against enrolled doctor profiles using Bhattacharyya distance.
Identity is decided by **voting over a sliding window** of recent matches
with hysteresis — a single noisy frame can't flip the name, and a new
identity only takes over after it persistently outvotes the current one.
A debounce timer avoids false triggers — the doctor must remain in the zone
for a configured number of seconds before clock_in/clock_out is confirmed.
Sessions are logged to SQLite and exposed through a REST API to the
auto-refreshing dashboard.

Only **enrolled doctors** are ever tracked; anyone not matching a
registered profile (e.g. a patient) is completely ignored by the
zone/logging logic.

## Installation

```bash
pip install -r requirements.txt
```

## Enrolling a Doctor

Point the camera so only the doctor is in view, then run:

```bash
python enroll.py --name "Dr. Smith"
```

Use a custom camera source or more frames if needed:

```bash
python enroll.py --name "Dr. Smith" --source 0 --frames 25
```

This creates `profiles/dr_smith.json` containing a numeric histogram array
— not an image. Repeat for each doctor you want to track.

## Starting the System

```bash
python tracker.py
```

Or use the one-shot launcher, which prints the tracker log **live in the same
console** and guarantees that pressing **Ctrl+C** stops the tracker (no orphaned
processes; any leftover server on port 5000 is cleaned up first):

```powershell
powershell -ExecutionPolicy Bypass -File start.ps1
```

Then open `http://localhost:5000` in a browser. The dashboard auto-refreshes
every 5 seconds. For remote/device access use a cloud deploy (Render or HF
Space), which provides the public HTTPS endpoint.

## Linking Device Cameras (up to 50)

Cameras are attached **only** through the device-link approval flow — the
server never opens any local/webcam/RTSP camera directly.

1. Open the dashboard → "Link a Device Camera" (the `/devices` page).
2. Enter a camera name and click **Generate Link** — a copyable link appears.
3. Open that link on the device that has the camera you want to use.
   (Use the public HTTPS URL for a phone; `getUserMedia` needs a secure
   context.)
4. On the device page click **Allow Camera**. The device starts streaming
   automatically — there is no confirm step on the device.
5. Back on the host, on the `/devices` page the link shows *Streaming — awaiting
   approval*. Click **Approve** to register the camera and start using it for
   tracking (approval is always done by the host, not the device). The `/feeds`
   page then shows its live feed and the `/devices` page lists it as connected
   (with a frame counter for real-time confirmation that video is actually
   arriving). Use **Discard** to cancel a link that is not needed, or
   **Remove** to unlink an active camera (device loses streaming access).

Multiple devices can stream simultaneously; each approved device becomes its
own camera (remote01, remote02, …) and the tracking/logging pipeline runs
per camera.

Links are kept in `links.json`, so they survive a tracker restart: after
re-running `start.ps1`, previously created links still appear on `/devices`
and can be approved again (device links re-stream automatically). Discarding
a link or removing a camera deletes it permanently from `links.json`.

Camera feeds only show live/connected cameras. When an enrolled doctor is
detected in a camera's chair zone, a green badge with their name is drawn on
that live feed while they are being tracked.

## Adjusting the Chair Zone Polygon

All thresholds and settings live in `config.json` — no code changes needed.

The `chair_zone_polygon` is a list of `[x, y]` corners defining the region
of interest in the camera frame (e.g. the area around the treatment chair).

To get pixel coordinates from a camera frame:

1. Position the camera exactly as it will be during live use.
2. Capture a single frame to inspect — for OpenCV you can use a short
   throwaway snippet:
   ```python
   import cv2
   cap = cv2.VideoCapture(0)          # or your RTSP URL
   ret, frame = cap.read()
   cv2.imwrite("_debug_frame.jpg", frame)   # delete this file afterwards
   cap.release()
   ```
3. Open the frame in any image editor and note the pixel coordinates of
   the chairs (or area) you want to track.
4. Enter those corners into `chair_zone_polygon` in `config.json`.

Other useful settings:

| Setting | Meaning |
|---|---|
| `chair_zone_polygon` | Zone applied to approved device cameras (defaults to the full frame) |
| `enter_confirm_seconds` | Debounce duration before confirming clock in |
| `exit_confirm_seconds` | Debounce duration before confirming clock out |
| `appearance_match_threshold` | Max Bhattacharyya distance accepted as a match |
| `torso_crop_ratio` | Portion of the bbox used as the torso region |
| `detection_confidence` | YOLO confidence threshold |
| `frame_skip` | Process every Nth frame |
| `tracking_window` | How many recent matches vote on a doctor's identity |
| `tracking_commit_frac` | Fraction of the window needed to commit a name |
| `tracking_release_frac` | Fraction below which a committed name is dropped |
| `tracking_start_gate` | Minimum frames before a vote can commit |
| `tracking_max_lost` | Detection rounds a person may vanish before the tracklet ends |
| `detection_log_interval` | Seconds between detection-log rows per doctor+camera (default `10.0`) |

## Detection Log

A dedicated **Detection Log** page (`/logs`, linked from the dashboard and
feeds pages) records each time a tracked doctor is seen in a camera's chair
zone. Each event is a
row with timestamp, doctor name, and the camera that saw them. If the same
doctor is on more than one camera at the same time, each camera gets its own
row. The log is throttled by `detection_log_interval` so continuous presence
does not flood the table.

## Streaming

Device cameras stream live frames to the tracker over WebSocket. The device
sends at up to ~30 fps and **adaptively** lowers the frame resolution and JPEG
quality only if the connection falls behind — so on a fast/wired connection the
feed stays as smooth and clean as possible. The server serves only freshly
published frames to the feeds page (no repeated re-encoding of stale frames),
which keeps the preview smooth and low-latency.

## Privacy

This system does **not** store any video or images. Frames are processed
live and discarded immediately after the histogram is extracted. The only
data persisted is `doctor_name`, detection events (doctor + camera + time),
and session `clock_in`/`clock_out`/`duration_seconds`. Appearance profiles
are numeric arrays only and cannot be reconstructed into an image of a person.

**PDPA (Tanzania) compliance** before live deployment at a clinic:
1. Register with the Personal Data Protection Commission (PDPC).
2. Provide a clear written notice to all staff that the workspace camera
   is monitoring chair-occupancy time for time-tracking purposes.
3. Consider a brief Data Protection Impact Assessment (DPIA).
4. Keep the notice and retention policy documented and accessible to staff.

---

## Deploying to the Cloud (Render or Hugging Face)

The same repo works on **both** platforms — `tracker.py` reads the `PORT`
env var, and the `Dockerfile` is the single build definition.

### Option A — Render (free tier, recommended)
1. Push this repo to GitHub (already done: `https://github.com/Calbrs/iwatch-`).
2. Render Dashboard → **New** → **Web Service** → choose **Docker** as the environment.
3. Connect the GitHub repo, set **Dockerfile path** to `Dockerfile`, plan **Free**.
4. Render auto‑sets `PORT`; our app binds it via `int(os.environ.get("PORT", 5000))`.
5. Add optional env vars (`CLOUD_FRAME_SKIP` etc.) under **Advanced**.
6. Deploy. The public URL (`https://<service>.onrender.com`) is the URL you
   open, generate device links, and hand to phones.

Constraints: 512 MB RAM / 0.1 CPU. YOLOv8n runs on CPU; the device-to-cloud
WebSocket path is direct (Render's own HTTPS), which feels smooth. The free
instance spins down after 15 min idle (also spins up again on the next request
— ~1 min cold start).

### Option B — Hugging Face Spaces (Docker, requires PRO to create)
HF reads the `sdk: docker` YAML at the top of this `README.md` and runs the
`Dockerfile`. It sets `PORT=7860` for you (`app_port: 7860` in the YAML).

1. Create a Space at `https://huggingface.co/new-space` — name it e.g.
   `clinic-time-tracker`, hardware **CPU basic**, SDK **Docker**.
2. Push the repo contents to the Space repo:
   ```bash
   git clone https://huggingface.co/spaces/<user>/clinic-time-tracker
   # copy this repo's files into that folder, then:
   cd clinic-time-tracker
   git add . && git commit -m "initial" && git push
   ```
3. HF builds the Docker image and starts the app. Your URL becomes
   `https://<user>-clinic-time-tracker.hf.space`.

Note: **free HF accounts cannot create Docker Spaces** — that costs a PRO
plan. The only free HF compute is Gradio “ZeroGPU” ($0/hr but limited to 2
spaces and requires a Gradio app, not Flask). If you don’t have PRO, use
Render Option A.

### Option C — Oracle Cloud Always Free (best free performance)
A real `VM.Standard.A1.Flex` ARM VPS: up to 4 OCPU / 24 GB RAM / 200 GB disk,
free forever (no idle shutdown). 2 OCPU / 14 GB runs YOLOv8n at several x the
speed of Render's 0.1 CPU.

1. Provision the instance (Compute → Create Instance → shape `A1.Flex`,
   Ubuntu 24.04 or Oracle Linux, download the SSH key). If "Out of capacity",
   retry in a different availability domain or take 1–2 OCPU first.
2. In the OCI Console open **TCP 5000** in the VCN Security List
   (VCN → Public Subnet → Security List → Add Ingress Rule, source
   `0.0.0.0/0`, port 5000). OCI blocks everything by default.
3. SSH from Windows: `ssh -i <key> ubuntu@<public-ip>` (Ubuntu) or `opc@`
   (Oracle Linux). If your key's permissions are rejected on Windows:
   `icacls <key> /inheritance:r /grant:r "$($env:USERNAME):R"`.
4. Run the one-shot deployer in the repo:
   ```bash
   bash deploy_oracle.sh
   ```
   It installs deps in a venv, opens the firewall, and registers a systemd
   service (`iwatch-tracker`) that auto-starts on boot and restarts on crash.
5. Verify: `http://<public-ip>:5000`.

**HTTPS for phones (required — browsers block camera access on plain HTTP):**
- Easiest, no domain, free: a Cloudflare quick tunnel.
  ```bash
  curl -L https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-arm64 -o cloudflared \
    && chmod +x cloudflared && sudo mv cloudflared /usr/local/bin/
  cloudflared tunnel --url http://localhost:5000
  ```
  It prints `https://<random>.trycloudflare.com` — use that URL for the
  dashboard and device links. Note: the URL changes on each restart.
- Stable HTTPS with a domain: install Caddy (auto Let's-Encrypt) or a named
  Cloudflare tunnel and point it at `http://localhost:5000`.

### Local‑network tip (no cloud)
If you prefer the absolute smoothest fps and have a local network, you can
run the tracker directly on a spare laptop/PC, expose it via Cloudflare Tunnel
(or any tunnel/RDP), and have devices connect to the public URL. This avoids
the 512 MB RAM ceiling and gives you full GPU access if available.

### Environment‑variable overrides (optional, for cloud tuning)
- `CLOUD_FRAME_SKIP` – e.g. `3` processes every 3rd frame, raising fps at the
  cost of detection frequency.
- `CLOUD_DETECTION_CONFIDENCE` – e.g. `0.4` lowers the YOLO threshold, more
  detections but more CPU load.
- `CLOUD_PREVIEW_QUALITY` – e.g. `0.5` reduces MJPEG encode cost.

---
