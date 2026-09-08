"""
enroll.py — Register a doctor's appearance profile before live use.

Usage:  python enroll.py --name "Dr. Smith"
"""
import argparse
import json
import os

import cv2
import numpy as np
from ultralytics import YOLO

from appearance import compute_torso_histogram, sanitize_filename

PROFILES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "profiles")


def load_config():
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="Enroll a doctor's appearance profile.")
    parser.add_argument("--name", required=True, help="Doctor's name")
    parser.add_argument("--source", default=0, help="Camera source (index or RTSP URL)")
    parser.add_argument("--frames", type=int, default=25, help="Number of frames to collect")
    args = parser.parse_args()

    config = load_config()
    if not config.get("torso_crop_ratio"):
        print("WARNING: config.json missing torso_crop_ratio; using defaults.")

    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open camera source: {args.source}")

    model = YOLO("yolov8n.pt")

    histograms = []
    max_attempts = args.frames * 10
    attempts = 0

    print(f"Enrolling '{args.name}' — collecting {args.frames} frames. Keep only the doctor in view...")

    while len(histograms) < args.frames and attempts < max_attempts:
        attempts += 1
        ret, frame = cap.read()
        if not ret:
            continue

        results = model(frame, conf=config["detection_confidence"], verbose=False)
        boxes = [r for r in results[0].boxes if int(r.cls) == 0]

        if len(boxes) == 1:
            box = boxes[0].xyxy[0].cpu().numpy()
            hist = compute_torso_histogram(frame, box, config)
            if hist is not None:
                histograms.append(hist)
                print(f"  collected {len(histograms)}/{args.frames}")
        elif len(boxes) > 1:
            print("WARNING: more than one person detected — make sure only the doctor is in the enrollment area.")

    cap.release()

    if not histograms:
        raise RuntimeError("No person detected during enrollment. No profile written.")

    mean_hist = np.mean(np.array(histograms, dtype=np.float32), axis=0).tolist()

    os.makedirs(PROFILES_DIR, exist_ok=True)
    filename = sanitize_filename(args.name) + ".json"
    profile_path = os.path.join(PROFILES_DIR, filename)

    profile = {"name": args.name, "histogram": mean_hist}
    with open(profile_path, "w", encoding="utf-8") as f:
        json.dump(profile, f)

    print(f"Enrollment complete. Profile saved to {profile_path}")


if __name__ == "__main__":
    main()
