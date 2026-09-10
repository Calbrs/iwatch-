import cv2, numpy as np
from ultralytics import YOLO

path = "/tmp/debug_frames/remote01_frame1001.jpg"
try:
    img = cv2.imread(path)
    print("loaded shape:", None if img is None else img.shape)
    if img is None:
        raise SystemExit(1)
    b, g, r = img[..., 0].mean(), img[..., 1].mean(), img[..., 2].mean()
    print(f"channel means B={b:.1f} G={g:.1f} R={r:.1f}")
    print("min/max:", int(img.min()), int(img.max()))
    h, w = img.shape[:2]
    center = img[h//2-3:h//2+3, w//2-3:w//2+3]
    print("center patch mean BGR:", tuple(round(float(x),1) for x in center.reshape(-1,3).mean(axis=0)))

    model = YOLO("yolov8n.pt")
    results = model.predict(img, conf=0.25, verbose=False)
    boxes = results[0].boxes
    print("total detections:", len(boxes))
    persons = [b for b in boxes if int(b.cls) == 0]
    print("persons detected:", len(persons))
    for b in boxes:
        print("  cls", int(b.cls), "conf", round(float(b.conf), 3),
              "xyxy", [round(float(v),1) for v in b.xyxy[0].tolist()])
except Exception as e:
    import traceback; traceback.print_exc()