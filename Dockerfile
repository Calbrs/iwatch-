FROM python:3.12-slim

# Minimal runtime libraries OpenCV needs on Debian slim
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# CPU-only torch (avoids CUDA RAM overhead).
# Install opencv-python-headless LAST so the headless cv2 build wins over the
# opencv-python wheel that ultralytics pulls in (the GUI build needs X11/GTK
# libs like libxcb.so.1 which slim images do not have).
RUN pip install --no-cache-dir ultralytics>=8.2.0 flask>=3.0.0 numpy>=1.26.0 simple-websocket>=1.0.0 \
 && pip install --no-cache-dir opencv-python-headless>=4.9.0

# Copy the app
WORKDIR /app
COPY . .

# Pre-download model so first start has no network delay
RUN python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"

# Expose the port Render/HF Spaces will use
ENV PORT=5000
EXPOSE 5000

# Run the tracker
CMD ["python", "tracker.py"]