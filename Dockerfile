FROM python:3.12-slim

# Install minimum deps + CPU-only torch (avoids CUDA RAM overhead)
RUN pip install --no-cache-dir ultralytics>=8.2.0 opencv-python>=4.9.0 flask>=3.0.0 numpy>=1.26.0 simple-websocket>=1.0.0

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