FROM python:3.11-slim

WORKDIR /app

# system deps สำหรับ OpenCV (libgl1 ต้องการเพราะ ultralytics ดึง opencv-python ที่ non-headless)
RUN apt-get update && apt-get install -y \
    libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1 libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# ติดตั้ง CPU-only torch ก่อน (ลด image size จาก ~3.5GB → ~1.3GB)
RUN pip install --no-cache-dir torch torchvision \
    --index-url https://download.pytorch.org/whl/cpu

# ติดตั้ง dependencies ที่เหลือ (torch/torchvision ที่ติดตั้งแล้วจะถูก skip)
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Cloud Run ใช้ port 8080
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
