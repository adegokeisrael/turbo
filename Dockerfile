FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/models \
    WHISPER_MODEL=large-v3-turbo \
    WHISPER_CACHE=/models

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip ffmpeg ca-certificates \
    && ln -s /usr/bin/python3 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Bake the model into the image so cold starts don't re-download it
RUN python -c "from faster_whisper import download_model; download_model('large-v3-turbo', cache_dir='/models')"

COPY handler.py .

CMD ["python", "-u", "handler.py"]
