# Whisper Large-v3-Turbo on RunPod Serverless

## 1. Build & push (must be linux/amd64)
```bash
docker build --platform linux/amd64 -t YOUR_DOCKERHUB_USER/whisper-turbo-runpod:v1 .
docker push YOUR_DOCKERHUB_USER/whisper-turbo-runpod:v1
```

## 2. Create the endpoint
RunPod Console -> Serverless -> New Endpoint -> Custom Docker image:
- Image: `YOUR_DOCKERHUB_USER/whisper-turbo-runpod:v1`
- GPU: any 16 GB+ (A4000 / L4 / A5000 are plenty; turbo needs ~3 GB)
- Active workers: 0 (scale to zero) or 1 (no cold starts)
- Max workers: as needed
- Container disk: 15 GB
- Enable FlashBoot

## 3. Test
```bash
pip install requests
export RUNPOD_API_KEY=...
export RUNPOD_ENDPOINT_ID=...
python test_client.py https://example.com/audio.mp3
```

## Request format
```json
{"input": {"audio_url": "https://...", "language": "en",
           "word_timestamps": false, "vad_filter": true}}
```
Use `audio_url` for anything large; /run caps payloads at ~10 MB, /runsync ~20 MB.
