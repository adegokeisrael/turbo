"""
Usage:
  export RUNPOD_API_KEY=...
  export RUNPOD_ENDPOINT_ID=...
  python test_client.py https://example.com/audio.mp3
  python test_client.py ./local_audio.wav
"""
import base64
import os
import sys
import time

import requests

API_KEY = os.environ["RUNPOD_API_KEY"]
ENDPOINT = os.environ["RUNPOD_ENDPOINT_ID"]
BASE = f"https://api.runpod.ai/v2/{ENDPOINT}"
HEADERS = {"Authorization": f"Bearer {API_KEY}"}


def build_input(src: str) -> dict:
    if src.startswith("http"):
        return {"audio_url": src}
    with open(src, "rb") as f:
        return {"audio_base64": base64.b64encode(f.read()).decode()}


def transcribe(src: str, poll_every: float = 2.0) -> dict:
    payload = {"input": {**build_input(src), "language": "en"}}
    job = requests.post(f"{BASE}/run", json=payload, headers=HEADERS, timeout=60).json()
    job_id = job["id"]
    while True:
        r = requests.get(f"{BASE}/status/{job_id}", headers=HEADERS, timeout=60).json()
        if r["status"] == "COMPLETED":
            return r["output"]
        if r["status"] in ("FAILED", "CANCELLED", "TIMED_OUT"):
            raise RuntimeError(r)
        time.sleep(poll_every)


if __name__ == "__main__":
    result = transcribe(sys.argv[1])
    print(result.get("text") or result)
    print(f"\n[{result.get('audio_duration')}s audio in {result.get('processing_time')}s]")
