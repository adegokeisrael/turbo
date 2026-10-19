import base64
import os
import tempfile
import time

import requests
import runpod
from faster_whisper import WhisperModel

MODEL_NAME = os.environ.get("WHISPER_MODEL", "large-v3-turbo")
CACHE_DIR = os.environ.get("WHISPER_CACHE", "/models")
MAX_DOWNLOAD_MB = int(os.environ.get("MAX_DOWNLOAD_MB", "500"))

# Loaded once per worker, reused across requests
print(f"Loading {MODEL_NAME} ...")
_t0 = time.time()
model = WhisperModel(
    MODEL_NAME,
    device="cuda",
    compute_type="float16",
    download_root=CACHE_DIR,
)
print(f"Model loaded in {time.time() - _t0:.1f}s")


def _download(url: str, dest: str) -> None:
    limit = MAX_DOWNLOAD_MB * 1024 * 1024
    size = 0
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                size += len(chunk)
                if size > limit:
                    raise ValueError(f"Audio exceeds {MAX_DOWNLOAD_MB} MB limit")
                f.write(chunk)


def handler(job):
    """
    Input:
      audio_url          : URL to an audio/video file   (or)
      audio_base64       : base64-encoded audio bytes
      language           : default "en"
      beam_size          : default 5
      vad_filter         : default true
      word_timestamps    : default false
      initial_prompt     : optional
      return_segments    : default true
    """
    inp = job.get("input", {})
    url = inp.get("audio_url")
    b64 = inp.get("audio_base64")

    if not url and not b64:
        return {"error": "Provide 'audio_url' or 'audio_base64'."}

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".audio") as tmp:
            tmp_path = tmp.name

        if url:
            _download(url, tmp_path)
        else:
            with open(tmp_path, "wb") as f:
                f.write(base64.b64decode(b64))

        t0 = time.time()
        segments, info = model.transcribe(
            tmp_path,
            language=inp.get("language", "en"),
            beam_size=int(inp.get("beam_size", 5)),
            vad_filter=bool(inp.get("vad_filter", True)),
            word_timestamps=bool(inp.get("word_timestamps", False)),
            initial_prompt=inp.get("initial_prompt"),
            condition_on_previous_text=False,
        )

        seg_list, texts = [], []
        for s in segments:  # generator: decoding happens here
            texts.append(s.text.strip())
            seg = {
                "start": round(s.start, 2),
                "end": round(s.end, 2),
                "text": s.text.strip(),
            }
            if inp.get("word_timestamps"):
                seg["words"] = [
                    {"word": w.word, "start": round(w.start, 2),
                     "end": round(w.end, 2), "prob": round(w.probability, 3)}
                    for w in (s.words or [])
                ]
            seg_list.append(seg)

        elapsed = time.time() - t0
        out = {
            "text": " ".join(texts),
            "language": info.language,
            "audio_duration": round(info.duration, 2),
            "processing_time": round(elapsed, 2),
            "model": MODEL_NAME,
        }
        if inp.get("return_segments", True):
            out["segments"] = seg_list
        return out

    except Exception as e:
        return {"error": str(e)}
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


runpod.serverless.start({"handler": handler})
