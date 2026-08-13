from __future__ import annotations

import asyncio
import base64
import json
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from app.engines.classical import estimate_vitals
from app.services.vision import FaceROIExtractor

app = FastAPI(title="rPPG Web PoC", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "engine": "classical-pos/chrom",
        "manual_model_download_required": False,
    }


def _decode_data_url(data_url: str) -> np.ndarray | None:
    try:
        payload = data_url.split(",", 1)[1] if "," in data_url else data_url
        raw = base64.b64decode(payload)
        arr = np.frombuffer(raw, dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception:
        return None


@app.websocket("/ws/rppg")
async def websocket_rppg(ws: WebSocket):
    await ws.accept()
    extractor = FaceROIExtractor()
    timestamps: deque[float] = deque()
    rgb_values: deque[tuple[float, float, float]] = deque()
    motion_values: deque[float] = deque()
    exposure_values: deque[float] = deque()
    engine = "pos"
    last_emit = 0.0

    try:
        while True:
            message = await ws.receive_text()
            data = json.loads(message)

            if data.get("type") == "config":
                requested = str(data.get("engine", "pos")).lower()
                engine = requested if requested in {"pos", "chrom"} else "pos"
                timestamps.clear()
                rgb_values.clear()
                motion_values.clear()
                exposure_values.clear()
                await ws.send_json({"type": "config_ack", "engine": engine})
                continue

            if data.get("type") != "frame":
                continue

            frame = _decode_data_url(data.get("image", ""))
            if frame is None:
                await ws.send_json({"type": "status", "status": "bad_frame"})
                continue

            sample = extractor.sample(frame)
            now = float(data.get("timestamp", time.time()))

            bbox_payload = None
            if sample.bbox is not None:
                x, y, w, h = sample.bbox
                ih, iw = frame.shape[:2]
                bbox_payload = {
                    "x": x / iw,
                    "y": y / ih,
                    "w": w / iw,
                    "h": h / ih,
                }

            if sample.rgb_mean is None:
                timestamps.clear()
                rgb_values.clear()
                motion_values.clear()
                exposure_values.clear()
                await ws.send_json({
                    "type": "vitals",
                    "status": "no_face",
                    "hr": None,
                    "rr": None,
                    "hrv_rmssd": None,
                    "signal_quality": 0.0,
                    "waveform": [],
                    "bbox": bbox_payload,
                    "engine": engine,
                    "window_seconds": 0.0,
                })
                continue

            timestamps.append(now)
            rgb_values.append(sample.rgb_mean)
            motion_values.append(sample.motion_score)
            exposure_values.append(sample.exposure_score)

            # Keep 24 seconds, enough for HR and a rough respiratory estimate.
            while timestamps and now - timestamps[0] > 24.0:
                timestamps.popleft()
                rgb_values.popleft()
                motion_values.popleft()
                exposure_values.popleft()

            # Emit at ~4Hz even if capture is 12-15fps.
            if now - last_emit < 0.24:
                continue
            last_emit = now

            estimate = estimate_vitals(
                list(timestamps),
                list(rgb_values),
                engine=engine,
                motion_score=float(np.mean(list(motion_values)[-30:])) if motion_values else 0.0,
                exposure_score=float(np.mean(list(exposure_values)[-30:])) if exposure_values else 0.0,
            )

            status = "measuring"
            if estimate.window_seconds >= 8.0 and estimate.hr_bpm is not None:
                status = "good" if estimate.signal_quality >= 0.45 else "low_quality"

            await ws.send_json({
                "type": "vitals",
                "status": status,
                "hr": estimate.hr_bpm,
                "rr": estimate.rr_bpm,
                "hrv_rmssd": estimate.hrv_rmssd_ms,
                "signal_quality": estimate.signal_quality,
                "waveform": estimate.waveform,
                "bbox": bbox_payload,
                "engine": estimate.engine,
                "fps": estimate.fps,
                "window_seconds": estimate.window_seconds,
                "motion": round(float(sample.motion_score), 3),
                "exposure": round(float(sample.exposure_score), 3),
                "roi_pixels": sample.roi_pixels,
            })
            await asyncio.sleep(0)

    except WebSocketDisconnect:
        return
    except Exception as exc:
        try:
            await ws.send_json({"type": "error", "message": str(exc)})
        except Exception:
            pass


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
