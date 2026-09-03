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


def _wellness_proxies(
    hr: float | None,
    hrv: float | None,
    quality: float,
    motion: float,
    exposure: float,
) -> tuple[float | None, dict[str, float] | None]:
    """Return transparent, non-clinical proxies for the result visualization.

    These are deliberately not presented as FaceHeart's proprietary indicators.
    They only summarize the rPPG/HRV values already measured by this PoC.
    """
    if hr is None or hrv is None:
        return None, None
    q = float(np.clip(quality, 0.0, 1.0))
    stillness = float(np.clip(1.0 - motion, 0.0, 1.0))
    light = float(np.clip(exposure, 0.0, 1.0))
    hrv_norm = float(np.clip((hrv - 15.0) / 45.0, 0.0, 1.0))
    hr_rest = float(np.clip(1.0 - abs(hr - 70.0) / 45.0, 0.0, 1.0))
    # Stress is a conservative trend proxy, not a diagnosis. SQI controls how
    # strongly we depart from neutral (50), rather than altering HRV itself.
    evidence = 100.0 * (1.0 - (0.55 * hrv_norm + 0.45 * hr_rest))
    reliability = float(np.clip((q - 0.35) / 0.40, 0.0, 1.0))
    stress = round(float(np.clip(50.0 + (evidence - 50.0) * reliability, 15.0, 85.0)), 1)

    def five(value: float) -> float:
        return round(float(np.clip(1.0 + 4.0 * value, 1.0, 5.0)), 1)

    return stress, {
        "activity": five(0.50 * (1.0 - hr_rest) + 0.50 * q),
        "sleep": five(0.55 * hrv_norm + 0.45 * q),
        "equilibrium": five(0.40 * hrv_norm + 0.35 * stillness + 0.25 * q),
        "metabolism": five(0.45 * hr_rest + 0.30 * q + 0.25 * light),
        "health": five(0.40 * hrv_norm + 0.35 * hr_rest + 0.25 * q),
        "relaxation": five(0.60 * hrv_norm + 0.40 * stillness),
    }


def _blood_pressure_proxy(
    hr: float | None,
    hrv: float | None,
    stress: float | None,
    quality: float,
) -> tuple[int | None, int | None]:
    """Return an uncalibrated display proxy—not a BP measurement."""
    if hr is None or hrv is None or stress is None or quality < 0.42:
        return None, None
    hrv_load = float(np.clip(45.0 - hrv, -25.0, 35.0))
    stress_load = float(np.clip(stress - 50.0, -30.0, 30.0))
    systolic = 107.0 + 0.06 * hr + 0.035 * hrv_load + 0.025 * stress_load
    diastolic = 73.0 + 0.075 * hr + 0.025 * hrv_load + 0.015 * stress_load
    systolic = int(round(float(np.clip(systolic, 95.0, 145.0))))
    diastolic = int(round(float(np.clip(diastolic, 60.0, min(95.0, systolic - 20.0)))))
    return systolic, diastolic


@app.websocket("/ws/rppg")
async def websocket_rppg(ws: WebSocket):
    await ws.accept()
    extractor = FaceROIExtractor()
    timestamps: deque[float] = deque()
    rgb_values: deque[tuple[float, float, float]] = deque()
    motion_values: deque[float] = deque()
    exposure_values: deque[float] = deque()
    roi_quality_values: deque[float] = deque()
    hr_history: deque[float] = deque(maxlen=9)
    rr_history: deque[float] = deque(maxlen=9)
    hrv_history: deque[float] = deque(maxlen=9)
    measurement_elapsed = 0.0
    last_valid_capture: float | None = None
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
                roi_quality_values.clear()
                hr_history.clear()
                rr_history.clear()
                hrv_history.clear()
                measurement_elapsed = 0.0
                last_valid_capture = None
                last_emit = 0.0
                await ws.send_json({"type": "config_ack", "engine": engine})
                continue

            if data.get("type") != "frame":
                continue

            frame = _decode_data_url(data.get("image", ""))
            if frame is None:
                await ws.send_json({"type": "status", "status": "bad_frame"})
                continue

            sample = extractor.sample(frame)
            capture_now = float(data.get("timestamp", time.time()))

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
                # Pause accumulation on a missing face. Keeping the accepted samples
                # prevents a blink or one detector miss from resetting all progress.
                last_valid_capture = None
                await ws.send_json({
                    "type": "vitals",
                    "status": "paused" if measurement_elapsed > 0 else "no_face",
                    "hr": round(float(np.median(hr_history)), 1) if hr_history else None,
                    "rr": round(float(np.clip(np.median(rr_history) * 2.0, 8.0, 40.0)), 1) if rr_history else None,
                    "hrv_rmssd": None,
                    "hrv_raw": None,
                    "hrv_status": "waiting_for_stable_pulse",
                    "spo2": None,
                    "spo2_experimental": True,
                    "stress_index": None,
                    "wellness": None,
                    "signal_quality": 0.0,
                    "waveform": [],
                    "bbox": bbox_payload,
                    "engine": engine,
                    "window_seconds": round(measurement_elapsed, 1),
                    "measurement_progress": round(min(100.0, measurement_elapsed * 2.0), 1),
                })
                continue

            if last_valid_capture is not None:
                delta = capture_now - last_valid_capture
                if 0.0 < delta <= 0.25:
                    measurement_elapsed += min(delta, 0.12)
            last_valid_capture = capture_now
            signal_now = measurement_elapsed

            timestamps.append(signal_now)
            rgb_values.append(sample.rgb_mean)
            motion_values.append(sample.motion_score)
            exposure_values.append(sample.exposure_score)
            roi_quality_values.append(sample.roi_quality)

            # A 50-second window is long enough for more stable RR and experimental HRV.
            while timestamps and signal_now - timestamps[0] > 50.0:
                timestamps.popleft()
                rgb_values.popleft()
                motion_values.popleft()
                exposure_values.popleft()
                roi_quality_values.popleft()

            # Emit at ~4Hz even if capture is 12-15fps.
            if signal_now - last_emit < 0.24:
                continue
            last_emit = signal_now

            estimate = estimate_vitals(
                list(timestamps),
                list(rgb_values),
                engine=engine,
                motion_score=float(np.mean(list(motion_values)[-30:])) if motion_values else 0.0,
                exposure_score=float(np.mean(list(exposure_values)[-30:])) if exposure_values else 0.0,
            )

            if estimate.hr_bpm is not None:
                hr_history.append(estimate.hr_bpm)
            if estimate.rr_bpm is not None:
                rr_history.append(estimate.rr_bpm)
            if estimate.hrv_rmssd_ms is not None:
                hrv_history.append(estimate.hrv_rmssd_ms)
            stable_hr = round(float(np.median(hr_history)), 1) if len(hr_history) >= 3 else None
            stable_rr = round(float(np.clip(np.median(rr_history) * 2.0, 8.0, 40.0)), 1) if len(rr_history) >= 3 else None
            stable_hrv = round(float(np.median(hrv_history)), 1) if len(hrv_history) >= 3 else None
            mean_motion = float(np.mean(list(motion_values)[-50:])) if motion_values else 0.0
            mean_exposure = float(np.mean(list(exposure_values)[-50:])) if exposure_values else 0.0
            stress_index, wellness = _wellness_proxies(
                stable_hr, stable_hrv, estimate.signal_quality, mean_motion, mean_exposure
            )
            bp_systolic, bp_diastolic = _blood_pressure_proxy(
                stable_hr, stable_hrv, stress_index, estimate.signal_quality
            )

            status = "measuring"
            if measurement_elapsed >= 50.0 and stable_hr is not None:
                status = "complete"
            elif estimate.window_seconds >= 30.0 and stable_hr is not None:
                status = "good" if estimate.signal_quality >= 0.45 else "low_quality"

            await ws.send_json({
                "type": "vitals",
                "status": status,
                "hr": stable_hr,
                "rr": stable_rr,
                "hrv_rmssd": round(stable_hrv / 2.0, 1) if stable_hrv is not None else None,
                "hrv_raw": estimate.hrv_raw_ms,
                "hrv_status": "ok" if stable_hrv is not None else "insufficient_beat_timing",
                "spo2": estimate.spo2_percent,
                "spo2_experimental": True,
                "stress_index": stress_index,
                "bp_systolic": bp_systolic,
                "bp_diastolic": bp_diastolic,
                "bp_experimental": True,
                "wellness": wellness,
                "signal_quality": estimate.signal_quality,
                "waveform": estimate.waveform,
                "bbox": bbox_payload,
                "engine": estimate.engine,
                "fps": estimate.fps,
                "window_seconds": round(measurement_elapsed, 1),
                "measurement_progress": round(min(100.0, measurement_elapsed * 2.0), 1),
                "motion": round(float(sample.motion_score), 3),
                "exposure": round(float(sample.exposure_score), 3),
                "roi_pixels": sample.roi_pixels,
                "roi_quality": round(float(np.mean(list(roi_quality_values)[-50:])), 3) if roi_quality_values else 0.0,
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
