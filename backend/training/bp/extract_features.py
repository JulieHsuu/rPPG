"""Extract camera-rPPG features from CLBP-style videos.

Expected filename: Subject001_M44_138_99_66_448.mov
                    subject   age SBP DBP HR lux
The script never writes video frames; it only saves aggregate signal features.
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.stats import kurtosis, skew

BACKEND_ROOT = Path(__file__).resolve().parents[2]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.engines.classical import estimate_vitals

NAME = re.compile(r"(Subject\d+)_([MF])(\d+)_(\d+)_(\d+)_(\d+)_(\d+)", re.I)
FEATURES = [
    "estimated_hr", "hrv_rmssd", "signal_quality", "red_cv", "green_cv", "blue_cv",
    "rg_ac_ratio", "rb_ac_ratio", "pulse_std", "pulse_skew", "pulse_kurtosis",
    "pulse_diff_std", "spectral_entropy",
]


def parse_label(path: Path) -> dict[str, object]:
    match = NAME.search(path.stem)
    if not match:
        raise ValueError(f"無法從檔名解析標籤：{path.name}")
    subject, sex, age, sbp, dbp, reference_hr, lux = match.groups()
    return {"subject_id": subject, "sex": sex.upper(), "age": int(age), "sbp": int(sbp),
            "dbp": int(dbp), "reference_hr": int(reference_hr), "lux": int(lux)}


def first_face(cap: cv2.VideoCapture, width: int = 640) -> tuple[int, int, int, int]:
    cascade = cv2.CascadeClassifier(str(Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"))
    best = None
    for _ in range(45):
        ok, frame = cap.read()
        if not ok:
            break
        scale = width / frame.shape[1]
        small = cv2.resize(frame, (width, round(frame.shape[0] * scale)))
        faces = cascade.detectMultiScale(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), 1.12, 5, minSize=(90, 90))
        if len(faces):
            best = max(faces, key=lambda b: b[2] * b[3])
            break
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    if best is None:
        raise RuntimeError("影片中找不到臉部")
    return tuple(int(v) for v in best)


def roi_rgb(frame: np.ndarray, face: tuple[int, int, int, int], width: int = 640) -> np.ndarray | None:
    scale = width / frame.shape[1]
    small = cv2.resize(frame, (width, round(frame.shape[0] * scale)))
    x, y, w, h = face
    boxes = [(x+.25*w,y+.12*h,.50*w,.18*h),(x+.12*w,y+.47*h,.25*w,.20*h),(x+.63*w,y+.47*h,.25*w,.20*h)]
    means = []
    for bx, by, bw, bh in boxes:
        roi = small[max(0,int(by)):int(by+bh), max(0,int(bx)):int(bx+bw)]
        if roi.size == 0:
            continue
        rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB).reshape(-1, 3).astype(float)
        lo, hi = np.percentile(rgb, [10, 90], axis=0)
        mask = np.all((rgb >= lo) & (rgb <= hi), axis=1)
        if mask.sum() > 30:
            means.append(np.mean(rgb[mask], axis=0))
    return np.median(means, axis=0) if means else None


def extract(path: Path) -> dict[str, object]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError("無法開啟影片")
    source_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    face = first_face(cap)
    step = max(1, round(source_fps / 25.0))
    timestamps, rgb = [], []
    index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if index % step == 0:
            value = roi_rgb(frame, face)
            if value is not None:
                timestamps.append(index / source_fps)
                rgb.append(value)
        index += 1
    cap.release()
    if len(rgb) < 500:
        raise RuntimeError(f"有效影格不足：{len(rgb)}")
    values = np.asarray(rgb)
    result = estimate_vitals(timestamps, values, engine="pos", motion_score=0.0, exposure_score=1.0)
    pulse = np.asarray(result.waveform, dtype=float)
    power = np.abs(np.fft.rfft(pulse - pulse.mean())) ** 2
    prob = power / (power.sum() + 1e-12)
    entropy = float(-np.sum(prob * np.log(prob + 1e-12)) / np.log(max(2, len(prob))))
    means = values.mean(axis=0); ac = values.std(axis=0) / np.maximum(means, 1e-9)
    row = parse_label(path)
    row.update({
        "video": path.name, "valid_seconds": round(timestamps[-1] - timestamps[0], 2),
        "estimated_hr": result.hr_bpm, "hrv_rmssd": result.hrv_rmssd_ms,
        "signal_quality": result.signal_quality, "red_cv": ac[0], "green_cv": ac[1], "blue_cv": ac[2],
        "rg_ac_ratio": ac[0] / max(ac[1], 1e-9), "rb_ac_ratio": ac[0] / max(ac[2], 1e-9),
        "pulse_std": float(np.std(pulse)), "pulse_skew": float(skew(pulse)),
        "pulse_kurtosis": float(kurtosis(pulse)), "pulse_diff_std": float(np.std(np.diff(pulse))),
        "spectral_entropy": entropy,
    })
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("video_dir", type=Path)
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("features.csv"))
    args = parser.parse_args()
    videos = sorted(p for p in args.video_dir.rglob("*") if p.suffix.lower() in {".mov", ".mp4", ".avi"})
    rows = []
    for i, video in enumerate(videos, 1):
        try:
            row = extract(video); rows.append(row)
            print(f"[{i}/{len(videos)}] 完成 {video.name}：HR={row['estimated_hr']}，SQI={row['signal_quality']}")
        except Exception as exc:
            print(f"[{i}/{len(videos)}] 略過 {video.name}：{exc}")
    if not rows:
        raise SystemExit("沒有可輸出的資料")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    print(f"已輸出 {len(rows)} 位受試者：{args.output}")


if __name__ == "__main__":
    main()
