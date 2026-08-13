from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.signal import butter, filtfilt, find_peaks, welch

EngineName = Literal["pos", "chrom"]


@dataclass
class VitalEstimate:
    hr_bpm: float | None
    rr_bpm: float | None
    hrv_rmssd_ms: float | None
    signal_quality: float
    waveform: list[float]
    engine: str
    fps: float
    window_seconds: float


def _safe_bandpass(x: np.ndarray, fs: float, low: float, high: float, order: int = 3) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.size < max(24, order * 8) or fs <= 0:
        return x - np.mean(x)
    nyq = fs / 2.0
    high = min(high, nyq * 0.95)
    if low <= 0 or high <= low:
        return x - np.mean(x)
    b, a = butter(order, [low / nyq, high / nyq], btype="band")
    padlen = 3 * max(len(a), len(b))
    if x.size <= padlen:
        return x - np.mean(x)
    return filtfilt(b, a, x)


def _resample_uniform(timestamps: np.ndarray, rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    ts = np.asarray(timestamps, dtype=np.float64)
    vals = np.asarray(rgb, dtype=np.float64)
    if len(ts) < 2:
        return ts, vals, 0.0
    duration = ts[-1] - ts[0]
    if duration <= 0:
        return ts, vals, 0.0
    fps = (len(ts) - 1) / duration
    fps = float(np.clip(fps, 5.0, 120.0))
    n = max(2, int(round(duration * fps)) + 1)
    uniform_ts = np.linspace(ts[0], ts[-1], n)
    uniform_rgb = np.column_stack([
        np.interp(uniform_ts, ts, vals[:, c]) for c in range(3)
    ])
    return uniform_ts, uniform_rgb, fps


def pos_signal(rgb: np.ndarray, fps: float, window_seconds: float = 1.6) -> np.ndarray:
    """POS rPPG signal, implemented from the projection-on-orthogonal-subspace principle.

    Input rgb is Nx3 with channel means from facial ROIs.
    """
    c = np.asarray(rgb, dtype=np.float64).T  # 3 x N
    n = c.shape[1]
    if n < 8:
        return np.zeros(n, dtype=np.float64)
    l = max(8, int(round(window_seconds * fps)))
    h = np.zeros(n, dtype=np.float64)
    wsum = np.zeros(n, dtype=np.float64)
    p = np.array([[0.0, 1.0, -1.0], [-2.0, 1.0, 1.0]], dtype=np.float64)

    for end in range(l, n + 1):
        start = end - l
        segment = c[:, start:end]
        mean = np.mean(segment, axis=1, keepdims=True)
        mean[np.abs(mean) < 1e-9] = 1.0
        cn = segment / mean
        s = p @ cn
        s0_std = np.std(s[0])
        s1_std = np.std(s[1])
        alpha = s0_std / (s1_std + 1e-9)
        hs = s[0] + alpha * s[1]
        hs -= np.mean(hs)
        win = np.hanning(l)
        if not np.any(win):
            win = np.ones(l)
        h[start:end] += hs * win
        wsum[start:end] += win

    valid = wsum > 1e-9
    h[valid] /= wsum[valid]
    h[~valid] = 0.0
    return h


def chrom_signal(rgb: np.ndarray) -> np.ndarray:
    c = np.asarray(rgb, dtype=np.float64)
    means = np.mean(c, axis=0)
    means[np.abs(means) < 1e-9] = 1.0
    cn = c / means - 1.0
    r, g, b = cn[:, 0], cn[:, 1], cn[:, 2]
    x = 3.0 * r - 2.0 * g
    y = 1.5 * r + g - 1.5 * b
    alpha = np.std(x) / (np.std(y) + 1e-9)
    h = x - alpha * y
    return h - np.mean(h)


def _dominant_rate(signal: np.ndarray, fs: float, low_hz: float, high_hz: float) -> tuple[float | None, float]:
    if signal.size < max(32, int(fs * 4)):
        return None, 0.0
    nperseg = min(signal.size, max(64, int(fs * 8)))
    # Zero-padding gives smoother peak localization without pretending the temporal
    # window itself has higher information content.
    nfft = max(2048, 1 << int(np.ceil(np.log2(max(nperseg, 2)))))
    freqs, power = welch(signal, fs=fs, nperseg=nperseg, nfft=nfft, detrend="constant")
    mask = (freqs >= low_hz) & (freqs <= high_hz)
    if not np.any(mask):
        return None, 0.0
    band_power = power[mask]
    band_freqs = freqs[mask]
    total = float(np.sum(band_power)) + 1e-12
    idx = int(np.argmax(band_power))
    peak_freq = float(band_freqs[idx])
    local = np.abs(band_freqs - peak_freq) <= 0.10
    local_power = float(np.sum(band_power[local]))
    rate = peak_freq * 60.0
    concentration = float(np.clip((local_power / total) * 1.7, 0.0, 1.0))
    return rate, concentration


def _pulse_consensus(rgb: np.ndarray, fs: float, engine: EngineName) -> tuple[float | None, float, np.ndarray]:
    """Estimate pulse only when POS and CHROM agree on the spectral peak.

    Webcam motion, auto-exposure and small ROI shifts commonly create a strong
    0.7-1.0 Hz component. A single-spectrum argmax can therefore sit around
    45-60 BPM indefinitely even when that component is not a pulse. POS and
    CHROM react differently to those artifacts, so disagreement is a useful
    reason to withhold the number rather than display false precision.
    """
    signals = {
        "pos": _safe_bandpass(pos_signal(rgb, fs), fs, 0.70, 3.00, order=3),
        "chrom": _safe_bandpass(chrom_signal(rgb), fs, 0.70, 3.00, order=3),
    }
    primary = signals[engine]
    secondary_name: EngineName = "chrom" if engine == "pos" else "pos"
    primary_rate, primary_quality = _dominant_rate(primary, fs, 0.70, 3.00)
    secondary_rate, secondary_quality = _dominant_rate(signals[secondary_name], fs, 0.70, 3.00)

    if primary_rate is None or secondary_rate is None:
        return None, 0.0, primary

    disagreement = abs(primary_rate - secondary_rate)
    if disagreement > 10.0:
        return None, min(primary_quality, secondary_quality) * 0.2, primary

    # Weight the selected engine more heavily while retaining the independent
    # estimate as a guard against a motion/illumination peak.
    total_weight = primary_quality + secondary_quality
    if total_weight <= 1e-9:
        return None, 0.0, primary
    rate = (primary_rate * primary_quality + secondary_rate * secondary_quality) / total_weight
    agreement_factor = float(np.clip(1.0 - disagreement / 10.0, 0.0, 1.0))
    quality = min(primary_quality, secondary_quality) * (0.65 + 0.35 * agreement_factor)
    return rate, quality, primary


def _estimate_hrv(filtered: np.ndarray, fs: float) -> float | None:
    if filtered.size < int(fs * 12):
        return None
    prominence = max(np.std(filtered) * 0.25, 1e-6)
    min_distance = max(1, int(fs * 60.0 / 180.0))
    peaks, _ = find_peaks(filtered, distance=min_distance, prominence=prominence)
    if peaks.size < 5:
        return None
    ibi_ms = np.diff(peaks) / fs * 1000.0
    ibi_ms = ibi_ms[(ibi_ms >= 333.0) & (ibi_ms <= 1500.0)]
    if ibi_ms.size < 4:
        return None
    successive = np.diff(ibi_ms)
    if successive.size == 0:
        return None
    return float(np.sqrt(np.mean(successive ** 2)))


def _estimate_respiration(rgb: np.ndarray, fs: float) -> tuple[float | None, float]:
    """Experimental respiratory-rate proxy from slow RGB intensity variation.

    This is intentionally separate from the pulse waveform to reduce beat/envelope
    artifacts being misreported as respiration. It remains a PoC-grade estimate.
    """
    rgb = np.asarray(rgb, dtype=np.float64)
    if rgb.shape[0] < int(fs * 18):
        return None, 0.0
    means = np.mean(rgb, axis=0)
    means[np.abs(means) < 1e-9] = 1.0
    normalized = rgb / means - 1.0
    # Green carries the strongest PPG component, while averaging channels makes
    # this low-frequency proxy less tied to the pulse projection itself.
    source = 0.25 * normalized[:, 0] + 0.50 * normalized[:, 1] + 0.25 * normalized[:, 2]
    resp = _safe_bandpass(source, fs, 0.10, 0.50, order=2)
    # Require a meaningful low-frequency RMS relative to overall normalized RGB noise.
    rms_resp = float(np.sqrt(np.mean(resp ** 2)))
    rms_source = float(np.sqrt(np.mean(source ** 2))) + 1e-12
    if rms_resp / rms_source < 0.08:
        return None, 0.0
    return _dominant_rate(resp, fs, 0.10, 0.50)


def estimate_vitals(
    timestamps: list[float] | np.ndarray,
    rgb: list[list[float]] | np.ndarray,
    engine: EngineName = "pos",
    motion_score: float = 0.0,
    exposure_score: float = 1.0,
) -> VitalEstimate:
    ts, values, fps = _resample_uniform(np.asarray(timestamps), np.asarray(rgb))
    if len(ts) < 2 or fps <= 0:
        return VitalEstimate(None, None, None, 0.0, [], engine, fps, 0.0)

    hr, spectral_quality, filtered = _pulse_consensus(values, fps, engine)
    rr, rr_quality = _estimate_respiration(values, fps)
    hrv = _estimate_hrv(filtered, fps)

    # Penalize camera/subject motion and poor exposure. Inputs are normalized to 0..1.
    motion_factor = float(np.clip(1.0 - motion_score, 0.0, 1.0))
    exposure_factor = float(np.clip(exposure_score, 0.0, 1.0))
    quality = spectral_quality * (0.65 + 0.35 * motion_factor) * exposure_factor
    if hr is None:
        quality = 0.0
    quality = float(np.clip(quality, 0.0, 1.0))

    # Do not surface a BPM when the spectrum or capture conditions are weak.
    # The UI can then ask the subject to hold still instead of showing a stable,
    # misleading low-frequency artifact as a heart rate.
    if quality < 0.40:
        hr = None

    wave = filtered[-min(len(filtered), int(max(1, fps * 8))):]
    if wave.size:
        denom = float(np.max(np.abs(wave))) + 1e-9
        wave = wave / denom
    window_seconds = float(ts[-1] - ts[0])

    # Avoid displaying unstable metrics before a useful temporal window has accumulated.
    if window_seconds < 8.0:
        hr = None
        quality = min(quality, 0.35)
    if window_seconds < 18.0:
        rr = None
        rr_quality = 0.0
    if rr is not None and (rr_quality < 0.35 or rr < 8.0 or rr > 30.0):
        rr = None

    return VitalEstimate(
        hr_bpm=round(hr, 1) if hr is not None else None,
        rr_bpm=round(rr, 1) if rr is not None else None,
        hrv_rmssd_ms=round(hrv, 1) if hrv is not None else None,
        signal_quality=round(quality, 3),
        waveform=[round(float(v), 4) for v in wave[:: max(1, len(wave) // 180)]],
        engine=engine,
        fps=round(fps, 1),
        window_seconds=round(window_seconds, 1),
    )
