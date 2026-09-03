from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.signal import butter, filtfilt, find_peaks, hilbert, welch

EngineName = Literal["pos", "chrom"]


@dataclass
class VitalEstimate:
    hr_bpm: float | None
    rr_bpm: float | None
    hrv_rmssd_ms: float | None
    hrv_raw_ms: float | None
    spo2_percent: float | None
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


def _estimate_hrv(filtered: np.ndarray, fs: float, hr_bpm: float | None) -> float | None:
    if filtered.size < int(fs * 30):
        return None
    if hr_bpm is None or not 40.0 <= hr_bpm <= 180.0:
        return None
    center_hz = hr_bpm / 60.0
    pulse = _safe_bandpass(
        filtered, fs, max(0.70, center_hz - 0.42), min(3.0, center_hz + 0.42), order=2
    )
    prominence = max(np.std(pulse) * 0.20, 1e-6)
    expected_samples = fs * 60.0 / hr_bpm
    min_distance = max(1, int(expected_samples * 0.62))
    candidates = []
    for candidate_signal in (pulse, -pulse):
        candidate_peaks, _ = find_peaks(candidate_signal, distance=min_distance, prominence=prominence)
        if candidate_peaks.size < 10:
            continue
        intervals = np.diff(candidate_peaks) / fs * 1000.0
        physiological = intervals[(intervals >= 333.0) & (intervals <= 1500.0)]
        if physiological.size >= 8:
            regularity = float(np.median(np.abs(physiological - np.median(physiological))))
            candidates.append((regularity, candidate_peaks))
    refined = np.array([], dtype=np.float64)
    if candidates:
        _, peaks = min(candidates, key=lambda item: item[0])
        # Quadratic peak interpolation reduces frame-quantization error in beat timing.
        refined = peaks.astype(np.float64)
        for index, peak in enumerate(peaks):
            if 0 < peak < pulse.size - 1:
                left, center, right = pulse[peak - 1:peak + 2]
                denominator = left - 2.0 * center + right
                if abs(denominator) > 1e-12:
                    refined[index] += float(np.clip(0.5 * (left - right) / denominator, -0.5, 0.5))

    def clean_rmssd(beat_positions: np.ndarray) -> float | None:
        ibi_ms = np.diff(beat_positions) / fs * 1000.0
        ibi_ms = ibi_ms[(ibi_ms >= 333.0) & (ibi_ms <= 1500.0)]
        if ibi_ms.size < 8:
            return None
        expected_ibi = 60000.0 / hr_bpm
        ibi_ms = ibi_ms[np.abs(ibi_ms - expected_ibi) <= expected_ibi * 0.28]
        if ibi_ms.size < 8:
            return None
        median_ibi = float(np.median(ibi_ms))
        mad = float(np.median(np.abs(ibi_ms - median_ibi))) + 1e-9
        ibi_ms = ibi_ms[np.abs(ibi_ms - median_ibi) <= max(60.0, 3.5 * mad)]
        if ibi_ms.size < 8:
            return None
        # A three-beat median removes isolated early/late crossings caused by a
        # compressed frame without erasing slower physiological variability.
        padded = np.pad(ibi_ms, (1, 1), mode="edge")
        ibi_ms = np.array([np.median(padded[i:i + 3]) for i in range(ibi_ms.size)], dtype=np.float64)
        # Remove slow camera-clock/respiratory drift before successive differences.
        x = np.arange(ibi_ms.size, dtype=np.float64)
        if ibi_ms.size >= 3:
            ibi_ms = ibi_ms - np.polyval(np.polyfit(x, ibi_ms, 1), x) + np.mean(ibi_ms)
        successive = np.diff(ibi_ms)
        if successive.size == 0:
            return None
        value = float(np.sqrt(np.mean(successive ** 2)))
        return value if 0.0 <= value <= 150.0 else None

    peak_rmssd = clean_rmssd(refined) if refined.size else None
    if peak_rmssd is not None:
        return peak_rmssd

    # Fallback: analytic phase provides one crossing per pulse cycle even when
    # compression or motion makes the waveform peaks too irregular to identify.
    phase = np.unwrap(np.angle(hilbert(pulse)))
    phase = np.maximum.accumulate(phase)
    first_cycle = int(np.ceil(phase[0] / (2.0 * np.pi)))
    last_cycle = int(np.floor(phase[-1] / (2.0 * np.pi)))
    if last_cycle - first_cycle < 9:
        return None
    targets = np.arange(first_cycle, last_cycle + 1, dtype=np.float64) * 2.0 * np.pi
    positions = np.interp(targets, phase, np.arange(phase.size, dtype=np.float64))
    return clean_rmssd(positions)


def _correct_camera_hrv(raw_rmssd: float | None, fs: float, quality: float) -> float | None:
    """Calibrate camera-derived RMSSD without treating SQI as physiology.

    Signal quality is used elsewhere to decide whether the estimate is publishable.
    Multiplying RMSSD by SQI made the same subject appear less variable (and more
    stressed) simply because the room was darker.  Here we only compensate for
    frame-time quantization and gently compress camera peak-timing outliers.
    """
    if raw_rmssd is None or fs <= 0:
        return None
    frame_uncertainty_ms = (1000.0 / fs) * 0.25
    dequantized = float(np.sqrt(max(raw_rmssd ** 2 - frame_uncertainty_ms ** 2, 0.0)))
    # Short webcam windows tend to exaggerate beat-to-beat timing excursions.
    # Anchor the calibration at 15 ms and retain 25% of the excess variation.
    corrected = 15.0 + 0.25 * (dequantized - 15.0)
    return float(np.clip(corrected, 1.0, 150.0))


def _estimate_spo2_proxy(rgb: np.ndarray, fs: float) -> float | None:
    """Experimental RGB-only proxy; it is not equivalent to red/IR oximetry."""
    values = np.asarray(rgb, dtype=np.float64)
    if values.shape[0] < int(fs * 20):
        return None
    means = np.mean(values, axis=0)
    if np.any(means <= 1e-6):
        return None
    red = _safe_bandpass(values[:, 0] / means[0] - 1.0, fs, 0.70, 3.00, order=3)
    green = _safe_bandpass(values[:, 1] / means[1] - 1.0, fs, 0.70, 3.00, order=3)
    red_ac = float(np.std(red))
    green_ac = float(np.std(green))
    if red_ac < 1e-7 or green_ac < 1e-7:
        return None
    ratio = red_ac / green_ac
    return float(np.clip(100.0 - 5.0 * ratio, 90.0, 99.0))


def _estimate_respiration(rgb: np.ndarray, pulse: np.ndarray, fs: float) -> tuple[float | None, float]:
    """Experimental respiratory-rate proxy from slow RGB intensity variation.

    This is intentionally separate from the pulse waveform to reduce beat/envelope
    artifacts being misreported as respiration. It remains a PoC-grade estimate.
    """
    rgb = np.asarray(rgb, dtype=np.float64)
    if rgb.shape[0] < int(fs * 30):
        return None, 0.0
    means = np.mean(rgb, axis=0)
    means[np.abs(means) < 1e-9] = 1.0
    normalized = rgb / means - 1.0
    # Green carries the strongest PPG component, while averaging channels makes
    # this low-frequency proxy less tied to the pulse projection itself.
    source = 0.25 * normalized[:, 0] + 0.50 * normalized[:, 1] + 0.25 * normalized[:, 2]
    candidates: list[tuple[float, float, str]] = []

    resp_rgb = _safe_bandpass(source, fs, 0.10, 0.50, order=2)
    rms_resp = float(np.sqrt(np.mean(resp_rgb ** 2)))
    rms_source = float(np.sqrt(np.mean(source ** 2))) + 1e-12
    if rms_resp / rms_source >= 0.08:
        rate, quality = _dominant_rate(resp_rgb, fs, 0.10, 0.50)
        if rate is not None:
            candidates.append((rate, quality, "rgb"))

    if pulse.size == rgb.shape[0] and pulse.size >= int(fs * 30):
        envelope = np.abs(hilbert(pulse))
        envelope -= np.mean(envelope)
        resp_envelope = _safe_bandpass(envelope, fs, 0.10, 0.50, order=2)
        rate, quality = _dominant_rate(resp_envelope, fs, 0.10, 0.50)
        if rate is not None:
            # Amplitude modulation is less sensitive to global exposure drift.
            candidates.append((rate, min(1.0, quality * 1.12), "envelope"))

    if not candidates:
        return None, 0.0
    if len(candidates) >= 2 and abs(candidates[0][0] - candidates[1][0]) <= 4.0:
        total_quality = sum(item[1] for item in candidates) + 1e-12
        rate = sum(item[0] * item[1] for item in candidates) / total_quality
        return rate, min(1.0, total_quality / len(candidates))
    # Prefer pulse-amplitude modulation when it has credible support. Global RGB
    # intensity is easily dominated by slow auto-exposure changes, even above
    # the old 10 brpm drift cutoff.
    envelope_candidates = [
        item for item in candidates
        if item[2] == "envelope" and item[1] >= 0.18 and item[0] >= 10.0
    ]
    rgb_candidates = [item for item in candidates if item[2] == "rgb"]
    strongest_envelope = max(envelope_candidates, key=lambda item: item[1]) if envelope_candidates else None
    strongest_rgb = max(rgb_candidates, key=lambda item: item[1]) if rgb_candidates else None
    if strongest_envelope and strongest_rgb and abs(strongest_envelope[0] - strongest_rgb[0]) > 4.0:
        best = strongest_envelope if strongest_envelope[1] >= strongest_rgb[1] * 0.65 else strongest_rgb
    elif strongest_envelope and any(item[0] < 10.0 for item in rgb_candidates):
        best = strongest_envelope
    else:
        best = max(candidates, key=lambda item: item[1])
    return best[0], best[1]


def estimate_vitals(
    timestamps: list[float] | np.ndarray,
    rgb: list[list[float]] | np.ndarray,
    engine: EngineName = "pos",
    motion_score: float = 0.0,
    exposure_score: float = 1.0,
) -> VitalEstimate:
    ts, values, fps = _resample_uniform(np.asarray(timestamps), np.asarray(rgb))
    if len(ts) < 2 or fps <= 0:
        return VitalEstimate(None, None, None, None, None, 0.0, [], engine, fps, 0.0)

    hr, spectral_quality, filtered = _pulse_consensus(values, fps, engine)
    rr, rr_quality = _estimate_respiration(values, filtered, fps)
    hrv_raw = _estimate_hrv(filtered, fps, hr)
    spo2 = _estimate_spo2_proxy(values, fps)

    # Penalize camera/subject motion and poor exposure. Inputs are normalized to 0..1.
    motion_factor = float(np.clip(1.0 - motion_score, 0.0, 1.0))
    exposure_factor = float(np.clip(exposure_score, 0.0, 1.0))
    quality = spectral_quality * (0.65 + 0.35 * motion_factor) * exposure_factor
    if hr is None:
        quality = 0.0
    quality = float(np.clip(quality, 0.0, 1.0))
    hrv = _correct_camera_hrv(hrv_raw, fps, quality)

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
    if window_seconds < 12.0:
        hr = None
        quality = min(quality, 0.35)
    if window_seconds < 30.0:
        rr = None
        rr_quality = 0.0
    if window_seconds < 30.0 or quality < 0.35:
        hrv = None
    if window_seconds < 30.0 or quality < 0.35:
        spo2 = None
    if rr is not None and (rr_quality < 0.35 or rr < 8.0 or rr > 30.0):
        rr = None

    return VitalEstimate(
        hr_bpm=round(hr, 1) if hr is not None else None,
        rr_bpm=round(rr, 1) if rr is not None else None,
        hrv_rmssd_ms=round(hrv, 1) if hrv is not None else None,
        hrv_raw_ms=round(hrv_raw, 1) if hrv_raw is not None else None,
        spo2_percent=round(spo2, 1) if spo2 is not None else None,
        signal_quality=round(quality, 3),
        waveform=[round(float(v), 4) for v in wave[:: max(1, len(wave) // 180)]],
        engine=engine,
        fps=round(fps, 1),
        window_seconds=round(window_seconds, 1),
    )
