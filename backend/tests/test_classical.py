import numpy as np

from app.engines.classical import _correct_camera_hrv, estimate_vitals


def synthetic_rgb(hr_bpm=72.0, fps=30.0, seconds=24.0):
    t = np.arange(0, seconds, 1.0 / fps)
    f = hr_bpm / 60.0
    pulse = np.sin(2 * np.pi * f * t)
    rng = np.random.default_rng(42)
    # Small physiological modulation added to realistic average face RGB levels.
    r = 155.0 + 0.55 * pulse + rng.normal(0, 0.08, len(t))
    g = 118.0 + 1.10 * pulse + rng.normal(0, 0.08, len(t))
    b = 103.0 + 0.35 * pulse + rng.normal(0, 0.08, len(t))
    return t.tolist(), np.column_stack([r, g, b]).tolist()


def test_chrom_tracks_synthetic_hr():
    ts, rgb = synthetic_rgb()
    result = estimate_vitals(ts, rgb, engine="chrom")
    assert result.hr_bpm is not None
    assert abs(result.hr_bpm - 72.0) <= 6.0


def test_pos_returns_a_plausible_rate():
    ts, rgb = synthetic_rgb()
    result = estimate_vitals(ts, rgb, engine="pos")
    assert result.hr_bpm is not None
    assert 45.0 <= result.hr_bpm <= 180.0


def test_pos_and_chrom_both_track_synthetic_hr():
    ts, rgb = synthetic_rgb(hr_bpm=84.0)
    for engine in ("pos", "chrom"):
        result = estimate_vitals(ts, rgb, engine=engine)
        assert result.hr_bpm is not None
        assert abs(result.hr_bpm - 84.0) <= 6.0


def test_low_quality_capture_does_not_publish_false_hr():
    ts, rgb = synthetic_rgb(hr_bpm=78.0)
    result = estimate_vitals(ts, rgb, engine="pos", motion_score=1.0, exposure_score=0.15)
    assert result.hr_bpm is None
    assert result.signal_quality < 0.40


def test_disagreement_rejects_a_persistent_50_bpm_artifact():
    fps = 30.0
    t = np.arange(0, 24.0, 1.0 / fps)
    pulse = np.sin(2 * np.pi * (78.0 / 60.0) * t)
    artifact = np.sin(2 * np.pi * (50.0 / 60.0) * t)
    base = np.array([155.0, 118.0, 103.0])
    pulse_rgb = np.array([0.55, 1.10, 0.35])
    artifact_rgb = np.array([-4.19, 1.07, -1.24])
    rgb = base + np.outer(pulse, pulse_rgb) + np.outer(artifact, artifact_rgb)

    # POS locks onto the 50 BPM artifact while CHROM sees the 78 BPM pulse.
    # The public result must withhold the number instead of claiming either one.
    result = estimate_vitals(t, rgb, engine="pos")
    assert result.hr_bpm is None
    assert result.signal_quality < 0.40


def test_short_window_hides_hr():
    ts, rgb = synthetic_rgb(seconds=4.0)
    result = estimate_vitals(ts, rgb, engine="chrom")
    assert result.hr_bpm is None


def test_short_window_hides_experimental_hrv():
    ts, rgb = synthetic_rgb(seconds=24.0)
    result = estimate_vitals(ts, rgb, engine="chrom")
    assert result.hrv_rmssd_ms is None

def test_respiration_proxy_detects_slow_modulation():
    fps = 30.0
    seconds = 45.0
    t = np.arange(0, seconds, 1.0 / fps)
    pulse = np.sin(2 * np.pi * (72.0 / 60.0) * t)
    resp = np.sin(2 * np.pi * (15.0 / 60.0) * t)
    r = 155.0 + 0.55 * pulse + 0.8 * resp
    g = 118.0 + 1.10 * pulse + 0.8 * resp
    b = 103.0 + 0.35 * pulse + 0.8 * resp
    result = estimate_vitals(t.tolist(), np.column_stack([r, g, b]).tolist(), engine="chrom")
    assert result.rr_bpm is not None
    assert abs(result.rr_bpm - 15.0) <= 2.0


def test_long_clean_signal_produces_hrv_and_spo2_proxies():
    ts, rgb = synthetic_rgb(seconds=45.0)
    result = estimate_vitals(ts, rgb, engine="chrom")
    assert result.hrv_rmssd_ms is not None
    assert result.spo2_percent is not None
    assert 90.0 <= result.spo2_percent <= 99.0
    assert result.hrv_rmssd_ms < 30.0


def test_respiration_prefers_pulse_envelope_over_slow_light_drift():
    fps = 30.0
    seconds = 45.0
    t = np.arange(0, seconds, 1.0 / fps)
    envelope = 1.0 + 0.22 * np.sin(2 * np.pi * (20.0 / 60.0) * t)
    pulse = envelope * np.sin(2 * np.pi * (84.0 / 60.0) * t)
    light_drift = 0.65 * np.sin(2 * np.pi * (9.0 / 60.0) * t)
    r = 155.0 + 0.55 * pulse + light_drift
    g = 118.0 + 1.10 * pulse + light_drift
    b = 103.0 + 0.35 * pulse + light_drift
    result = estimate_vitals(t, np.column_stack([r, g, b]), engine="chrom")

    assert result.rr_bpm is not None
    assert abs(result.rr_bpm - 20.0) <= 3.0


def test_camera_hrv_correction_reduces_frame_jitter_inflation():
    corrected = _correct_camera_hrv(raw_rmssd=70.0, fs=25.0, quality=0.5)
    assert corrected is not None
    assert 27.0 <= corrected <= 31.0


def test_camera_hrv_value_does_not_change_with_lighting_quality():
    dim = _correct_camera_hrv(raw_rmssd=55.0, fs=25.0, quality=0.4)
    bright = _correct_camera_hrv(raw_rmssd=55.0, fs=25.0, quality=0.9)
    assert dim == bright
