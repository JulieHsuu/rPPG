from app.main import _blood_pressure_proxy, _wellness_proxies


def test_wellness_proxies_require_hrv_instead_of_inventing_values():
    stress, wellness = _wellness_proxies(72.0, None, 0.8, 0.1, 0.9)
    assert stress is None
    assert wellness is None


def test_wellness_proxies_are_bounded():
    stress, wellness = _wellness_proxies(72.0, 45.0, 0.8, 0.1, 0.9)
    assert stress is not None and 0.0 <= stress <= 100.0
    assert wellness is not None
    assert set(wellness) == {"activity", "sleep", "equilibrium", "metabolism", "health", "relaxation"}
    assert all(1.0 <= value <= 5.0 for value in wellness.values())


def test_low_sqi_keeps_stress_near_neutral_instead_of_claiming_high_stress():
    stress, _ = _wellness_proxies(85.0, 24.0, 0.44, 0.1, 0.9)
    assert 45.0 <= stress < 60.0


def test_bp_proxy_matches_requested_faceheart_like_display_range():
    systolic, diastolic = _blood_pressure_proxy(85.0, 26.0, 50.0, 0.7)
    assert 110 <= systolic <= 116
    assert 77 <= diastolic <= 83


def test_bp_proxy_is_hidden_when_signal_is_unreliable():
    assert _blood_pressure_proxy(85.0, 26.0, 50.0, 0.3) == (None, None)
