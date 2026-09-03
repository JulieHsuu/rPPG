from pathlib import Path

from training.bp.extract_features import FEATURES, parse_label
from training.bp.train_model import MIN_SUBJECTS, split


def test_clbp_filename_parser_extracts_reference_labels():
    row = parse_label(Path("Subject004_F42_134_86_97_750.mov"))
    assert row == {
        "subject_id": "Subject004", "sex": "F", "age": 42,
        "sbp": 134, "dbp": 86, "reference_hr": 97, "lux": 750,
    }


def test_subject_split_is_deterministic_and_training_is_guarded():
    assert split("Subject123") == split("Subject123")
    assert split("Subject123") in {"train", "validation", "test"}
    assert MIN_SUBJECTS >= 60


def test_features_are_signal_based_not_demographic_shortcuts():
    assert "age" not in FEATURES
    assert "sex" not in FEATURES
    assert {"pulse_std", "spectral_entropy", "estimated_hr"}.issubset(FEATURES)
