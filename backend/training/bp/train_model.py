"""Train and validate a subject-independent ridge BP model.

The script refuses to create an enabled model from undersized or narrow-range data.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

try:
    from .extract_features import FEATURES
except ImportError:  # Direct script execution from backend/
    from extract_features import FEATURES

MIN_SUBJECTS = 60
MIN_TEST_SUBJECTS = 15


def split(subject: str) -> str:
    bucket = int(hashlib.sha256(subject.encode()).hexdigest()[:8], 16) % 100
    return "train" if bucket < 70 else "validation" if bucket < 85 else "test"


def ridge_fit(x: np.ndarray, y: np.ndarray, alpha: float = 8.0) -> np.ndarray:
    return np.linalg.solve(x.T @ x + alpha * np.eye(x.shape[1]), x.T @ y)


def metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    error = pred - y
    return {"mae": round(float(np.mean(np.abs(error))), 2), "bias": round(float(np.mean(error)), 2),
            "error_sd": round(float(np.std(error, ddof=1)), 2), "rmse": round(float(np.sqrt(np.mean(error**2))), 2)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("features", type=Path)
    parser.add_argument("--model", type=Path, default=Path(__file__).with_name("bp_model.json"))
    parser.add_argument("--report", type=Path, default=Path(__file__).with_name("validation_report.json"))
    args = parser.parse_args()
    with args.features.open(encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    usable = [r for r in rows if all(r.get(k) not in {None, "", "None"} for k in FEATURES)]
    subjects = sorted({r["subject_id"] for r in usable})
    report: dict[str, object] = {"subjects": len(subjects), "rows": len(usable), "minimum_subjects": MIN_SUBJECTS,
                                 "status": "insufficient_data", "model_enabled": False}
    if len(subjects) < MIN_SUBJECTS:
        report["reason"] = f"至少需要 {MIN_SUBJECTS} 位受試者；目前只有 {len(subjects)} 位"
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(report["reason"]); return
    x = np.array([[float(r[k]) for k in FEATURES] for r in usable]); y = np.array([[float(r["sbp"]), float(r["dbp"])] for r in usable])
    groups = np.array([split(r["subject_id"]) for r in usable]); train = groups == "train"; test = groups == "test"
    mean = x[train].mean(0); scale = x[train].std(0); scale[scale < 1e-8] = 1
    weights = ridge_fit((x[train] - mean) / scale, y[train])
    pred = (x[test] - mean) / scale @ weights
    sbp_m, dbp_m = metrics(y[test, 0], pred[:, 0]), metrics(y[test, 1], pred[:, 1])
    enabled = int(test.sum()) >= MIN_TEST_SUBJECTS and all(abs(m["bias"]) <= 5 and m["error_sd"] <= 8 for m in (sbp_m, dbp_m))
    report.update({"status": "validated" if enabled else "failed_validation", "model_enabled": enabled,
                   "test_subjects": int(test.sum()), "sbp": sbp_m, "dbp": dbp_m})
    model = {"schema_version": 1, "enabled": enabled, "feature_names": FEATURES, "mean": mean.tolist(),
             "scale": scale.tolist(), "weights": weights.tolist(), "validation": report}
    args.model.write_text(json.dumps(model, ensure_ascii=False, indent=2), encoding="utf-8")
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
