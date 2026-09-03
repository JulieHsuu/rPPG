from app.services.vision import FaceROIExtractor
import numpy as np


def test_bbox_smoothing_reduces_detector_jitter():
    previous = (100, 80, 200, 240)
    detected = (106, 76, 204, 236)
    smoothed = FaceROIExtractor._smooth_bbox(previous, detected)

    assert smoothed == (101, 79, 201, 239)
    assert FaceROIExtractor._motion(previous, smoothed) < FaceROIExtractor._motion(previous, detected)


def test_bbox_smoothing_resets_after_large_jump():
    previous = (100, 80, 200, 240)
    detected = (500, 300, 90, 100)

    assert FaceROIExtractor._smooth_bbox(previous, detected) == detected


def test_trimmed_rgb_mean_rejects_bright_highlight():
    skin = np.tile(np.array([[150.0, 112.0, 96.0]]), (100, 1))
    highlights = np.tile(np.array([[255.0, 255.0, 255.0]]), (5, 1))
    result = FaceROIExtractor._trimmed_rgb_mean(np.vstack([skin, highlights]))

    assert np.allclose(result, [150.0, 112.0, 96.0], atol=1.0)
