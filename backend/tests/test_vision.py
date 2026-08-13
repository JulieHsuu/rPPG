from app.services.vision import FaceROIExtractor


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
