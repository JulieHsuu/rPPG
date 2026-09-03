from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class VisionSample:
    rgb_mean: tuple[float, float, float] | None
    bbox: tuple[int, int, int, int] | None
    motion_score: float
    exposure_score: float
    roi_pixels: int
    roi_quality: float = 0.0


class FaceROIExtractor:
    """Zero-manual-model rPPG ROI extractor using OpenCV's packaged Haar cascade.

    This intentionally avoids a separately downloaded landmark model for the first PoC.
    """

    def __init__(self) -> None:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.detector = cv2.CascadeClassifier(cascade_path)
        if self.detector.empty():
            raise RuntimeError(f"Unable to load OpenCV face cascade: {cascade_path}")
        self.last_bbox: tuple[int, int, int, int] | None = None
        self.last_face_gray: np.ndarray | None = None
        self.frames_since_detection = 999

    @staticmethod
    def _clip_rect(x1: int, y1: int, x2: int, y2: int, width: int, height: int):
        return max(0, x1), max(0, y1), min(width, x2), min(height, y2)

    def _detect_face(self, gray: np.ndarray) -> tuple[int, int, int, int] | None:
        faces = self.detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(80, 80),
            flags=cv2.CASCADE_SCALE_IMAGE,
        )
        if len(faces) == 0:
            return None
        # Largest face is the subject.
        x, y, w, h = max(faces, key=lambda f: int(f[2]) * int(f[3]))
        return int(x), int(y), int(w), int(h)

    @staticmethod
    def _motion(prev: tuple[int, int, int, int] | None, curr: tuple[int, int, int, int]) -> float:
        if prev is None:
            return 0.0
        px, py, pw, ph = prev
        x, y, w, h = curr
        scale = max(float(w + h), 1.0)
        center_shift = np.hypot((x + w / 2) - (px + pw / 2), (y + h / 2) - (py + ph / 2))
        size_shift = abs(w - pw) + abs(h - ph)
        return float(np.clip((center_shift * 2.0 + size_shift) / scale, 0.0, 1.0))

    @staticmethod
    def _smooth_bbox(
        prev: tuple[int, int, int, int] | None,
        curr: tuple[int, int, int, int],
        alpha: float = 0.20,
    ) -> tuple[int, int, int, int]:
        """Reduce Haar-box jitter that otherwise becomes a false rPPG signal."""
        if prev is None:
            return curr
        px, py, pw, ph = prev
        x, y, w, h = curr
        # A large jump is probably a new subject/detection and should not be
        # interpolated through unrelated image regions.
        if FaceROIExtractor._motion(prev, curr) > 0.55:
            return curr
        a = float(np.clip(alpha, 0.0, 1.0))
        return tuple(int(round(old * (1.0 - a) + new * a)) for old, new in zip(prev, curr))

    @staticmethod
    def _skin_mask(bgr: np.ndarray) -> np.ndarray:
        ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
        _, cr, cb = cv2.split(ycrcb)
        # Broad skin-color bounds; final mean also rejects saturated/dark pixels.
        mask = ((cr >= 130) & (cr <= 180) & (cb >= 75) & (cb <= 140)).astype(np.uint8) * 255
        return mask

    def _texture_motion(self, gray: np.ndarray, bbox: tuple[int, int, int, int]) -> float:
        """Measure motion inside the face, not only movement of the detector box."""
        x, y, w, h = bbox
        face = gray[max(0, y):y + h, max(0, x):x + w]
        if face.size == 0:
            self.last_face_gray = None
            return 1.0
        face = cv2.resize(face, (96, 96), interpolation=cv2.INTER_AREA).astype(np.float32)
        # Normalization makes the score less sensitive to a global exposure step.
        face = (face - float(np.mean(face))) / (float(np.std(face)) + 1e-6)
        previous = self.last_face_gray
        self.last_face_gray = face
        if previous is None:
            return 0.0
        difference = float(np.median(np.abs(face - previous)))
        return float(np.clip(difference / 0.65, 0.0, 1.0))

    @staticmethod
    def _trimmed_rgb_mean(rgb_pixels: np.ndarray) -> np.ndarray:
        """Reject highlights and deep shadows before averaging a skin region."""
        if rgb_pixels.shape[0] < 20:
            return np.mean(rgb_pixels, axis=0)
        low, high = np.percentile(rgb_pixels, [10, 90], axis=0)
        keep = np.all((rgb_pixels >= low) & (rgb_pixels <= high), axis=1)
        selected = rgb_pixels[keep]
        return np.mean(selected if selected.shape[0] >= 20 else rgb_pixels, axis=0)

    def sample(self, frame_bgr: np.ndarray) -> VisionSample:
        h_img, w_img = frame_bgr.shape[:2]
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        detected_bbox = self._detect_face(gray)
        if detected_bbox is None:
            self.last_bbox = None
            self.last_face_gray = None
            return VisionSample(None, None, 1.0, 0.0, 0)

        previous_bbox = self.last_bbox
        bbox_motion = self._motion(previous_bbox, detected_bbox)
        bbox = self._smooth_bbox(previous_bbox, detected_bbox)
        x, y, w, h = bbox
        self.last_bbox = bbox
        texture_motion = self._texture_motion(gray, bbox)
        motion = float(np.clip(max(bbox_motion, texture_motion), 0.0, 1.0))

        # Three rPPG-friendly rectangles: forehead + left/right cheeks.
        rects = [
            (x + int(0.30 * w), y + int(0.12 * h), x + int(0.70 * w), y + int(0.30 * h)),
            (x + int(0.12 * w), y + int(0.48 * h), x + int(0.38 * w), y + int(0.70 * h)),
            (x + int(0.62 * w), y + int(0.48 * h), x + int(0.88 * w), y + int(0.70 * h)),
        ]

        pixels = []
        roi_means = []
        exposure_vals = []
        valid_coverages = []
        for rect in rects:
            x1, y1, x2, y2 = self._clip_rect(*rect, w_img, h_img)
            if x2 <= x1 or y2 <= y1:
                continue
            roi = frame_bgr[y1:y2, x1:x2]
            if roi.size == 0:
                continue
            mask = self._skin_mask(roi)
            rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            luminance = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            valid = (mask > 0) & (luminance > 35) & (luminance < 245)
            if np.count_nonzero(valid) < 30:
                valid = (luminance > 35) & (luminance < 245)
            if np.count_nonzero(valid) < 30:
                continue
            valid_pixels = rgb[valid].reshape(-1, 3).astype(np.float64)
            pixels.append(valid_pixels)
            roi_means.append(self._trimmed_rgb_mean(valid_pixels))
            exposure_vals.append(luminance[valid])
            valid_coverages.append(float(np.count_nonzero(valid)) / float(valid.size))

        if not pixels:
            return VisionSample(None, bbox, motion, 0.0, 0)

        all_pixels = np.vstack(pixels).astype(np.float64)
        all_lum = np.concatenate(exposure_vals).astype(np.float64)
        roi_array = np.vstack(roi_means)
        # Equal ROI weighting prevents one large cheek/highlight from dominating.
        rgb_mean = tuple(float(v) for v in np.median(roi_array, axis=0))

        median_lum = float(np.median(all_lum))
        dark_penalty = np.clip((median_lum - 35.0) / 55.0, 0.0, 1.0)
        bright_penalty = np.clip((245.0 - median_lum) / 55.0, 0.0, 1.0)
        spread = float(np.std(all_lum))
        contrast_factor = float(np.clip(spread / 18.0, 0.4, 1.0))
        skin_coverage = float(np.clip(np.mean(valid_coverages) / 0.45, 0.0, 1.0))
        if roi_array.shape[0] > 1:
            normalized = roi_array / (np.mean(roi_array, axis=1, keepdims=True) + 1e-9)
            roi_spread = float(np.mean(np.std(normalized, axis=0)))
            roi_consistency = float(np.clip(1.0 - roi_spread / 0.16, 0.0, 1.0))
        else:
            roi_consistency = 0.45
        roi_quality = float(np.clip(skin_coverage * roi_consistency, 0.0, 1.0))
        exposure = float(np.clip(min(dark_penalty, bright_penalty) * contrast_factor * (0.65 + 0.35 * roi_quality), 0.0, 1.0))

        return VisionSample(rgb_mean, bbox, motion, exposure, int(all_pixels.shape[0]), roi_quality)
