"""test_ocr.py — tests for hoop_stats.ocr (JerseyOCR)"""
import numpy as np
import cv2
import pytest
from hoop_stats.ocr import JerseyOCR


@pytest.fixture(scope="module")
def ocr():
    return JerseyOCR()


class TestJerseyOCR:
    def test_instantiates(self, ocr):
        assert ocr is not None

    def test_returns_string_or_none_on_black_crop(self, ocr):
        crop = np.zeros((64, 64, 3), dtype=np.uint8)
        result = ocr.identify_number(crop)
        assert result is None or isinstance(result, str)

    def test_does_not_crash_on_tiny_crop(self, ocr):
        crop = np.zeros((4, 4, 3), dtype=np.uint8)
        result = ocr.identify_number(crop)
        assert result is None or isinstance(result, str)

    def test_synthetic_number_crop(self, ocr):
        """
        Draw the number '23' in white on a dark background and pass to OCR.
        Result may not be perfect, but should not raise an exception.
        """
        crop = np.zeros((80, 80, 3), dtype=np.uint8)
        cv2.putText(
            crop, "23", (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX, 1.8, (255, 255, 255), 3
        )
        result = ocr.identify_number(crop)
        # OCR may return None or a string depending on the OCR backend
        assert result is None or isinstance(result, str)
        print(f"\nOCR result for '23' crop: {result!r}")
