import easyocr
import cv2
import numpy as np

class JerseyOCR:
    def __init__(self):
        # Initialize EasyOCR with English digits
        self.reader = easyocr.Reader(['en'], gpu=False) # Use GPU=True if available

    def identify_number(self, crop):
        """
        Processes a crop of a player and attempts to find a number.
        Uses adaptive bicubic upscaling and CLAHE equalization for high recall on distant game video.
        """
        num, _ = self.identify_number_with_conf(crop)
        return num

    def identify_number_with_conf(self, crop):
        """
        Processes player crop and returns (number_str, confidence_float) or (None, 0.0).
        """
        if crop is None or crop.size == 0:
            return None, 0.0

        # Convert to grayscale
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop

        # Ensure minimum resolution for EasyOCR CRAFT text detector
        h, w = gray.shape[:2]
        if h < 64 or w < 64:
            scale = max(64.0 / max(1, h), 64.0 / max(1, w), 2.5)
            new_w = int(w * scale)
            new_h = int(h * scale)
            gray = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

        results = self.reader.readtext(gray, allowlist='0123456789')
        if not results:
            return None, 0.0

        # Return the result with the highest confidence
        # results is a list of [bbox, text, confidence]
        results.sort(key=lambda x: x[2], reverse=True)
        return str(results[0][1]), float(results[0][2])
