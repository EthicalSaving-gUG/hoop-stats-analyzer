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
        """
        if crop is None or crop.size == 0:
            return None
            
        # Optional: Preprocessing for better OCR
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        # Increase contrast
        gray = cv2.convertScaleAbs(gray, alpha=1.5, beta=0)
        
        results = self.reader.readtext(gray, allowlist='0123456789')
        
        if not results:
            return None
            
        # Return the result with the highest confidence
        # results is a list of [bbox, text, confidence]
        results.sort(key=lambda x: x[2], reverse=True)
        return results[0][1]
