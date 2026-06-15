import time
import cv2
import numpy as np

def mock_get_jersey_crop(frame, box):
    return np.zeros((50, 50, 3), dtype=np.uint8)

class MockOCR:
    def identify_number(self, crop):
        time.sleep(0.05) # Simulate expensive OCR
        return None

def run_benchmark():
    ocr = MockOCR()
    player_metadata = {}
    obj_ids = [1, 2, 3]

    t0 = time.time()
    for frame_idx in range(100):
        for obj_id in obj_ids:
            if obj_id not in player_metadata:
                player_metadata[obj_id] = {"jersey": "Unknown", "team": "TBD", "ocr_attempts": 0, "ocr_cooldown": 0}

            # With cache/cooldown logic
            if player_metadata[obj_id].get("jersey", "Unknown") == "Unknown":
                if player_metadata[obj_id].get("ocr_cooldown", 0) > 0:
                    player_metadata[obj_id]["ocr_cooldown"] -= 1
                    continue

                crop = mock_get_jersey_crop(None, None)
                number = ocr.identify_number(crop)
                if number:
                    player_metadata[obj_id]["jersey"] = number
                else:
                    player_metadata[obj_id]["ocr_attempts"] = player_metadata[obj_id].get("ocr_attempts", 0) + 1
                    player_metadata[obj_id]["ocr_cooldown"] = 30 # Wait 30 frames before trying again

    t1 = time.time()
    print(f"Time taken with cache: {t1 - t0:.4f}s")

if __name__ == "__main__":
    run_benchmark()
