import os
from ultralytics import YOLO
import torch
from .primitives import Ball, Rim, PlayerCapsule, bbox_to_ball, bbox_to_player, bbox_to_rim


DEFAULT_MODEL_CANDIDATES = ("yolo26n.pt", "yolo11n.pt", "yolov8n.pt")


def resolve_model_variant(model_variant=None):
    if model_variant:
        return model_variant

    env_model = os.getenv("HOOP_STATS_YOLO_MODEL")
    if env_model:
        return env_model

    for candidate in DEFAULT_MODEL_CANDIDATES:
        if os.path.exists(candidate):
            return candidate

    return "yolov8n.pt"


class HoopDetector:
    def __init__(self, model_variant=None, hoop_model_variant=None):
        model_variant = resolve_model_variant(model_variant)
        hoop_model_variant = hoop_model_variant or os.getenv(
            "HOOP_STATS_HOOP_MODEL",
            "models/hoop_detector_yolo11.pt",
        )
        self.selected_model = model_variant
        self.optimized_model_path = model_variant
        self.hoop_model_variant = hoop_model_variant
        self.hoop_model = self._load_optional_hoop_model(hoop_model_variant)

        # Load the base PyTorch model first (needed to check/export)
        try:
            base_model = YOLO(model_variant)
        except Exception as exc:
            fallback_model = os.getenv("HOOP_STATS_YOLO_FALLBACK", "yolov8n.pt")
            print(f"  [HoopDetector] Could not load '{model_variant}': {exc}")
            print(f"  [HoopDetector] Falling back to '{fallback_model}'.")
            model_variant = fallback_model
            self.selected_model = fallback_model
            base_model = YOLO(fallback_model)

        self.device_name = "CPU"
        optimized_model_path = model_variant

        # ── Hardware Auto-Detection & Optimization ────────────────────
        
        # 1. Check for NVIDIA CUDA
        if torch.cuda.is_available():
            self.device_name = f"GPU (CUDA: {torch.cuda.get_device_name(0)})"
            self.model = base_model
            # PyTorch runs fine natively on CUDA, no export needed
            
        # 2. Check for Windows DirectML (AMD/Intel GPUs)
        else:
            try:
                import onnxruntime as ort
                providers = ort.get_available_providers()
                if 'DmlExecutionProvider' in providers:
                    # AMD/Intel integrated GPU via DirectML
                    self.device_name = "GPU (DirectML / AMD)"
                    onnx_path = model_variant.replace('.pt', '.onnx')
                    if os.path.exists(onnx_path):
                        optimized_model_path = onnx_path
                    elif os.getenv("HOOP_STATS_AUTO_EXPORT", "0") == "1":
                        print(f"  [Optimizer] Exporting {model_variant} to ONNX for AMD/DirectML acceleration...")
                        base_model.export(format="onnx", dynamic=True)
                        optimized_model_path = onnx_path
                    else:
                        self.device_name = "CPU"
            except ImportError:
                pass

            # 3. If no GPU, optimize for CPU using OpenVINO
            if self.device_name == "CPU":
                try:
                    import openvino  # noqa: F401
                    ov_dir = model_variant.replace('.pt', '_openvino_model/')
                    if os.path.exists(ov_dir):
                        self.device_name = "CPU (OpenVINO Optimized)"
                        optimized_model_path = ov_dir
                    elif os.getenv("HOOP_STATS_AUTO_EXPORT", "0") == "1":
                        self.device_name = "CPU (OpenVINO Optimized)"
                        print(f"  [Optimizer] Exporting {model_variant} to OpenVINO for CPU acceleration...")
                        base_model.export(format="openvino")
                except ImportError:
                    pass
                    
            # Load the optimized model
            if optimized_model_path != model_variant:
                self.model = YOLO(optimized_model_path, task='detect')
            else:
                self.model = base_model
            self.optimized_model_path = optimized_model_path

    def _load_optional_hoop_model(self, hoop_model_variant):
        if not hoop_model_variant or not os.path.exists(hoop_model_variant):
            print(
                "  [HoopDetector] Hoop-specific model not found; "
                "rim detection will use fallback heuristics only."
            )
            return None

        try:
            return YOLO(hoop_model_variant)
        except Exception as exc:
            print(f"  [HoopDetector] Could not load hoop model '{hoop_model_variant}': {exc}")
            return None

    def detect_players(self, frame) -> list[PlayerCapsule]:
        """
        Detects players (class 0 in COCO) and returns PlayerCapsule primitives.
        """
        results = self.model(frame, classes=[0], verbose=False)
        detections = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                detections.append(bbox_to_player(x1, y1, x2, y2))
        return detections

    def detect_players_raw(self, frame) -> list[list[float]]:
        """
        Detects players and returns raw bounding boxes [x1, y1, x2, y2].
        Kept for backward compatibility with ROI tracker init.
        """
        results = self.model(frame, classes=[0], verbose=False)
        detections = []
        for r in results:
            for box in r.boxes:
                detections.append(box.xyxy[0].tolist())
        return detections

    def detect_ball(self, frame) -> list[Ball]:
        """
        Detects sports balls (class 32 in COCO) and returns Ball primitives.
        """
        results = self.model(frame, classes=[32], verbose=False, conf=0.25)
        detections = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                detections.append(bbox_to_ball(x1, y1, x2, y2))
        return detections

    def detect_ball_raw(self, frame) -> list[list[float]]:
        """
        Detects sports balls and returns raw bounding boxes.
        Kept for backward compatibility with ROI tracker init.
        """
        results = self.model(frame, classes=[32], verbose=False, conf=0.25)
        detections = []
        for r in results:
            for box in r.boxes:
                detections.append(box.xyxy[0].tolist())
        return detections

    def get_jersey_crop(self, frame, bbox):
        """
        Crops the player's torso area where the jersey number is likely.
        Accepts either [x1, y1, x2, y2] list or a PlayerCapsule.
        """
        if isinstance(bbox, PlayerCapsule):
            x1 = int(bbox.head.x - bbox.width / 2)
            y1 = int(bbox.head.y)
            x2 = int(bbox.head.x + bbox.width / 2)
            y2 = int(bbox.foot.y)
        else:
            x1, y1, x2, y2 = map(int, bbox)
        h_frame, w_frame = frame.shape[:2]
        x1 = max(0, min(x1, w_frame - 1))
        x2 = max(0, min(x2, w_frame))
        y1 = max(0, min(y1, h_frame - 1))
        y2 = max(0, min(y2, h_frame))
        if x2 <= x1 or y2 <= y1:
            return frame[0:0, 0:0]

        # Heuristic: Jersey number is usually in the upper-middle part of the bbox
        h = y2 - y1
        crop = frame[y1 + int(h*0.1): y1 + int(h*0.5), x1:x2]
        return crop

    def detect_rims(self, frame) -> list[Rim]:
        """
        Detects basketball hoops (class 1 in our custom model)
        and returns Rim primitives.
        """
        if self.hoop_model is None:
            return []

        results = self.hoop_model(frame, classes=[1], verbose=False, conf=0.35)
        detections = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                detections.append(bbox_to_rim(x1, y1, x2, y2))
        return detections

