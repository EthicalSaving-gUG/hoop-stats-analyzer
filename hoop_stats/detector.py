import os
from ultralytics import YOLO
import torch
from .primitives import Ball, PlayerCapsule, bbox_to_ball, bbox_to_player


class HoopDetector:
    def __init__(self, model_variant="yolov8n.pt"):
        # Load the base PyTorch model first (needed to check/export)
        try:
            base_model = YOLO(model_variant)
        except:
            model_variant = "yolov8n.pt"
            base_model = YOLO("yolov8n.pt")

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
                    if not os.path.exists(onnx_path):
                        print(f"  [Optimizer] Exporting {model_variant} to ONNX for AMD/DirectML acceleration...")
                        base_model.export(format="onnx", dynamic=True)
                    optimized_model_path = onnx_path
            except ImportError:
                pass

            # 3. If no GPU, optimize for CPU using OpenVINO
            if self.device_name == "CPU":
                try:
                    import importlib.util
                    if importlib.util.find_spec('openvino') is None:
                        raise ImportError
                    self.device_name = "CPU (OpenVINO Optimized)"
                    ov_dir = model_variant.replace('.pt', '_openvino_model/')
                    if not os.path.exists(ov_dir):
                        print(f"  [Optimizer] Exporting {model_variant} to OpenVINO for CPU acceleration...")
                        base_model.export(format="openvino")
                    optimized_model_path = ov_dir
                except ImportError:
                    pass

            # Load the optimized model
            if optimized_model_path != model_variant:
                self.model = YOLO(optimized_model_path, task='detect')
            else:
                self.model = base_model

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
        # Heuristic: Jersey number is usually in the upper-middle part of the bbox
        h = y2 - y1
        crop = frame[y1 + int(h*0.1): y1 + int(h*0.5), x1:x2]
        return crop
