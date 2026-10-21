import os
from ultralytics import YOLO
import torch
from .primitives import Ball, Rim, PlayerCapsule, bbox_to_ball, bbox_to_player, bbox_to_rim

try:
    from sahi import AutoDetectionModel
    from sahi.predict import get_sliced_prediction
    HAS_SAHI = True
except ImportError:
    HAS_SAHI = False


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
        
        # 1. Check for NVIDIA CUDA & ONNX Runtime CUDA Acceleration
        if torch.cuda.is_available():
            self.device_name = f"GPU (CUDA: {torch.cuda.get_device_name(0)})"
            try:
                import onnxruntime as ort
                if 'CUDAExecutionProvider' in ort.get_available_providers():
                    onnx_path = model_variant.replace('.pt', '.onnx')
                    if os.path.exists(onnx_path):
                        optimized_model_path = onnx_path
                        self.device_name += " [ONNX CUDA Accelerated]"
            except ImportError:
                pass
            self.model = YOLO(optimized_model_path if optimized_model_path != model_variant else base_model)
            
        # 2. Check for Windows DirectML / OpenVINO ONNX Execution Providers
        else:
            try:
                import onnxruntime as ort
                providers = ort.get_available_providers()
                if 'CUDAExecutionProvider' in providers:
                    self.device_name = "GPU (ONNX CUDA)"
                    onnx_path = model_variant.replace('.pt', '.onnx')
                    if os.path.exists(onnx_path):
                        optimized_model_path = onnx_path
                elif 'DmlExecutionProvider' in providers:
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
                elif 'OpenVINOExecutionProvider' in providers:
                    self.device_name = "CPU (ONNX OpenVINO)"
                    onnx_path = model_variant.replace('.pt', '.onnx')
                    if os.path.exists(onnx_path):
                        optimized_model_path = onnx_path
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

    def detect_ball(self, frame, rim_focus_boxes=None, use_sahi: bool = False) -> list[Ball]:
        """
        Detects sports balls (class 32 in COCO) and returns Ball primitives.
        Supports:
        1. Korb-Fokus-Tracking: high-res inference on focused rim zones (radius ~220px).
        2. SAHI (Sliced Aided Hyper Inference) for small 12-16px balls in wide shots.
        3. Standard full-frame YOLO with adaptive confidence fallback.
        """
        raw_boxes = self.detect_ball_raw(frame, rim_focus_boxes=rim_focus_boxes, use_sahi=use_sahi)
        return [bbox_to_ball(b[0], b[1], b[2], b[3]) for b in raw_boxes]

    def detect_ball_raw(self, frame, rim_focus_boxes=None, use_sahi: bool = False) -> list[list[float]]:
        raw_boxes = []

        # 1. Korb-Fokus-Tracking: If rim focus boxes provided, crop & run with high sensitivity
        if rim_focus_boxes:
            fh, fw = frame.shape[:2]
            for box in rim_focus_boxes:
                rx1, ry1, rx2, ry2 = box
                rx1, ry1 = max(0, int(rx1)), max(0, int(ry1))
                rx2, ry2 = min(fw, int(rx2)), min(fh, int(ry2))
                if rx2 - rx1 > 20 and ry2 - ry1 > 20:
                    patch = frame[ry1:ry2, rx1:rx2]
                    patch_res = self.model(patch, classes=[32], verbose=False, conf=0.12)
                    for r in patch_res:
                        for pbox in r.boxes:
                            bx1, by1, bx2, by2 = pbox.xyxy[0].tolist()
                            raw_boxes.append([bx1 + rx1, by1 + ry1, bx2 + rx1, by2 + ry1])

        # 2. SAHI Sliced Inference
        if (use_sahi or os.getenv("HOOP_STATS_USE_SAHI", "0") == "1") and HAS_SAHI:
            if not hasattr(self, "sahi_model") or self.sahi_model is None:
                try:
                    self.sahi_model = AutoDetectionModel.from_pretrained(
                        model_type='yolov8',
                        model_path=self.selected_model,
                        confidence_threshold=0.15,
                        device='cuda:0' if torch.cuda.is_available() else 'cpu'
                    )
                except Exception:
                    self.sahi_model = None

            if getattr(self, "sahi_model", None) is not None:
                try:
                    sahi_res = get_sliced_prediction(
                        frame,
                        self.sahi_model,
                        slice_height=320,
                        slice_width=320,
                        overlap_height_ratio=0.2,
                        overlap_width_ratio=0.2,
                        verbose=0
                    )
                    for pred in sahi_res.object_prediction_list:
                        if pred.category.id == 32 or "ball" in str(pred.category.name).lower():
                            bbox = pred.bbox
                            raw_boxes.append([bbox.minx, bbox.miny, bbox.maxx, bbox.maxy])
                except Exception:
                    pass

        # 3. Standard full-frame YOLO detection (conf=0.18 for higher recall on balls)
        if not raw_boxes:
            results = self.model(frame, classes=[32], verbose=False, conf=0.18)
            for r in results:
                for rbox in r.boxes:
                    raw_boxes.append(rbox.xyxy[0].tolist())

        # Deduplicate overlapping boxes
        if len(raw_boxes) <= 1:
            return raw_boxes

        deduped = []
        for b in raw_boxes:
            is_dup = False
            for d in deduped:
                ix1 = max(b[0], d[0])
                iy1 = max(b[1], d[1])
                ix2 = min(b[2], d[2])
                iy2 = min(b[3], d[3])
                inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                area_b = (b[2] - b[0]) * (b[3] - b[1])
                area_d = (d[2] - d[0]) * (d[3] - d[1])
                union = area_b + area_d - inter
                if union > 0 and (inter / union) > 0.4:
                    is_dup = True
                    break
            if not is_dup:
                deduped.append(b)

        return deduped

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

