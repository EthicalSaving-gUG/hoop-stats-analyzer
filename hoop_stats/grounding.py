"""Optional visual grounding adapters for LocateAnything-style models."""

from __future__ import annotations

import base64
import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Literal


TaskName = Literal["detect", "ground_single", "ground_multi", "ground_text", "detect_text", "point"]

SUPPORTED_TASKS = {
    "detect",
    "ground_single",
    "ground_multi",
    "ground_text",
    "detect_text",
    "point",
}


class GroundingUnavailable(RuntimeError):
    """Raised when the selected grounding backend is not configured or unavailable."""


@dataclass(frozen=True)
class GroundingBox:
    x1: float
    y1: float
    x2: float
    y2: float
    label: str | None = None

    def as_dict(self) -> dict[str, float | str | None]:
        return {
            "x1": self.x1,
            "y1": self.y1,
            "x2": self.x2,
            "y2": self.y2,
            "label": self.label,
        }


@dataclass(frozen=True)
class GroundingPoint:
    x: float
    y: float
    label: str | None = None

    def as_dict(self) -> dict[str, float | str | None]:
        return {"x": self.x, "y": self.y, "label": self.label}


def build_prompt(task: str, query: str) -> str:
    query = (query or "").strip()
    if task not in SUPPORTED_TASKS:
        raise ValueError(f"Unsupported grounding task: {task}")
    if task != "detect_text" and not query:
        raise ValueError("Grounding query must not be empty")

    if task == "detect":
        categories = "</c>".join([part.strip() for part in query.split(",") if part.strip()])
        return f"Locate all the instances that matches the following description: {categories}."
    if task == "ground_single":
        return f"Locate a single instance that matches the following description: {query}."
    if task == "ground_multi":
        return f"Locate all the instances that match the following description: {query}."
    if task == "ground_text":
        return f"Please locate the text referred as {query}."
    if task == "detect_text":
        return "Detect all the text in box format."
    return f"Point to: {query}."


def parse_boxes(answer: str, image_width: int, image_height: int) -> list[GroundingBox]:
    boxes: list[GroundingBox] = []
    for match in re.finditer(r"<box><(\d+)><(\d+)><(\d+)><(\d+)></box>", answer or ""):
        x1, y1, x2, y2 = [int(group) for group in match.groups()]
        boxes.append(
            GroundingBox(
                x1=max(0.0, min(image_width, x1 / 1000 * image_width)),
                y1=max(0.0, min(image_height, y1 / 1000 * image_height)),
                x2=max(0.0, min(image_width, x2 / 1000 * image_width)),
                y2=max(0.0, min(image_height, y2 / 1000 * image_height)),
            )
        )
    return boxes


def parse_points(answer: str, image_width: int, image_height: int) -> list[GroundingPoint]:
    points: list[GroundingPoint] = []
    for match in re.finditer(r"<box><(\d+)><(\d+)></box>", answer or ""):
        x, y = [int(group) for group in match.groups()]
        points.append(
            GroundingPoint(
                x=max(0.0, min(image_width, x / 1000 * image_width)),
                y=max(0.0, min(image_height, y / 1000 * image_height)),
            )
        )
    return points


def boxes_to_yolo_labels(
    boxes: list[dict[str, Any]],
    image_width: int,
    image_height: int,
    class_id: int = 0,
) -> list[str]:
    """Convert pixel-space boxes to YOLO txt label rows."""
    if image_width <= 0 or image_height <= 0:
        raise ValueError("Image width and height must be positive")

    rows: list[str] = []
    for box in boxes:
        x1 = float(box["x1"])
        y1 = float(box["y1"])
        x2 = float(box["x2"])
        y2 = float(box["y2"])
        cx = ((x1 + x2) / 2) / image_width
        cy = ((y1 + y2) / 2) / image_height
        width = abs(x2 - x1) / image_width
        height = abs(y2 - y1) / image_height
        rows.append(f"{class_id} {cx:.6f} {cy:.6f} {width:.6f} {height:.6f}")
    return rows


def _image_to_data_url(image_path: Path) -> str:
    suffix = image_path.suffix.lower().lstrip(".")
    mime = "image/jpeg" if suffix in {"jpg", "jpeg"} else f"image/{suffix or 'png'}"
    payload = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{payload}"


class OpenAICompatibleGrounder:
    """Adapter for local vLLM/SGLang OpenAI-compatible LocateAnything servers."""

    def __init__(self, api_url: str, model: str, timeout: float = 120.0):
        self.api_url = api_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def locate(self, image_path: Path, task: str, query: str, mode: str = "hybrid") -> dict[str, Any]:
        prompt = build_prompt(task, query)
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": _image_to_data_url(image_path)}},
                    ],
                }
            ],
            "max_tokens": int(os.getenv("HOOP_STATS_GROUNDING_MAX_TOKENS", "2048")),
            "temperature": float(os.getenv("HOOP_STATS_GROUNDING_TEMPERATURE", "0.0")),
            "extra_body": {"generation_mode": mode},
        }
        request = urllib.request.Request(
            f"{self.api_url}/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise GroundingUnavailable(f"Grounding API not reachable at {self.api_url}") from exc

        choices = data.get("choices") or []
        message = choices[0].get("message", {}) if choices else {}
        answer = message.get("content") or data.get("answer") or ""
        return {"answer": answer, "prompt": prompt, "backend": "openai"}


class LocalLocateAnythingGrounder:
    """Direct Transformers adapter. Heavy; only loads when explicitly selected."""

    def __init__(self, model_path: str, device: str, dtype_name: str):
        try:
            import torch
            from transformers import AutoModel, AutoProcessor, AutoTokenizer
        except ImportError as exc:
            raise GroundingUnavailable("Install torch, transformers and Pillow to use local grounding") from exc

        dtype = getattr(torch, dtype_name, torch.bfloat16)
        self.device = device
        self.dtype = dtype
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=dtype,
            trust_remote_code=True,
        ).to(device).eval()
        self.torch = torch

    def locate(self, image_path: Path, task: str, query: str, mode: str = "hybrid") -> dict[str, Any]:
        try:
            from PIL import Image
        except ImportError as exc:
            raise GroundingUnavailable("Install Pillow to load images for grounding") from exc

        prompt = build_prompt(task, query)
        image = Image.open(image_path).convert("RGB")
        messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}]
        text = self.processor.py_apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        images, videos = self.processor.process_vision_info(messages)
        inputs = self.processor(text=[text], images=images, videos=videos, return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            response = self.model.generate(
                pixel_values=inputs["pixel_values"].to(self.dtype),
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                image_grid_hws=inputs.get("image_grid_hws"),
                tokenizer=self.tokenizer,
                max_new_tokens=int(os.getenv("HOOP_STATS_GROUNDING_MAX_TOKENS", "2048")),
                use_cache=True,
                generation_mode=mode,
                temperature=float(os.getenv("HOOP_STATS_GROUNDING_TEMPERATURE", "0.7")),
                do_sample=os.getenv("HOOP_STATS_GROUNDING_SAMPLE", "1") == "1",
                top_p=0.9,
                repetition_penalty=1.1,
                verbose=False,
            )
        answer = response[0] if isinstance(response, tuple) else response
        return {"answer": answer, "prompt": prompt, "backend": "local"}


def grounding_status() -> dict[str, Any]:
    backend = os.getenv("HOOP_STATS_GROUNDING_BACKEND", "disabled").strip().lower()
    return {
        "backend": backend,
        "enabled": backend in {"openai", "local"},
        "model": os.getenv("HOOP_STATS_GROUNDING_MODEL", "nvidia/LocateAnything-3B"),
        "api_url": os.getenv("HOOP_STATS_GROUNDING_API_URL", "http://127.0.0.1:30000"),
    }


def create_grounder() -> OpenAICompatibleGrounder | LocalLocateAnythingGrounder:
    status = grounding_status()
    backend = status["backend"]
    model = status["model"]
    if backend == "openai":
        return OpenAICompatibleGrounder(
            api_url=status["api_url"],
            model=model,
            timeout=float(os.getenv("HOOP_STATS_GROUNDING_TIMEOUT", "120")),
        )
    if backend == "local":
        return LocalLocateAnythingGrounder(
            model_path=model,
            device=os.getenv("HOOP_STATS_GROUNDING_DEVICE", "cuda"),
            dtype_name=os.getenv("HOOP_STATS_GROUNDING_DTYPE", "bfloat16"),
        )
    raise GroundingUnavailable("Set HOOP_STATS_GROUNDING_BACKEND=openai or local to enable visual grounding")
