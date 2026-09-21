"""Model lifetime and CPU inference with bounded concurrency.

The service constructs exactly one Ultralytics model at startup, serializes
in-flight predictions with a semaphore (bounded backlog) and always passes
``device="cpu"`` explicitly (project invariant INV-02).
"""

from __future__ import annotations

import logging
import math
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from .config import Settings
from .schemas import Detection

logger = logging.getLogger(__name__)

DEVICE = "cpu"  # constant: the only permitted inference device (INV-02)


class ModelNotReady(Exception):
    """Inference was requested before (or without) a successful model load."""


class InferenceOverloaded(Exception):
    """The bounded inference queue is full; the caller should retry."""


@dataclass
class InferenceResult:
    detections: list[Detection]
    inference_ms: float
    width: int
    height: int = field(default=0)


ModelFactory = Callable[[Path], Any]


def _default_model_factory(weights_path: Path) -> Any:
    from ultralytics import YOLO  # deferred: heavy import, only at startup

    return YOLO(str(weights_path))


class InferenceService:
    """Owns a single pretrained model and serializes CPU prediction."""

    def __init__(self, settings: Settings, factory: ModelFactory | None = None) -> None:
        self.settings = settings
        self._factory = factory or _default_model_factory
        self._model: Any | None = None
        self._init_lock = threading.Lock()
        self._slots = threading.Semaphore(settings.inference_concurrency)
        self.ready = False
        self.load_error: str | None = None

    # -- lifetime ---------------------------------------------------------

    def startup(self) -> None:
        """Load the model once; set ``ready`` only after a warm-up inference."""
        with self._init_lock:
            if self.ready or self._model is not None:
                return
            try:
                if self.settings.torch_num_threads is not None:
                    torch.set_num_threads(self.settings.torch_num_threads)
                weights_path = self.settings.model_path()
                logger.info(
                    "Loading model %s (device=%s, imgsz=%d, conf=%s)",
                    self.settings.yolo_model,
                    DEVICE,
                    self.settings.yolo_image_size,
                    self.settings.yolo_confidence,
                )
                started = time.perf_counter()
                self._model = self._factory(weights_path)
                if not getattr(self._model, "names", None):
                    raise RuntimeError("Model did not provide a class-name mapping.")
                self._warmup()
                self.ready = True
                logger.info(
                    "Model ready in %.0f ms (torch threads: %d)",
                    (time.perf_counter() - started) * 1000,
                    torch.get_num_threads(),
                )
            except Exception as exc:  # noqa: BLE001 - surface any load failure
                self.load_error = str(exc)
                logger.exception("Model load failed; service will report 503")

    def _warmup(self) -> None:
        """One throwaway inference so first real request skips cold-start costs."""
        size = self.settings.yolo_image_size
        dummy = np.zeros((size, size, 3), dtype=np.uint8)
        self._model.predict(
            dummy,
            device=DEVICE,
            imgsz=size,
            conf=self.settings.yolo_confidence,
            verbose=False,
        )

    # -- prediction -------------------------------------------------------

    def predict(self, image: Image.Image) -> InferenceResult:
        """Run CPU inference on an orientation-corrected RGB image."""
        if not self.ready or self._model is None:
            raise ModelNotReady("The detection model is not loaded.")
        if not self._slots.acquire(blocking=False):
            raise InferenceOverloaded("All inference slots are busy.")
        try:
            array = np.asarray(image, dtype=np.uint8)
            started = time.perf_counter()
            results = self._model.predict(
                array,
                device=DEVICE,
                imgsz=self.settings.yolo_image_size,
                conf=self.settings.yolo_confidence,
                verbose=False,
            )
            inference_ms = (time.perf_counter() - started) * 1000
            return self._parse(results, image.size, inference_ms)
        finally:
            self._slots.release()

    def _parse(
        self, results: Any, size: tuple[int, int], inference_ms: float
    ) -> InferenceResult:
        width, height = size
        dets: list[Detection] = []
        boxes = getattr(results[0], "boxes", None)
        if boxes is not None and len(boxes) > 0:
            xyxy = boxes.xyxy.detach().cpu().numpy()
            confs = boxes.conf.detach().cpu().numpy()
            clss = boxes.cls.detach().cpu().numpy()
            # Most confident first; bounded by the configured maximum.
            for idx in np.argsort(-confs)[: self.settings.yolo_max_detections]:
                conf = float(confs[idx])
                if not math.isfinite(conf):
                    continue
                x1, y1, x2, y2 = (float(v) for v in xyxy[idx])
                # Clamp into the documented original-pixel space (INV-10).
                x1 = min(max(x1, 0.0), float(width))
                y1 = min(max(y1, 0.0), float(height))
                x2 = min(max(x2, 0.0), float(width))
                y2 = min(max(y2, 0.0), float(height))
                if x2 <= x1 or y2 <= y1:
                    continue  # degenerate after clamping
                class_id = int(clss[idx])
                name = self._model.names.get(class_id, f"class_{class_id}")
                dets.append(
                    Detection(
                        class_id=class_id,
                        class_name=str(name),
                        confidence=round(conf, 4),
                        box=[x1, y1, x2, y2],
                    )
                )
        return InferenceResult(
            detections=dets, inference_ms=inference_ms, width=width, height=height
        )
