"""Shared fixtures and deterministic fakes for mocked-inference tests.

A mocked detector is *not* proof that the real model works; the real-model
integration check lives in ``test_live_model.py`` (marker: ``live``).
"""

from __future__ import annotations

import io
from typing import Any

import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image

from backend.config import Settings
from backend.inference import InferenceService
from backend.main import create_app


@pytest.fixture(autouse=True)
def _restore_pil_bomb_limit():
    """Reset PIL's global decompression-bomb limit after each test.

    ``backend.image_io.decode_image`` intentionally sets
    ``Image.MAX_IMAGE_PIXELS`` per request; without this fixture the value
    would leak into other tests' direct ``Image.open`` calls.
    """
    original = Image.MAX_IMAGE_PIXELS
    yield
    Image.MAX_IMAGE_PIXELS = original


# ------------------------------------------------------------------ images


def jpeg_bytes(width: int = 320, height: int = 240, color=(180, 60, 60)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (width, height), color).save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def png_bytes(width: int = 320, height: int = 240, color=(60, 180, 60)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (width, height), color).save(buf, format="PNG")
    return buf.getvalue()


def upload(client: TestClient, data: bytes, filename: str = "image.jpg",
           content_type: str = "image/jpeg"):
    return client.post("/detect", files={"file": (filename, data, content_type)})


# -------------------------------------------------------------------- fakes


class FakeBoxes:
    """Mimics the tensor attributes of ``ultralytics.models.yolo.detect.Results.boxes``."""

    def __init__(self, xyxy: list, conf: list, cls: list) -> None:
        self.xyxy = torch.tensor(xyxy, dtype=torch.float32)
        self.conf = torch.tensor(conf, dtype=torch.float32)
        self.cls = torch.tensor(cls, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.conf)


class FakeResult:
    def __init__(self, boxes: FakeBoxes, orig_shape: tuple[int, int]) -> None:
        self.boxes = boxes
        self.orig_shape = orig_shape


def behavior_boxes(boxes: list, conf: list, cls: list):
    """Factory: model returns fixed boxes (original-image pixel space)."""

    def _behavior(model: Any, source: Any) -> list[FakeResult]:
        height, width = source.shape[:2]
        return [FakeResult(FakeBoxes(boxes, conf, cls), (height, width))]

    return _behavior


def behavior_empty(_model: Any, source: Any) -> list[FakeResult]:
    height, width = source.shape[:2]
    return [FakeResult(FakeBoxes([], [], []), (height, width))]


def behavior_raise(exc: Exception):
    def _behavior(_model: Any, _source: Any) -> list[FakeResult]:
        raise exc

    return _behavior


class FakeModel:
    def __init__(self, weights_path: Any, names: dict[int, str]) -> None:
        self.weights_path = weights_path
        self.names = names
        self.predict_calls: list[dict] = []
        self.behavior = behavior_empty

    def predict(self, source: Any, **kwargs: Any) -> list[FakeResult]:
        self.predict_calls.append({"source": source, "kwargs": kwargs})
        return self.behavior(self, source)


class FakeModelFactory:
    """Callable passed to :class:`InferenceService` instead of ``YOLO``."""

    def __init__(
        self,
        names: dict[int, str] | None = None,
        fail_load: bool = False,
        behavior=None,
    ) -> None:
        self.names = names or {0: "person", 5: "bus", 16: "chair"}
        self.fail_load = fail_load
        self.behavior = behavior or behavior_empty
        self.models: list[FakeModel] = []

    def __call__(self, weights_path: Any) -> FakeModel:
        if self.fail_load:
            # Deliberately contains a path + "secret" so tests can prove the
            # HTTP layer does not leak load-failure internals.
            raise RuntimeError("simulated load failure: /secret/path/token=abc123")
        model = FakeModel(weights_path, self.names)
        model.behavior = self.behavior
        self.models.append(model)
        return model


# ------------------------------------------------------------------ fixtures


@pytest.fixture()
def make_app():
    """Build an app wired to a fake model factory supplied by the test."""

    def _make(settings: Settings | None = None, factory: FakeModelFactory | None = None):
        factory = factory or FakeModelFactory()
        return create_app(
            settings or Settings(),
            make_service=lambda s, f=factory: InferenceService(s, factory=f),
        )

    return _make


@pytest.fixture()
def app_and_client(make_app):
    """App + TestClient with a fake model returning one person box by default."""
    factory = FakeModelFactory(
        behavior=behavior_boxes(
            boxes=[[10.0, 20.0, 100.0, 200.0]], conf=[0.9], cls=[0]
        )
    )
    app = make_app(factory=factory)
    with TestClient(app) as client:
        yield client, factory
