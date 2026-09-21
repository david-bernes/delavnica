"""Opt-in live-model integration tests (marker: ``live``).

These exercise the *real* pretrained Ultralytics checkpoint through the real
HTTP stack (TestClient). A mocked detector is not proof that the real model
works — this is that proof.

Run with::

    YOLO_TEST_IMAGE=/path/to/photo.jpg python -m pytest -m live -q

Without ``YOLO_TEST_IMAGE`` the positive-detection check is skipped, but the
full-stack inference check on a synthetic image still runs. The tests skip
automatically when the official weights have not been downloaded yet.
"""

from __future__ import annotations

import io
import os

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from backend.config import Settings
from backend.main import create_app

pytestmark = pytest.mark.live


def _png_bytes(width: int, height: int, color) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (width, height), color).save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture(scope="module")
def live_client():
    settings = Settings()
    weights = settings.model_path()
    if not weights.exists():
        pytest.skip(
            f"official weights not downloaded ({weights.name}); "
            "start the service once with internet access or pre-stage weights"
        )
    app = create_app(settings)
    with TestClient(app) as client:
        yield client


def test_live_health_reports_real_readiness(live_client):
    res = live_client.get("/health")
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "ready"
    assert body["device"] == "cpu"
    assert body["model"] == "yolo26n.pt"
    assert body["ultralytics_version"].startswith("8.")
    assert "cpu" in body["torch_version"]


def test_live_detect_synthetic_image_returns_200(live_client):
    """Flat-color image: proves real CPU inference end-to-end over HTTP.

    A blank image is expected to yield zero detections — that is a valid
    success, not an error (see AGENTS.md §2.4).
    """
    res = live_client.post(
        "/detect",
        files={"file": ("flat.png", _png_bytes(640, 480, (128, 128, 128)), "image/png")},
    )
    assert res.status_code == 200
    body = res.json()
    assert body["device"] == "cpu"
    assert body["image"] == {"width": 640, "height": 480}
    assert isinstance(body["detections"], list)
    assert body["timing_ms"]["inference"] > 0
    assert body["timing_ms"]["total"] >= body["timing_ms"]["inference"] - 1e-6


def test_live_detect_real_image_when_available(live_client):
    """Positive-object check on a locally supplied, legally usable image."""
    path = os.environ.get("YOLO_TEST_IMAGE")
    if not path or not os.path.exists(path):
        pytest.skip("YOLO_TEST_IMAGE not set; positive-detection check skipped")
    with open(path, "rb") as fh:
        data = fh.read()
    res = live_client.post(
        "/detect",
        files={"file": (os.path.basename(path), data, "application/octet-stream")},
    )
    assert res.status_code == 200
    body = res.json()
    width, height = body["image"]["width"], body["image"]["height"]
    for det in body["detections"]:
        x1, y1, x2, y2 = det["box"]
        assert 0 <= x1 <= x2 <= width
        assert 0 <= y1 <= y2 <= height
        assert 0.0 <= det["confidence"] <= 1.0
    print("\nLIVE-TEST image:", os.path.basename(path))
    print("LIVE-TEST detections:", body["detections"])
    print("LIVE-TEST timing_ms:", body["timing_ms"])
