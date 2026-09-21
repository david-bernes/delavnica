"""HTTP API tests with a deterministic mocked detector (offline, no weights)."""

from __future__ import annotations

import threading

from fastapi.testclient import TestClient

from tests.conftest import (
    FakeModelFactory,
    behavior_boxes,
    behavior_raise,
    jpeg_bytes,
    png_bytes,
    upload,
)

# ------------------------------------------------------------- health (TEST-01, TEST-02)


def test_health_reports_ready(app_and_client):
    client, _ = app_and_client
    res = client.get("/health")
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "ready"
    assert body["model"] == "yolo26n.pt"
    assert body["device"] == "cpu"
    assert isinstance(body["ultralytics_version"], str) and body["ultralytics_version"]
    assert body["image_size"] == 640
    assert body["confidence"] == 0.25


def test_health_reports_not_ready_when_model_fails(make_app):
    factory = FakeModelFactory(fail_load=True)
    app = make_app(factory=factory)
    with TestClient(app) as client:
        res = client.get("/health")
        assert res.status_code == 503
        body = res.json()
        assert body["status"] == "not_ready"
        assert body["device"] == "cpu"
        assert body["error"]["code"] == "model_not_loaded"
        # No internal paths or exception details may leak.
        assert "/" not in body["error"]["message"]
        assert "secret" not in body["error"]["message"].lower()
        # /detect must also refuse with 503 (no fabricated detections).
        det = upload(client, jpeg_bytes())
        assert det.status_code == 503
        assert det.json()["error"]["code"] == "model_not_loaded"


# -------------------------------------------------------------- /detect (TEST-03..05)


def test_detect_returns_documented_structure(app_and_client):
    client, _ = app_and_client
    res = upload(client, jpeg_bytes(320, 240))
    assert res.status_code == 200
    body = res.json()
    assert body["model"] == "yolo26n.pt"
    assert body["device"] == "cpu"
    assert body["image"] == {"width": 320, "height": 240}
    assert isinstance(body["detections"], list)
    det = body["detections"][0]
    assert set(det) == {"class_id", "class_name", "confidence", "box"}
    assert det["class_id"] >= 0
    assert isinstance(det["confidence"], (int, float))
    assert 0.0 <= det["confidence"] <= 1.0
    assert len(det["box"]) == 4
    timing = body["timing_ms"]
    assert isinstance(timing["total"], (int, float))
    assert isinstance(timing["inference"], (int, float))
    assert timing["total"] >= 0
    assert timing["inference"] >= 0
    assert timing["total"] >= timing["inference"] - 1e-6


def test_detection_coordinates_and_class_mapping(app_and_client):
    client, factory = app_and_client
    res = upload(client, jpeg_bytes())
    assert res.status_code == 200
    body = res.json()
    # The fake model returns one box in original-image pixel space; it must
    # come back unchanged (no letterbox/tensor-space remapping).
    assert body["detections"] == [
        {
            "class_id": 0,
            "class_name": "person",
            "confidence": 0.9,
            "box": [10.0, 20.0, 100.0, 200.0],
        }
    ]
    # Class name must come from the model's mapping, not the raw id.
    assert factory.models[0].names[0] == "person"


def test_zero_detections_is_successful(make_app):
    from tests.conftest import behavior_empty

    factory = FakeModelFactory(behavior=behavior_empty)
    app = make_app(factory=factory)
    with TestClient(app) as client:
        res = upload(client, jpeg_bytes())
    assert res.status_code == 200
    assert res.json()["detections"] == []


def test_multiple_detections_sorted_by_confidence(make_app):
    factory = FakeModelFactory(
        behavior=behavior_boxes(
            boxes=[[10.0, 10.0, 50.0, 60.0], [80.0, 20.0, 150.0, 90.0]],
            conf=[0.4, 0.95],
            cls=[5, 0],
        )
    )
    app = make_app(factory=factory)
    with TestClient(app) as client:
        res = upload(client, jpeg_bytes())
    assert res.status_code == 200
    dets = res.json()["detections"]
    assert [d["class_name"] for d in dets] == ["person", "bus"]
    assert [d["confidence"] for d in dets] == [0.95, 0.4]
    assert dets[0]["box"] == [80.0, 20.0, 150.0, 90.0]
    assert dets[1]["box"] == [10.0, 10.0, 50.0, 60.0]


# ------------------------------------------------------------- error paths (TEST-06..08)


def test_invalid_image_returns_controlled_error(app_and_client):
    client, _ = app_and_client
    res = upload(client, b"this is definitely not an image", "fake.jpg", "image/jpeg")
    assert res.status_code == 400
    body = res.json()
    assert body["error"]["code"] == "invalid_image"
    assert body["error"]["message"]
    # No traceback / internals in the body.
    assert "Traceback" not in res.text


def test_corrupt_truncated_image_rejected(app_and_client):
    client, _ = app_and_client
    data = jpeg_bytes(200, 150)
    res = upload(client, data[: int(len(data) * 0.4)], "trunc.jpg", "image/jpeg")
    assert res.status_code == 400
    assert res.json()["error"]["code"] == "invalid_image"


def test_unsupported_format_rejected(app_and_client):
    client, _ = app_and_client
    import io

    from PIL import Image as PILImage

    buf = io.BytesIO()
    PILImage.new("RGB", (64, 64), (9, 9, 9)).save(buf, format="GIF")
    res = upload(client, buf.getvalue(), "anim.gif", "image/gif")
    assert res.status_code == 415
    assert res.json()["error"]["code"] == "unsupported_media_type"


def test_video_media_type_rejected(app_and_client):
    client, _ = app_and_client
    res = upload(client, b"\x00\x00\x00", "clip.mp4", "video/mp4")
    assert res.status_code == 415
    assert "video" in res.json()["error"]["message"].lower()


def test_oversized_compressed_upload_rejected(make_app):
    from backend.config import Settings

    factory = FakeModelFactory()
    app = make_app(settings=Settings(max_upload_bytes=4096), factory=factory)
    with TestClient(app) as client:
        res = upload(client, jpeg_bytes(600, 450))  # well over 4 KiB
    assert res.status_code == 413
    assert res.json()["error"]["code"] == "image_too_large"


def test_oversized_decoded_image_rejected(make_app):
    from backend.config import Settings

    factory = FakeModelFactory()
    # 200x200 = 40_000 px > 10_000 px limit.
    app = make_app(settings=Settings(max_image_pixels=10_000), factory=factory)
    with TestClient(app) as client:
        res = upload(client, png_bytes(200, 200), "big.png", "image/png")
    assert res.status_code == 413
    assert res.json()["error"]["code"] == "image_too_large"


def test_missing_file_field_returns_400(app_and_client):
    client, _ = app_and_client
    res = client.post("/detect")
    assert res.status_code == 400
    assert res.json()["error"]["code"] == "missing_file"


# ------------------------------------------------- lifetime & device (TEST-09, TEST-10)


def test_model_initialized_once_across_requests(app_and_client):
    client, factory = app_and_client
    for _ in range(3):
        res = upload(client, jpeg_bytes())
        assert res.status_code == 200
    assert len(factory.models) == 1  # exactly one model instance
    # One warm-up at startup + one prediction per request.
    assert len(factory.models[0].predict_calls) == 4


def test_cpu_device_explicitly_selected(app_and_client):
    client, factory = app_and_client
    upload(client, jpeg_bytes())
    calls = factory.models[0].predict_calls
    assert calls
    for call in calls:  # warm-up and every request
        assert call["kwargs"]["device"] == "cpu"
        assert call["kwargs"]["imgsz"] == 640
        assert call["kwargs"]["conf"] == 0.25


def test_internal_inference_failure_is_sanitized(app_and_client):
    client, factory = app_and_client
    # Simulate an unexpected model failure with sensitive-looking details.
    factory.models[0].behavior = behavior_raise(
        RuntimeError("CUDA out of memory at /secret/weights.pt token=abc123")
    )
    res = upload(client, jpeg_bytes())
    assert res.status_code == 500
    body = res.json()
    assert body["error"]["code"] == "internal_error"
    assert "secret" not in res.text
    assert "token" not in res.text
    assert "CUDA" not in res.text


def test_overload_returns_429_with_retry_after(make_app):
    import time as _time

    from backend.config import Settings

    gate_sleep = 0.6

    def slow(_model, source):
        _time.sleep(gate_sleep)
        from tests.conftest import FakeBoxes, FakeResult

        height, width = source.shape[:2]
        return [FakeResult(FakeBoxes([], [], []), (height, width))]

    factory = FakeModelFactory(behavior=slow)
    app = make_app(settings=Settings(inference_concurrency=1), factory=factory)
    with TestClient(app) as first:
        started = threading.Event()

        def worker():
            started.set()
            upload(first, jpeg_bytes())

        thread = threading.Thread(target=worker)
        thread.start()
        started.wait()
        _time.sleep(0.25)  # first request is now inside inference

        second = TestClient(app)  # no lifespan: service already started
        res = upload(second, jpeg_bytes())
        thread.join()

    assert res.status_code == 429
    assert res.headers.get("Retry-After") == "1"
    assert res.json()["error"]["code"] == "inference_overloaded"


# ------------------------------------------------------------------ frontend (TEST-12)


def test_root_serves_html5_frontend(app_and_client):
    client, _ = app_and_client
    res = client.get("/")
    assert res.status_code == 200
    assert "text/html" in res.headers["content-type"]
    assert "delavnica" in res.text
    assert 'id="detect"' in res.text


def test_static_assets_served(app_and_client):
    client, _ = app_and_client
    for path, content_type in (
        ("/static/app.js", "javascript"),
        ("/static/styles.css", "css"),
    ):
        res = client.get(path)
        assert res.status_code == 200, path
        assert content_type in res.headers["content-type"], path


def test_docs_endpoint_available(app_and_client):
    client, _ = app_and_client
    res = client.get("/docs")
    assert res.status_code == 200
    assert "text/html" in res.headers["content-type"]
