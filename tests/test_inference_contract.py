"""Contract tests for configuration validation and the inference service."""

from __future__ import annotations

import threading
import time

import pytest
from PIL import Image

from backend.config import ConfigError, Settings, get_settings
from backend.inference import (
    InferenceOverloaded,
    InferenceService,
    ModelNotReady,
)
from tests.conftest import FakeModelFactory, behavior_boxes, behavior_empty

# ------------------------------------------------------------------- settings


def test_settings_defaults():
    s = get_settings({})
    assert s.yolo_model == "yolo26n.pt"
    assert s.yolo_device == "cpu"
    assert s.yolo_image_size == 640
    assert s.yolo_confidence == 0.25
    assert s.yolo_max_detections == 100
    assert s.max_upload_bytes == 8 * 1024 * 1024
    assert s.max_image_pixels == 24_000_000
    assert s.inference_concurrency == 1
    assert s.torch_num_threads is None
    assert s.host == "127.0.0.1"
    assert s.port == 8000


@pytest.mark.parametrize("device", ["cuda", "GPU", "0", "mps"])
def test_settings_reject_non_cpu_device(device):
    with pytest.raises(ConfigError):
        get_settings({"YOLO_DEVICE": device})


@pytest.mark.parametrize(
    "env",
    [
        {"YOLO_CONFIDENCE": "1.5"},
        {"YOLO_CONFIDENCE": "-0.1"},
        {"YOLO_IMAGE_SIZE": "8"},  # below minimum 16
        {"YOLO_IMAGE_SIZE": "4096"},  # above maximum 1280
        {"YOLO_MAX_DETECTIONS": "0"},
        {"INFERENCE_CONCURRENCY": "0"},
        {"PORT": "70000"},
        {"MAX_UPLOAD_BYTES": "10"},
        {"YOLO_MODEL": "  "},
    ],
)
def test_settings_reject_invalid_values(env):
    with pytest.raises(ConfigError):
        get_settings(env)


def test_settings_parse_environment_types():
    s = get_settings(
        {
            "YOLO_IMAGE_SIZE": "512",
            "YOLO_CONFIDENCE": "0.5",
            "YOLO_MAX_DETECTIONS": "42",
            "INFERENCE_CONCURRENCY": "2",
            "TORCH_NUM_THREADS": "8",
            "PORT": "9001",
            "YOLO_DEVICE": "CPU",  # case-insensitive, normalized to cpu
        }
    )
    assert s.yolo_image_size == 512
    assert s.yolo_confidence == 0.5
    assert s.yolo_max_detections == 42
    assert s.inference_concurrency == 2
    assert s.torch_num_threads == 8
    assert s.port == 9001
    assert s.yolo_device == "cpu"


# ------------------------------------------------------------ service contract


def _image(w: int = 64, h: int = 48) -> Image.Image:
    return Image.new("RGB", (w, h), (128, 64, 32))


def test_predict_before_startup_raises_model_not_ready():
    service = InferenceService(Settings(), factory=FakeModelFactory())
    with pytest.raises(ModelNotReady):
        service.predict(_image())


def test_startup_warms_up_and_marks_ready():
    factory = FakeModelFactory(behavior=behavior_empty)
    service = InferenceService(Settings(), factory=factory)
    service.startup()
    assert service.ready
    model = factory.models[0]
    # Exactly one warm-up inference, on CPU, at the configured size.
    assert len(model.predict_calls) == 1
    call = model.predict_calls[0]
    assert call["kwargs"]["device"] == "cpu"
    assert call["kwargs"]["imgsz"] == 640
    assert call["source"].shape == (640, 640, 3)


def test_startup_failure_keeps_service_not_ready():
    service = InferenceService(Settings(), factory=FakeModelFactory(fail_load=True))
    service.startup()
    assert not service.ready
    assert service.load_error


def test_startup_is_idempotent():
    factory = FakeModelFactory(behavior=behavior_empty)
    service = InferenceService(Settings(), factory=factory)
    service.startup()
    service.startup()
    assert len(factory.models) == 1


def test_boxes_clamped_to_image_bounds():
    factory = FakeModelFactory(
        behavior=behavior_boxes(
            boxes=[[-10.0, -20.0, 300.0, 999.0]], conf=[0.8], cls=[0]
        )
    )
    service = InferenceService(Settings(), factory=factory)
    service.startup()
    result = service.predict(_image(100, 80))
    assert result.detections[0].box == [0.0, 0.0, 100.0, 80.0]


def test_degenerate_box_dropped():
    factory = FakeModelFactory(
        behavior=behavior_boxes(boxes=[[10.0, 10.0, 10.0, 50.0]], conf=[0.8], cls=[0])
    )
    service = InferenceService(Settings(), factory=factory)
    service.startup()
    result = service.predict(_image())
    assert result.detections == []


def test_max_detections_bounded_and_sorted_by_confidence():
    factory = FakeModelFactory(
        behavior=behavior_boxes(
            boxes=[[1.0, 1.0, 9.0, 9.0]] * 5,
            conf=[0.1, 0.9, 0.5, 0.7, 0.3],
            cls=[0, 0, 0, 0, 0],
        )
    )
    service = InferenceService(Settings(yolo_max_detections=2), factory=factory)
    service.startup()
    result = service.predict(_image())
    assert [d.confidence for d in result.detections] == [0.9, 0.7]


def test_non_finite_confidence_dropped():
    factory = FakeModelFactory(
        behavior=behavior_boxes(
            boxes=[[1.0, 1.0, 9.0, 9.0], [5.0, 5.0, 9.0, 9.0]],
            conf=[float("nan"), 0.6],
            cls=[0, 0],
        )
    )
    service = InferenceService(Settings(), factory=factory)
    service.startup()
    result = service.predict(_image())
    assert len(result.detections) == 1
    assert result.detections[0].confidence == 0.6


def test_unknown_class_id_gets_stable_fallback_name():
    factory = FakeModelFactory(
        behavior=behavior_boxes(boxes=[[1.0, 1.0, 9.0, 9.0]], conf=[0.4], cls=[99])
    )
    service = InferenceService(Settings(), factory=factory)
    service.startup()
    result = service.predict(_image())
    assert result.detections[0].class_name == "class_99"
    assert result.detections[0].class_id == 99


def test_inference_timing_is_reported():
    factory = FakeModelFactory(behavior=behavior_empty)
    service = InferenceService(Settings(), factory=factory)
    service.startup()
    result = service.predict(_image())
    assert result.inference_ms >= 0.0


def test_overload_raises_when_slots_exhausted():
    gate = threading.Event()

    def blocking(_model, source):
        gate.wait(timeout=5)
        height, width = source.shape[:2]
        return behavior_empty(_model, source)

    factory = FakeModelFactory(behavior=blocking)
    service = InferenceService(Settings(inference_concurrency=1), factory=factory)
    service.startup()

    thread = threading.Thread(target=lambda: service.predict(_image()))
    thread.start()
    time.sleep(0.2)  # let the first prediction take the only slot
    with pytest.raises(InferenceOverloaded):
        service.predict(_image())
    gate.set()
    thread.join(timeout=5)


def test_numpy_arrays_do_not_leak_into_result():
    factory = FakeModelFactory(
        behavior=behavior_boxes(boxes=[[1.0, 1.0, 9.0, 9.0]], conf=[0.4], cls=[0])
    )
    service = InferenceService(Settings(), factory=factory)
    service.startup()
    result = service.predict(_image())
    det = result.detections[0]
    assert isinstance(det.box, list)
    for value in det.box:
        assert isinstance(value, float)
    assert isinstance(det.confidence, float)
