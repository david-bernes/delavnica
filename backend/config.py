"""Validated environment configuration for the detection service.

All settings are read from environment variables (optionally via a local
``.env`` file). Invalid values fail fast at startup with an actionable,
secret-safe message.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError, field_validator

DEFAULT_WEIGHTS_DIR = Path.home() / ".cache" / "ultralytics" / "weights"


class ConfigError(ValueError):
    """Raised when the environment configuration is invalid."""


class Settings(BaseModel):
    """Runtime settings for the service (all values validated)."""

    yolo_model: str = Field(default="yolo26n.pt")
    yolo_device: str = Field(default="cpu")
    yolo_image_size: int = Field(default=640, ge=16, le=1280)
    yolo_confidence: float = Field(default=0.25, ge=0.0, le=1.0)
    yolo_max_detections: int = Field(default=100, ge=1, le=1000)
    max_upload_bytes: int = Field(default=8 * 1024 * 1024, ge=1024)
    max_image_pixels: int = Field(default=24_000_000, ge=1)
    inference_concurrency: int = Field(default=1, ge=1, le=16)
    torch_num_threads: int | None = Field(default=None, ge=1, le=256)
    host: str = Field(default="127.0.0.1")
    port: int = Field(default=8000, ge=1, le=65535)

    @field_validator("yolo_device")
    @classmethod
    def _cpu_only(cls, value: str) -> str:
        if value.strip().lower() != "cpu":
            raise ValueError(
                f"YOLO_DEVICE must be 'cpu' (got {value.strip()!r}); "
                "this service is CPU-only by design (INV-02)."
            )
        return "cpu"

    @field_validator("yolo_model")
    @classmethod
    def _model_name(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("YOLO_MODEL must not be empty.")
        if any(ch.isspace() for ch in value):
            raise ValueError("YOLO_MODEL must not contain whitespace.")
        return value

    @field_validator("host")
    @classmethod
    def _host(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("HOST must not be empty.")
        return value

    def model_path(self) -> Path:
        """Resolve the checkpoint to a local file path.

        Bare model names (e.g. ``yolo26n.pt``) are resolved into the weights
        directory (``YOLO_WEIGHTS_DIR`` or ``~/.cache/ultralytics/weights``);
        ultralytics will download the official checkpoint there on first run.
        Absolute paths are used as-is. HTTP clients can never influence this.
        """
        raw = Path(self.yolo_model)
        if raw.is_absolute():
            return raw
        weights_dir = Path(
            os.environ.get("YOLO_WEIGHTS_DIR", str(DEFAULT_WEIGHTS_DIR))
        ).expanduser()
        weights_dir.mkdir(parents=True, exist_ok=True)
        return weights_dir / raw.name


_ENV_MAP = {
    "yolo_model": "YOLO_MODEL",
    "yolo_device": "YOLO_DEVICE",
    "yolo_image_size": "YOLO_IMAGE_SIZE",
    "yolo_confidence": "YOLO_CONFIDENCE",
    "yolo_max_detections": "YOLO_MAX_DETECTIONS",
    "max_upload_bytes": "MAX_UPLOAD_BYTES",
    "max_image_pixels": "MAX_IMAGE_PIXELS",
    "inference_concurrency": "INFERENCE_CONCURRENCY",
    "torch_num_threads": "TORCH_NUM_THREADS",
    "host": "HOST",
    "port": "PORT",
}


def _format_validation_error(exc: ValidationError) -> str:
    parts = []
    for err in exc.errors():
        field = err.get("loc")
        field_name = _ENV_MAP.get(str(field[-1]) if field else "", str(field))
        msg = err.get("msg", "invalid value")
        parts.append(f"{field_name}: {msg}")
    return "Invalid configuration: " + "; ".join(parts)


def get_settings(environ: Mapping[str, str] | None = None) -> Settings:
    """Build validated :class:`Settings` from the environment.

    ``environ`` may be supplied explicitly (tests); otherwise ``.env`` (if
    present) is loaded without overriding real environment variables, and
    ``os.environ`` is used.
    """
    if environ is None:
        load_dotenv(override=False)
        environ = os.environ
    values: dict[str, str] = {}
    for field_name, env_name in _ENV_MAP.items():
        raw = environ.get(env_name)
        if raw is not None and raw != "":
            values[field_name] = raw
    try:
        return Settings(**values)
    except ValidationError as exc:
        raise ConfigError(_format_validation_error(exc)) from exc
