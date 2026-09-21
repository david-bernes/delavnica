"""Stable JSON response contracts for the HTTP API (see README / AGENTS.md)."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ErrorDetail(BaseModel):
    code: str
    message: str


class ErrorResponse(BaseModel):
    error: ErrorDetail


class HealthReady(BaseModel):
    status: str = "ready"
    model: str
    device: str = "cpu"
    ultralytics_version: str
    torch_version: str | None = None
    image_size: int
    confidence: float


class HealthNotReady(BaseModel):
    status: str = "not_ready"
    device: str = "cpu"
    error: ErrorDetail


class ImageInfo(BaseModel):
    width: int = Field(ge=1)
    height: int = Field(ge=1)


class Detection(BaseModel):
    class_id: int = Field(ge=0)
    class_name: str
    confidence: float = Field(ge=0.0, le=1.0)
    #: [x1, y1, x2, y2] in pixels of the decoded, orientation-corrected image.
    box: list[float] = Field(min_length=4, max_length=4)


class TimingMs(BaseModel):
    #: Server-side processing time: upload read + decode + inference + serialization.
    total: float = Field(ge=0.0)
    #: Ultralytics ``predict()`` call duration only (documented boundary).
    inference: float = Field(ge=0.0)


class DetectResponse(BaseModel):
    model: str
    device: str = "cpu"
    image: ImageInfo
    detections: list[Detection]
    timing_ms: TimingMs
