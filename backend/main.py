"""FastAPI application: /health, /detect and the static HTML5 demo.

Single-process design (one Uvicorn worker, one model instance). Inference
runs in a worker thread pool so the event loop is never blocked by CPU
prediction, and a semaphore bounds the in-flight inference backlog.
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

import torch
import ultralytics
from fastapi import FastAPI, File, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .config import Settings, get_settings
from .image_io import (
    DecodedImageTooLarge,
    InvalidImage,
    UnsupportedFormat,
    UploadTooLarge,
    decode_image,
    read_upload,
)
from .inference import (
    DEVICE,
    InferenceOverloaded,
    InferenceService,
    ModelNotReady,
)
from .schemas import (
    DetectResponse,
    HealthNotReady,
    HealthReady,
    ImageInfo,
    TimingMs,
)

logger = logging.getLogger("delavnica")

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"

NOT_READY_MESSAGE = (
    "The detection model is not available. Check the server logs for details."
)
INTERNAL_MESSAGE = "An unexpected server error occurred."


def _error(status_code: int, code: str, message: str, **headers) -> JSONResponse:
    response = JSONResponse(
        status_code=status_code,
        content={"error": {"code": code, "message": message}},
    )
    for key, value in headers.items():
        response.headers[key] = value
    return response


def _first_validation_message(exc: RequestValidationError) -> str:
    errors = exc.errors()
    if not errors:
        return "Malformed request."
    first = errors[0]
    loc = ".".join(str(part) for part in first.get("loc", ()))
    msg = str(first.get("msg", "invalid value"))
    return f"Malformed request: {loc}: {msg}" if loc else f"Malformed request: {msg}"


def create_app(settings: Settings | None = None, make_service=None) -> FastAPI:
    """Application factory.

    ``make_service`` may be injected in tests to swap in a mocked model
    factory; production uses the real Ultralytics service.
    """
    if settings is None:
        settings = get_settings()

    service = (
        make_service(settings) if make_service is not None else InferenceService(settings)
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await run_in_threadpool(service.startup)
        yield

    app = FastAPI(
        title="delavnica — CPU YOLO object detection",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.state.service = service

    # -- error normalization ----------------------------------------------

    @app.exception_handler(RequestValidationError)
    async def _validation_handler(request: Request, exc: RequestValidationError):
        logger.info("Rejected %s %s: validation error", request.method, request.url.path)
        return _error(422, "invalid_request", _first_validation_message(exc))

    @app.exception_handler(Exception)
    async def _unhandled_handler(request: Request, exc: Exception):
        logger.exception("Unhandled error on %s %s", request.method, request.url.path)
        return _error(500, "internal_error", INTERNAL_MESSAGE)

    # -- endpoints ----------------------------------------------------------

    @app.get("/health", response_model=None)
    async def health() -> JSONResponse | HealthReady:
        """Readiness and diagnostics; no inference is performed."""
        if service.ready:
            return HealthReady(
                model=settings.yolo_model,
                device=DEVICE,
                ultralytics_version=ultralytics.__version__,
                torch_version=torch.__version__,
                image_size=settings.yolo_image_size,
                confidence=settings.yolo_confidence,
            )
        body = HealthNotReady(
            error={"code": "model_not_loaded", "message": NOT_READY_MESSAGE}
        ).model_dump()
        return JSONResponse(status_code=503, content=body)

    @app.post("/detect", response_model=DetectResponse, response_model_exclude_none=True)
    async def detect(
        request: Request,
        file: UploadFile | None = File(default=None, description="JPEG or PNG still image"),  # noqa: B008
    ):
        """Run CPU object detection on an uploaded still image."""
        started = time.perf_counter()
        if file is None:
            logger.info("detect rejected: missing file field")
            return _error(400, "missing_file", "The 'file' form field is required.")

        media_type = (file.content_type or "").lower()
        if media_type.startswith("video/"):
            return _error(
                415,
                "unsupported_media_type",
                "Video files are not supported by /detect; upload a still image (JPEG or PNG).",
            )

        # Early rejection when Content-Length is present; the bounded stream
        # read below is the authoritative enforcement.
        content_length = request.headers.get("content-length")
        if (
            content_length
            and content_length.isdigit()
            and int(content_length) > settings.max_upload_bytes + 1024
        ):
            logger.info("detect rejected: content-length over upload limit")
            return _error(
                413,
                "image_too_large",
                f"The upload exceeds the {settings.max_upload_bytes} byte limit.",
            )

        try:
            data = await read_upload(file, settings.max_upload_bytes)
        except UploadTooLarge:
            logger.info("detect rejected: upload exceeded byte limit")
            return _error(
                413,
                "image_too_large",
                f"The upload exceeds the {settings.max_upload_bytes} byte limit.",
            )

        try:
            image = decode_image(data, settings.max_image_pixels)
        except UnsupportedFormat as exc:
            logger.info("detect rejected: unsupported format (%s)", exc)
            return _error(415, "unsupported_media_type", str(exc))
        except DecodedImageTooLarge as exc:
            logger.info("detect rejected: decoded image too large")
            return _error(413, "image_too_large", str(exc))
        except InvalidImage as exc:
            logger.info("detect rejected: invalid image")
            return _error(400, "invalid_image", str(exc))

        if not service.ready:
            return _error(503, "model_not_loaded", NOT_READY_MESSAGE)

        try:
            result = await run_in_threadpool(service.predict, image)
        except ModelNotReady:
            return _error(503, "model_not_loaded", NOT_READY_MESSAGE)
        except InferenceOverloaded:
            logger.info("detect rejected: inference backlog full (429)")
            return _error(
                429,
                "inference_overloaded",
                "The service is at its inference limit; retry shortly.",
                **{"Retry-After": "1"},
            )
        except Exception:
            logger.exception("Inference failed for this request")
            return _error(500, "internal_error", INTERNAL_MESSAGE)

        total_ms = (time.perf_counter() - started) * 1000
        logger.info(
            "detect ok: %dx%d, %d detections, inference=%.1f ms, total=%.1f ms",
            result.width,
            result.height,
            len(result.detections),
            result.inference_ms,
            total_ms,
        )
        return DetectResponse(
            model=settings.yolo_model,
            device=DEVICE,
            image=ImageInfo(width=result.width, height=result.height),
            detections=result.detections,
            timing_ms=TimingMs(
                total=round(total_ms, 1), inference=round(result.inference_ms, 1)
            ),
        )

    @app.get("/", include_in_schema=False)
    async def index() -> FileResponse:
        """Serve the HTML5 demo page."""
        return FileResponse(FRONTEND_DIR / "index.html")

    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

    return app


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    # Keep third-party noise down.
    logging.getLogger("multipart.multipart").setLevel(logging.WARNING)


app = create_app()


if __name__ == "__main__":
    _configure_logging()
    import uvicorn

    # Single worker: one model instance, one coherent CPU/FPS story (AGENTS.md 4.3).
    uvicorn.run(
        app,
        host=app.state.settings.host,
        port=app.state.settings.port,
        workers=1,
    )
