# delavnica — CPU-Only YOLO Object Detection Web Service

> Pretrained object detection that runs **entirely on a multicore CPU**, served to a browser demo
> page and to a plain JSON HTTP API. No GPU, no CUDA, no training — just the official
> [`ultralytics`](https://docs.ultralytics.com/) package, FastAPI, and a vanilla HTML/CSS/JavaScript
> client.
>
> (*delavnica* — Slovenian for “workshop”. This is a demonstration workshop that is expected to
> become a reusable inference endpoint.)

---

## Current status

| Item | Status |
|---|---|
| Runtime + CPU model proof (venv, `yolo26n.pt`, CPU inference) | **Implemented & verified** (WO-001) |
| Backend HTTP contract (`/health`, `/detect`, validation, controlled errors) | **Implemented & verified** (WO-001) |
| Browser image upload with aligned boxes/labels/timing | **Implemented & verified** (WO-001) |
| Browser webcam Start/Stop with backpressure | **Not implemented** (future work order) |
| Demo operability (one-command start, runbook, timings, recovery) | **Implemented** (WO-001, see [`docs/demo-runbook.md`](docs/demo-runbook.md)) |
| Video-file input, LAN access, profiling/exports | **Deferred** (future work orders) |
| First live demo | Targeted **2026-09-22** |
| Deliverable class | **Demo / prototype** — explicitly *not* a hardened public production service |

Verified on 2026-09-21 on: Ubuntu 26.04.1 (WSL2, kernel 6.18.33), AMD Ryzen AI 7 350
(8 cores / 16 threads), 15 GiB RAM, CPython 3.14.4, ultralytics 8.4.157, torch 2.14.0+cpu.

## What is this project?

A single, small, locally hosted inference service that demonstrates general-purpose object
detection without any GPU:

- A **pretrained COCO object detector** from the official `ultralytics` framework
  (`yolo26n.pt`, nano — confirmed loadable and runnable on the installed package, never assumed).
- A **FastAPI/Uvicorn** HTTP service that loads the model **once at startup** (plus one warm-up
  inference) and never per request.
- A **plain HTML5/JS browser page**: upload a still JPEG/PNG and see annotated boxes, labels,
  confidence percentages and measured server timing drawn over the image.
- A **stable JSON API** so any other client (curl, Python, future machine-vision tooling) can
  consume the same detections.

The intended machine has no usable GPU, and that is a *design constraint, not a workaround*:
explicit `device="cpu"` inference, no CUDA dependencies, no automatic device selection.

### Explicit non-goals (first demo)

Custom datasets, training or fine-tuning; segmentation / pose / tracking / face recognition;
**webcam capture and video-file input (deferred, not yet implemented)**; video transcoding or
storage; user accounts or authentication; cloud hosting; production hardening; WebSockets / RTSP /
message brokers / databases; GPU acceleration of any kind.

## Architecture

```text
Browser (HTML5/CSS/JavaScript)
  |  image file  ->  POST /detect  multipart/form-data, field: file
  v
FastAPI / Uvicorn (Ubuntu WSL2 or native Linux, single worker)
  |  bounded upload size (8 MiB streaming limit), type and decode validation (24 MP limit)
  |  EXIF-aware orientation + RGB conversion (Pillow)
  |  bounded inference concurrency (semaphore, default 1; excess -> 429)
  |  inference runs in a worker thread (event loop never blocked)
  v
Ultralytics YOLO pretrained COCO detector
  |  device="cpu" (explicit, every prediction path); imgsz=640; conf=0.25
  v
CPU inference and Ultralytics result decoding
  |  original-image pixel boxes, classes, confidences, durations
  v
JSON response -> browser canvas overlay or third-party HTTP client

GET /health -> readiness / model / device / versions
GET /       -> static browser demo
GET /docs   -> FastAPI-generated interactive API documentation
```

**Repository layout:**

```text
.
├── AGENTS.md                  # Project constitution (repository law for agents)
├── README.md                  # This file
├── .env.example               # Safe documented configuration defaults
├── requirements.txt           # Tested runtime dependencies (CPU torch pins)
├── requirements-dev.txt       # Dev/test tooling (pytest, httpx, ruff)
├── pyproject.toml             # pytest markers + ruff config only
├── backend/
│   ├── main.py                # App factory/startup, routes, static files
│   ├── config.py              # Validated environment settings (rejects non-CPU)
│   ├── inference.py           # Model lifetime, warm-up, CPU prediction, semaphore
│   ├── schemas.py             # Stable API response contracts
│   └── image_io.py            # Upload size limits, decode, EXIF, RGB normalization
├── frontend/
│   ├── index.html
│   ├── app.js
│   └── styles.css
├── tests/
│   ├── conftest.py            # Deterministic fake-model fixtures
│   ├── test_api.py            # HTTP contract tests (mocked detector)
│   ├── test_image_io.py       # Untrusted-input decoding/limits unit tests
│   ├── test_inference_contract.py  # Config validation + service contract
│   └── test_live_model.py     # Real-checkpoint HTTP tests (marker: live)
└── docs/
    ├── demo-runbook.md        # Exact live-demonstration steps + recovery table
    └── adr/                   # Approved architecture decisions only
```

## Getting started (tested commands)

Prerequisites: Ubuntu/WSL2 (or native Linux) with CPython 3.10–3.14 (`python3-venv` installed).
First run needs Internet once, for the ~5.3 MB checkpoint download.

```bash
git clone https://github.com/david-bernes/delavnica
cd delavnica
python3 -m venv .venv
source .venv/bin/activate

# 1) CPU-only PyTorch FIRST, from the official PyTorch CPU wheel index
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 2) The rest (pins are satisfied by the +cpu builds from step 1)
python -m pip install -r requirements.txt
```

> **Why step 1 separately?** A plain `pip install torch` on Linux pulls the CUDA-enabled wheel
> (~2 GB + NVIDIA libraries). Step 1 installs `torch==2.14.0+cpu` / `torchvision==0.29.0+cpu`;
> the pins in `requirements.txt` remain satisfied by those builds (PEP 440 local-version rules),
> so pip will not swap in CUDA wheels.

Start the service (one command; reads `HOST`/`PORT`/model settings from the environment or `.env`):

```bash
python -m backend.main
```

Expected startup (measured ≈ 1 s after import on the reference machine):

```text
INFO backend.inference: Loading model yolo26n.pt (device=cpu, imgsz=640, conf=0.25)
INFO backend.inference: Model ready in 1064 ms (torch threads: 8)
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
```

- Browser demo: **http://127.0.0.1:8000/**
- Interactive API docs: **http://127.0.0.1:8000/docs**
- Stop: `Ctrl+C` (single worker, no state to clean up)

Alternative launch (equivalent): `uvicorn backend.main:app --host 127.0.0.1 --port 8000`.

## HTTP API

`application/json` for detection results and controlled API errors; the browser is served as HTML.

### `GET /health` — readiness without inference

`200` when the model is actually loaded and warmed up; `503` otherwise (never fake readiness).
Actual response from the verified run:

```json
{
  "status": "ready",
  "model": "yolo26n.pt",
  "device": "cpu",
  "ultralytics_version": "8.4.157",
  "torch_version": "2.14.0+cpu",
  "image_size": 640,
  "confidence": 0.25
}
```

### `POST /detect` — detect objects in one still image

Input: `multipart/form-data`, required field **`file`**, one still image (JPEG or PNG).
Video files and unsupported formats are explicitly rejected; filename extensions and
`Content-Type` alone are never trusted — the **decoded** format is validated.

Example:

```bash
curl -s -F "file=@photo.jpg" http://127.0.0.1:8000/detect | python3 -m json.tool
```

Actual response (verified, 810×1080 photo, values abridged):

```json
{
  "model": "yolo26n.pt",
  "device": "cpu",
  "image": { "width": 810, "height": 1080 },
  "detections": [
    {
      "class_id": 5,
      "class_name": "bus",
      "confidence": 0.8832,
      "box": [0.0, 230.69, 802.36, 750.91]
    },
    {
      "class_id": 0,
      "class_name": "person",
      "confidence": 0.8771,
      "box": [49.17, 397.47, 237.7, 902.6]
    }
  ],
  "timing_ms": { "total": 27.5, "inference": 23.5 }
}
```

**Contract rules**

1. `class_id` is a non-negative integer from the model's label map; `class_name` is its exact
   corresponding name (never guessed from the ID).
2. `confidence` is a finite number in `[0.0, 1.0]` (the UI displays it as `87.7%`).
3. **Coordinate convention:** `box` is `[x1, y1, x2, y2]` in *floating-point pixels of the
   decoded, EXIF-orientation-corrected input image* (i.e. `image.width` × `image.height`),
   top-left origin, X right, Y down. Ultralytics letterbox/tensor coordinates are **never**
   returned; boxes are clamped into the image and degenerate boxes dropped.
4. An image with nothing detected returns `"detections": []` with HTTP `200` — empty is a valid
   result, not an error.
5. The response carries boxes and metadata, **not** base64 annotated images; the browser draws
   overlays locally.
6. Response size is bounded (`YOLO_MAX_DETECTIONS`, default 100, most-confident first) and
   strictly JSON-serializable — no NumPy/PyTorch types leak to clients.
7. `timing_ms.inference` measures the Ultralytics `predict()` call only; `timing_ms.total` is
   server-side processing (upload read + decode + inference + serialization). HTTP round-trip
   time is always reported separately, never conflated with these.

### Controlled errors

Application errors share one shape:

```json
{ "error": { "code": "invalid_image", "message": "The uploaded file is not a supported image." } }
```

| Status | Code(s) | Example cause (all verified with curl) |
|---|---|---|
| `400` | `missing_file`, `invalid_image` | No `file` field; text/corrupt/truncated image |
| `413` | `image_too_large` | Upload > 8 MiB (streaming limit) or decoded > 24 MP |
| `415` | `unsupported_media_type` | GIF/WEBP/TIFF/BMP/… or video `Content-Type` |
| `429` | `inference_overloaded` | Concurrency limit (default 1) reached; `Retry-After: 1` |
| `503` | `model_not_loaded` | Model load failed / service not ready — no fabricated detections |
| `500` | `internal_error` | Unexpected failure — sanitized message, full detail only in server logs |

No stack traces, filesystem paths, exception details or image contents ever reach the client.

**Input limits (enforced, configurable):** max compressed upload **8 MiB** (`MAX_UPLOAD_BYTES`),
max decoded **24 MP** (`MAX_IMAGE_PIXELS`, checked from the image header before decode),
animated/multi-frame images rejected, decompression-bomb protection, EXIF orientation applied
before dimensions and inference, alpha deterministically composited over white.

## Browser demo (implemented in WO-001)

Open http://127.0.0.1:8000/ :

- Service/model readiness indicator (from `/health`, with bounded retry while the model loads).
- Select a JPEG/PNG → preview with correct aspect ratio and orientation → automatic detection.
- Annotated overlay (boxes, class labels with percentages), detection count, and server timing
  (`inference` and `total` ms) — drawn on a canvas that tracks displayed size, window resizing
  and device pixel ratio.
- A **Detect objects** button re-runs on the same image; selecting a new image immediately clears
  old boxes and aborts any in-flight request (stale responses are discarded — annotations from
  image A can never land on image B).
- Meaningful, recoverable error messages (bad file type, size limits, service errors, 429
  auto-retry) without a page refresh.
- Same-origin relative URLs only; no external CDNs; works offline once dependencies + weights exist.

**Not implemented in WO-001 (do not present as available):** webcam capture, video-file input,
frame sampling/FPS display. These are explicitly deferred to later work orders.

## Configuration

Validated at startup with actionable, secret-safe failures (see `.env.example`):

| Setting | Default | Constraint |
|---|---|---|
| `YOLO_MODEL` | `yolo26n.pt` | Approved pretrained name (resolved into `YOLO_WEIGHTS_DIR`) or absolute local path; never set by HTTP clients |
| `YOLO_DEVICE` | `cpu` | **Any non-CPU value is rejected at startup** (INV-02) |
| `YOLO_IMAGE_SIZE` | `640` | Integer 16–1280 |
| `YOLO_CONFIDENCE` | `0.25` | Number in `[0, 1]` |
| `YOLO_MAX_DETECTIONS` | `100` | Integer 1–1000 |
| `MAX_UPLOAD_BYTES` | `8388608` (8 MiB) | Integer ≥ 1 KiB |
| `MAX_IMAGE_PIXELS` | `24000000` (24 MP) | Integer ≥ 1 |
| `INFERENCE_CONCURRENCY` | `1` | Integer 1–16; raise only with verified thread-safety/performance |
| `TORCH_NUM_THREADS` | (unset → PyTorch auto, measured 8) | Integer ≥ 1; avoid oversubscription |
| `HOST` | `127.0.0.1` | Loopback by default (INV-07); LAN binding requires explicit approval |
| `PORT` | `8000` | Non-privileged port |
| `YOLO_WEIGHTS_DIR` | `~/.cache/ultralytics/weights` | Local weights directory for offline demos |

## CPU-only guarantee

- `device="cpu"` is set explicitly on **every** Ultralytics prediction path (warm-up included);
  automatic device selection is never relied upon.
- No CUDA/TensorRT/ROCm/DirectML dependency or code path exists; the environment installs the
  `+cpu` PyTorch wheels; `YOLO_DEVICE` values other than `cpu` are rejected at startup.
- `/health` reports the runtime device; mocked tests assert `device == "cpu"` on every
  (mocked) prediction call, and the live-model test runs a real checkpoint through the real
  HTTP endpoint on CPU.
- Compliance is judged by **actual CPU operation**, not wheel names.

## Measured performance (2026-09-21, reference machine)

| Metric | Value |
|---|---|
| CPU | AMD Ryzen AI 7 350 (Zen 5, 8 cores / 16 threads), WSL2 on Ubuntu 26.04.1 |
| RAM | 15 GiB |
| Python / torch / ultralytics | 3.14.4 / 2.14.0+cpu / 8.4.157 |
| Checkpoint | `yolo26n.pt` (official Ultralytics asset, 5.3 MB) |
| Image | 810×1080 JPEG, `imgsz=640`, `conf=0.25`, torch threads = 8 (auto) |
| Warm-up | 1 inference at startup + 1 warm-up request before timing |
| Samples | 20 timed HTTP requests |

| Measurement (server-side) | mean | median | p95 | min–max |
|---|---|---|---|---|
| Inference (`predict()` only) | 22.5 ms | 22.4 ms | 25.8 ms | 18.2–26.6 ms |
| Total server processing | 26.4 ms | 26.4 ms | 29.6 ms | 22.3–30.7 ms |
| HTTP round-trip (curl-class client) | 28.3 ms | 28.3 ms | 31.4 ms | 24.0–32.9 ms |

**≈ 35 FPS end-to-end over HTTP** on the reference machine — comfortably above the 2–5 FPS
aspiration, though that aspiration is a target, not a guarantee: results will be re-measured on
the actual demo hardware. These numbers measure the thing users experience (HTTP request/response),
not marketing benchmarks. Practical next-step tuning (only under a separate work order):
`imgsz` vs accuracy, `TORCH_NUM_THREADS`, nano vs small checkpoint.

## Testing & verification

```bash
source .venv/bin/activate
python -m pytest -q                       # 64 mocked tests (fast, offline, deterministic)
python -m pytest -m live -q               # 3 real-checkpoint HTTP tests (needs weights)
YOLO_TEST_IMAGE=/path/to/photo.jpg python -m pytest -m live -q -s   # + positive-detection check
python -m ruff check backend tests        # linter
python -m compileall backend tests        # syntax gate
python -m pip check                       # dependency consistency
```

Verified on 2026-09-21: **67 passed** (64 mocked + 3 live), `ruff` clean, `pip check` clean.

- **Mocked tests** prove the HTTP contract: readiness semantics (ready **and** not-ready),
  JSON structure, exact coordinate/class mapping, zero-detection success, all error paths
  (400/413/415/429/503/500), model-initialized-once, explicit `device="cpu"` on every call,
  sanitized internal errors, static asset serving.
- **Live-model tests** (`-m live`) prove the real checkpoint works through the real HTTP stack:
  readiness with real versions, a 200 on a synthetic image (empty detections = valid success),
  and — when `YOLO_TEST_IMAGE` points at a legally usable photo — bounded, in-image detection
  boxes with actual timings.
- A mocked detector is **not** proof that the real model works, and a Python inference call is
  **not** proof that the HTTP service works — both layers are tested separately, and the live
  curl session in the agent report covers the uninstrumented HTTP path as well.
- No private or unlicensed photographs are committed; the demo works with any user-selected
  local image.

## Known limitations

- **No webcam / video-file input yet** (deferred work orders); the demo is image-upload only.
- Loopback-only by default; no LAN exposure (requires explicit approval + real remote test).
- Unauthenticated single-process demo; **not** suitable for public Internet deployment.
- COCO general-object detection only — no industrial inspection precision, no counting
  guarantees, no absence-of-object guarantees.
- First start needs Internet for the one-time checkpoint download (offline path documented).

## Troubleshooting

See the full recovery table in [`docs/demo-runbook.md`](docs/demo-runbook.md) §6. Quick hits:

- `/health` returns 503 → model not loaded; check the server log (usually a failed weights
  download; pre-stage weights for offline demos, runbook §5).
- `413 image_too_large` → image exceeds 8 MiB compressed or 24 MP decoded; resize/compress.
- `429 inference_overloaded` → another request is mid-inference (concurrency 1); wait ~1 s.
- Port busy → `PORT=8010 python -m backend.main`.
- `torchvision::nms does not exist` → torchvision/torch wheel mismatch; reinstall torchvision
  from the CPU index (runbook §6).

## Development model (OAP)

This repository is governed by the Orchestrated Agentic Programming (OAP) model defined in
[`AGENTS.md`](AGENTS.md) — read it before contributing:

| Role | Owns |
|---|---|
| **Human lead** | Goals, risk acceptance, scope approval, PR merge, demo/release decision |
| **Strategic AI** | Discovery, architecture, constitution, precise work orders, evidence review |
| **Execution agent** (e.g. Codex CLI) | Bounded implementation in the approved workspace, tests, commits, PRs, evidence reports |

Rules that bind every contribution: one bounded branch per work order from an agreed base
(`main`), no direct commits to `main`, no self-merge, tests and docs updated in the same change
as behavior, and every claim of performance/CPU operation backed by actual measurement.
Significant approved decisions get a short ADR in `docs/adr/`.

## License

- **This repository:** [Apache-2.0](LICENSE).
- **Third-party licensing notice (important):** the demo depends on the `ultralytics` package and
  its published pretrained weights. Ultralytics operates an **AGPL-3.0 / Enterprise** licensing
  model; running a local demo does **not** automatically resolve business licensing questions.
  Company deployment or distribution requires human/legal review of the upstream terms and of
  this project's own Apache-2.0 code. Nothing in this README is a statement that commercial use
  is unrestricted.

## References

- [Ultralytics available models](https://docs.ultralytics.com/models/) ·
  [YOLO26 detection models](https://docs.ultralytics.com/models/yolo26/) ·
  [Prediction / device docs](https://docs.ultralytics.com/modes/predict/) ·
  [Quickstart](https://docs.ultralytics.com/quickstart/)
- [FastAPI documentation](https://fastapi.tiangolo.com/)
- [Project constitution](AGENTS.md) — the governing document for this repository
