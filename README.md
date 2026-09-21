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
| Repository state | **Greenfield** — project governance and documentation only; no application code yet |
| Application code | Not implemented (delivered by upcoming P0 work orders) |
| First live demo | Targeted **2026-09-22** |
| Deliverable class | **Demo / prototype** — explicitly *not* a hardened public production service |

This README documents the project's purpose, the agreed architecture, the stable API contract the
implementation must satisfy, and the project's non-negotiable constraints. Anything marked
*planned* is a design commitment from the project constitution ([`AGENTS.md`](AGENTS.md)) — **not**
verified functionality. When the P0 implementation lands, the planned sections are replaced with
**actual tested** commands and measured numbers.

## What is this project?

A single, small, locally hosted inference service that demonstrates general-purpose object
detection without any GPU:

- A **pretrained COCO object detector** from the official `ultralytics` framework (initial
  candidate: `yolo26n.pt`, nano size — confirmed against the installed package version at
  implementation time, never assumed).
- A **FastAPI/Uvicorn** HTTP service that loads the model **once at startup** and never per
  request.
- A **plain HTML5/JS browser page** for interactive demos: upload a still image or take webcam
  snapshots, with annotated boxes, labels, confidence percentages and measured timing drawn as
  overlays.
- A **stable JSON API** so any other client (curl, Python, future machine-vision tooling) can
  consume the same detections.

The intended machine has no usable GPU, and that is a *design constraint, not a workaround*:
explicit `device="cpu"` inference, no CUDA dependencies, no automatic device selection.

### Explicit non-goals (first demo)

Custom datasets, training or fine-tuning; segmentation / pose / tracking / face recognition;
video transcoding or video storage; user accounts or authentication; cloud hosting; production
hardening; WebSockets / RTSP / message brokers / databases.

## Architecture

```text
Browser (HTML5/CSS/JavaScript)
  |  image file, webcam snapshot (future: sampled local video frame)
  |  POST /detect  multipart/form-data, field: file
  v
FastAPI / Uvicorn (Ubuntu WSL2 or native Linux)
  |  bounded upload size, type and decode validation
  |  EXIF-aware orientation + RGB conversion
  |  bounded inference concurrency
  v
Ultralytics YOLO pretrained COCO detector
  |  device="cpu"; configured image size / confidence threshold
  v
CPU inference and Ultralytics result decoding
  |  original-image pixel boxes, classes, confidences, durations
  v
JSON response -> browser canvas overlay or third-party HTTP client

GET /health -> readiness / model / device / versions
GET /       -> static browser demo
GET /docs   -> FastAPI-generated interactive API documentation
```

**Repository layout (target shape)** — created by the implementation work orders as needed:

```text
.
├── AGENTS.md                  # Project constitution (repository law for agents)
├── README.md                  # This file
├── .gitignore                 # Environments, caches, secrets, generated weights
├── .env.example               # Safe documented defaults; no secrets
├── requirements.txt           # Tested, reproducible dependencies
├── backend/
│   ├── main.py                # App factory/startup, routes, static files
│   ├── config.py              # Validated environment settings
│   ├── inference.py           # Ultralytics model lifetime, CPU prediction
│   ├── schemas.py             # Stable API response contracts
│   └── image_io.py            # Upload, decode, orientation, pixel limits
├── frontend/
│   ├── index.html
│   ├── app.js
│   └── styles.css
├── tests/
│   └── …                      # Unit/API tests (mocked) + opt-in live-model CPU test
└── docs/
    ├── demo-runbook.md        # Exact live-demonstration steps
    └── adr/                   # Approved architecture decisions only
```

## Technology stack

| Layer | Choice | Notes |
|---|---|---|
| Runtime | CPython (tested version documented at implementation) | `python3 -m venv .venv`; never system-site-packages installs |
| Backend | FastAPI + Uvicorn | One long-lived model; single Uvicorn worker initially |
| Inference | Official `ultralytics` + PyTorch **CPU** execution | `device="cpu"` explicit on every prediction path |
| Imaging | Pillow (decode, validation, EXIF orientation) | Bounded decode, RGB conversion |
| Frontend | Plain `index.html` + CSS + vanilla JS | No npm toolchain, no frontend framework |
| Testing | pytest + FastAPI test client | Deterministic mocked tests + explicitly marked live-model CPU test |
| Quality | ruff (single configured linter) | No speculative build-system complexity |
| Prohibited | CUDA, TensorRT, ROCm, DirectML, any GPU path | See invariants below |

## HTTP API — baseline contract

The contract below is the **baseline design** mandated by the constitution. JSON examples are
**illustrative** (values are examples, not measured output). Breaking changes require explicit
approval and tests.

### `GET /health` — readiness without inference

- `200` when the pretrained model is actually loaded and available.
- `503` when the model has not loaded (e.g. offline first run) — never fake readiness.
- Never exposes host paths, environment variables, tokens or network internals.

Illustrative `200` response:

```json
{
  "status": "ready",
  "model": "yolo26n.pt",
  "device": "cpu",
  "ultralytics_version": "<installed-version>"
}
```

### `POST /detect` — detect objects in one still image

Input: `multipart/form-data` with a required field **`file`** containing one still image
(initially JPEG and PNG). Video files are explicitly rejected here. No URL-fetching, no
filesystem paths, no client-controlled model selection.

Illustrative `200` response:

```json
{
  "model": "yolo26n.pt",
  "device": "cpu",
  "image": { "width": 1280, "height": 720 },
  "detections": [
    {
      "class_id": 0,
      "class_name": "person",
      "confidence": 0.927,
      "box": [101.2, 49.0, 384.6, 699.2]
    }
  ],
  "timing_ms": { "total": 123.4, "inference": 101.2 }
}
```

**Contract rules**

1. `class_id` is a non-negative integer from the model's label map; `class_name` is its exact
   corresponding name.
2. `confidence` is a finite number in `[0.0, 1.0]` (the UI may display it as `92.7%`).
3. **Coordinate convention:** `box` is `[x1, y1, x2, y2]` in *floating-point pixels of the
   decoded, EXIF-orientation-corrected input image* (i.e. `image.width` × `image.height`),
   top-left origin, X right, Y down. Internal letterbox/tensor coordinates are **never**
   returned. The browser scales overlays to displayed size.
4. An image with nothing detected returns `"detections": []` with HTTP `200` — empty is a valid
   result, not an error.
5. The response carries boxes and metadata, **not** base64 annotated images; the browser draws
   overlays locally.
6. Response size is bounded (configured max detections) and strictly JSON-serializable — no
   NumPy/PyTorch types leak to clients.
7. `timing_ms.total` is server-side processing duration; `timing_ms.inference` has a documented
   measurement boundary. Internal `speed` figures are never presented as HTTP round-trip time.

### Controlled errors

Application errors share one shape:

```json
{ "error": { "code": "invalid_image", "message": "The uploaded file is not a supported image." } }
```

| Status | Example cause | Effect |
|---|---|---|
| `400` / `422` | Missing/corrupt image, malformed form | No inference, informative message |
| `413` | Upload or decoded image beyond policy limits | Rejected, never decoded unboundedly |
| `415` | Unsupported media type/format (extension alone is never trusted) | Rejected |
| `429` | Bounded inference concurrency exceeded | Retry allowed, optional `Retry-After` |
| `503` | Model unavailable / not ready | No stale or fabricated detections |
| `500` | Unexpected internal failure | Sanitized message; no stack traces or paths leak |

**Input limits (initial policy, configurable):** max upload **8 MiB**, max decoded **24 MP**,
EXIF orientation applied before dimensions/inference, animated/multipage and decompression-bomb
files rejected, alpha handled deterministically.

## Browser demo (planned behavior)

- Upload an image → preview with correct aspect ratio/orientation → live annotated overlay,
  detection count, labels with percentages, server duration.
- **Start/Stop Webcam** using `getUserMedia`: compressed JPEG snapshots at a bounded rate,
  **at most one in-flight `/detect` request per client**, stale responses discarded
  (sequence-tagged / `AbortController`), frame sampling instead of request pileup.
- Measured end-to-end frame rate and latency shown with correct labels — dropped frames are not
  hidden.
- Visible, recoverable error states for: camera permission denied, camera in use, insecure
  origin, backend not ready, network failure. **Webcam requires `localhost` or HTTPS** (browser
  secure-context rule); remote-HTTP clients may use image upload while webcam may not work —
  that distinction is stated, not papered over.

## Configuration (planned)

Validated at startup with actionable, secret-safe failures. Design targets from the
constitution:

| Setting | Initial default | Constraint |
|---|---|---|
| `YOLO_MODEL` | `yolo26n.pt` | Approved pretrained choice; never set by HTTP client |
| `YOLO_DEVICE` | `cpu` | **Any non-CPU value is rejected**, not auto-selected |
| `YOLO_IMAGE_SIZE` | `640` | Tested valid range |
| `YOLO_CONFIDENCE` | `0.25` | Number in `[0, 1]` |
| `YOLO_MAX_DETECTIONS` | `100` | Positive bounded integer |
| `MAX_UPLOAD_BYTES` | `8388608` (8 MiB) | Bounded upload |
| `MAX_IMAGE_PIXELS` | `24000000` (24 MP) | Bounded decode |
| `INFERENCE_CONCURRENCY` | `1` | Raised only with verified thread-safety/perf |
| `TORCH_NUM_THREADS` | measured/tunable | No CPU oversubscription; recorded |
| `HOST` | `127.0.0.1` | LAN binding requires explicit human approval |
| `PORT` | `8000` | Non-privileged, configurable |

## CPU-only guarantee

- `device="cpu"` is set explicitly on every Ultralytics prediction path; automatic device
  selection is never relied upon.
- No CUDA/TensorRT/ROCm/DirectML dependency or code path exists or may be added; `YOLO_DEVICE`
  values other than `cpu` are rejected at startup.
- `/health` reports the runtime device and tests assert it is exactly `cpu`.
- Compliance is judged by **actual CPU operation**, not by wheel names; the live-model
  integration test runs a real pretrained checkpoint through the real HTTP endpoint on CPU.

## Security & privacy posture

- **Loopback by default** (`127.0.0.1`). LAN or Internet exposure is a security decision
  requiring explicit human approval, route/firewall configuration and a real remote-client test.
- **No image storage**: uploaded bytes exist only for request processing. No debug dumps, no
  frame caches, no uploads of user images to any third-party service. (Model weight download
  from the official Ultralytics source is a dependency operation, not image exfiltration.)
- **Untrusted uploads fail closed**: bounded size before decode, real-decode validation,
  format allowlist, decompression-bomb protection; controlled 4xx/5xx errors, never stack
  traces, filesystem paths, or image contents in responses/logs.
- **No secrets** in code, logs, fixtures or commits; no authentication is implemented, so the
  service is not suitable for unauthenticated public deployment.
- Development runs in a dedicated, rebuildable WSL2/Linux workspace; host Windows security
  settings are never weakened to make the demo work.

## Performance policy

- **~2–5 FPS on the actual CPU is an aspiration, not a guarantee.** Nothing is claimed until
  measured.
- Reported metrics always include: CPU model, cores/threads, OS/WSL2, Python and package
  versions, checkpoint, `imgsz`, confidence, warm-up policy, sample count (≥20 where practical),
  mean/median/p95 latency, and separate decode/inference/HTTP/browser-FPS boundaries.
- Model-only inference time is never substituted for end-to-end HTTP or UI throughput.
- If the real CPU does not reach the aspiration, the honest measured number plus a bounded
  tuning proposal (nano vs small, `imgsz`, threads, frame sampling) is presented.

## Delivery slices & feature matrix

| Priority | Slice | Status |
|---|---|---|
| P0 | Runtime + CPU model proof (venv, pretrained weights, CPU inference) | NOT IMPLEMENTED |
| P0 | Backend HTTP contract (`/health`, `/detect`, validation, errors) | NOT IMPLEMENTED |
| P0 | Browser image upload with aligned overlays | NOT IMPLEMENTED |
| P0 | Browser webcam Start/Stop with backpressure | NOT IMPLEMENTED |
| P0 | Demo operability (one-command start, runbook, timings, recovery) | NOT IMPLEMENTED |
| P1 | Video-file input (browser-sampled frames via same API) | DEFERRED — only after P0 passes end-to-end |
| P1 | LAN access from a second computer | DEFERRED — requires security approval + real test |
| P2 | Profiling / optional official CPU export | DEFERRED — separate approved work order |

## Development model (OAP)

This repository is governed by the Orchestrated Agentic Programming (OAP) model defined in
[`AGENTS.md`](AGENTS.md) — read it before contributing:

| Role | Owns |
|---|---|
| **Human lead** | Goals, risk acceptance, scope approval, PR merge, demo/release decision |
| **Strategic AI** | Discovery, architecture, constitution, precise work orders, evidence review |
| **Execution agent** (e.g. Codex CLI) | Bounded implementation in the approved workspace, tests, commits, PRs, evidence reports |

Rules that bind every contribution: one bounded branch per work order from an agreed base
(`docs/readme`, `feat/cpu-inference`, …), no direct commits to `main`, no self-merge, tests and
docs updated in the same change as behavior, and every claim of performance/CPU operation
backed by actual measurement. Significant approved decisions get a short ADR in `docs/adr/`.

## Getting started

> **Greenfield notice:** there is no runnable code in this repository yet. The workflow below is
> *planned* and will be verified and corrected by the implementation work order that lands
> `requirements.txt` and `backend/`. Do not treat these commands as tested.

Planned workflow (Ubuntu / WSL2):

```bash
git clone https://github.com/david-bernes/delavnica
cd delavnica
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt        # added with P0 implementation
# First start downloads pretrained weights from the official Ultralytics source (Internet needed once)
uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

Planned checks:

```bash
curl -s http://127.0.0.1:8000/health
curl -s -F "file=@sample.jpg" http://127.0.0.1:8000/detect
# Open http://127.0.0.1:8000/ in a browser for the interactive demo
# Stop the service: Ctrl+C (single Uvicorn worker, no state to clean up)
```

A full step-by-step `docs/demo-runbook.md` (start, image test, webcam test, curl, stop, common
failure recovery) is part of the P0 deliverables.

## Testing & verification approach

- **Unit/API tests** (mocked inference, fast, offline): config validation incl. GPU-rejection,
  image validation boundaries, result-schema/geometry contract, error paths, no-persistence.
- **Live-model CPU integration** (opt-in marker, needs official weights): real pretrained
  checkpoint, `device="cpu"`, real HTTP endpoint, known test image; a mocked test is explicitly
  *not* proof of real CPU inference.
- **Browser smoke**: upload/overlay, image switching, no-detection case, permission errors,
  Start/Stop, single in-flight frame, stale-response rejection — manual portions labeled as such.
- **Quality gates** per change: `pytest`, `ruff check`, `python -m compileall`, `pip check`,
  Uvicorn launch + `curl` smoke.

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
