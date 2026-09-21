# Demo runbook — delavnica (WO-001)

Exact steps for the live CPU-only YOLO detection demo on Ubuntu/WSL2.
Commands below were executed and verified on the target machine
(AMD Ryzen AI 7 350, 8 cores / 16 threads, 15 GiB RAM, Ubuntu 26.04.1 under WSL2,
Python 3.14.4) on 2026-09-21.

## 1. Fresh setup (one time per machine)

```bash
sudo apt-get update && sudo apt-get install -y python3.14-venv   # if venv is missing
git clone https://github.com/david-bernes/delavnica
cd delavnica
python3 -m venv .venv
source .venv/bin/activate

# 1) CPU-only PyTorch first (official PyTorch CPU wheel index)
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 2) The rest (pins are satisfied by the +cpu builds from step 1)
python -m pip install -r requirements.txt
```

First start downloads the official `yolo26n.pt` checkpoint (~5.3 MB) from the
Ultralytics asset release into `~/.cache/ultralytics/weights/` (Internet needed
once; for an offline demo copy the file there beforehand — see §5).

## 2. Start the service

```bash
source .venv/bin/activate
python -m backend.main
```

Expected log:

```text
INFO backend.inference: Loading model yolo26n.pt (device=cpu, imgsz=640, conf=0.25)
INFO backend.inference: Model ready in ~1000 ms (torch threads: 8)
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:8000
```

The service is ready only after model load **and** one warm-up inference
(`/health` returns 503 until then — by design, never fake readiness).

## 3. Verify (two terminals)

Terminal A — start the service (§2). Terminal B:

```bash
# readiness
curl -s http://127.0.0.1:8000/health
# expected: {"status":"ready","model":"yolo26n.pt","device":"cpu",...}

# detection on any local JPEG/PNG (use a photo you may show)
curl -s -F "file=@/path/to/photo.jpg" http://127.0.0.1:8000/detect | python3 -m json.tool
```

Verified example (2026-09-21, `/tmp/bus.jpg`, 810×1080, Ultralytics sample
image, not committed): 5 detections — `bus` 88.3% and four `person` boxes,
server inference ≈ 22 ms (median over 20 warm requests).

Browser demo: open **http://127.0.0.1:8000/** — select a JPEG/PNG, boxes,
labels, percentages and server timing are drawn over the image.
Interactive API docs: http://127.0.0.1:8000/docs

## 4. Stop the service

`Ctrl+C` in the server terminal. Single worker, no state to clean up.

## 5. Offline demo: pre-stage weights

```bash
# on a machine with internet:
curl -L -o ~/.cache/ultralytics/weights/yolo26n.pt \
  https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt
```

Copy `~/.cache/ultralytics/weights/yolo26n.pt` to the same path on the demo
machine (or set `YOLO_MODEL=/abs/path/to/yolo26n.pt`). The service then starts
fully offline. Never commit weights to Git.

## 6. Common failures and recovery

| Symptom | Cause | Fix |
|---|---|---|
| `/health` 503 forever, log `Model load failed` | weights download failed (no internet) | pre-stage weights (§5) or restore connectivity; check log for the actual error |
| `ModuleNotFoundError: ultralytics` / `torch` | wrong Python environment | re-run §1; use `.venv` (never system Python) |
| `operator torchvision::nms does not exist` | torchvision from PyPI mixed with +cpu torch | `python -m pip install --force-reinstall --no-deps torchvision --index-url https://download.pytorch.org/whl/cpu` |
| `Error: [Errno 98] address already in use` | port 8000 taken | `PORT=8010 python -m backend.main` |
| `413 image_too_large` on upload | > 8 MiB compressed or > 24 MP decoded | resize/compress the image, or raise `MAX_UPLOAD_BYTES`/`MAX_IMAGE_PIXELS` deliberately |
| `429 inference_overloaded` | another request is mid-inference (concurrency 1) | wait `Retry-After` seconds; normal for competing browser tabs |
| Webcam does not work from another PC over LAN | browser requires a secure context for camera access | expected limitation of WO-001 (no webcam yet); image upload works over LAN only after approved exposure |
| Slow first request after start | should not happen (warm-up at startup) | check server log for `Model ready in … ms` |

## 7. Known boundaries (do not present as features)

- No webcam, no video-file input (deferred to later work orders).
- Loopback binding by default; LAN exposure is a separate, approved task.
- COCO general-object detection — not an industrial inspection system.
- Demo/prototype: unauthenticated, single process, not production-hardened.
