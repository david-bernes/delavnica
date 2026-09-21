"use strict";

// delavnica browser demo — vanilla JS, same-origin relative URLs only.
//
// Rules implemented here (see AGENTS.md §6):
//  - at most one outstanding /detect request (new image aborts the old one);
//  - stale responses (replaced image / aborted fetch) are ignored;
//  - old boxes are cleared immediately when the image changes;
//  - overlays are scaled from API image pixels to displayed CSS pixels.

const API = {
  health: "/health",
  detect: "/detect",
};

const MAX_UPLOAD_BYTES = 8 * 1024 * 1024;

const els = {
  status: document.getElementById("status"),
  file: document.getElementById("file"),
  detect: document.getElementById("detect"),
  loading: document.getElementById("loading"),
  stage: document.getElementById("stage"),
  empty: document.getElementById("empty"),
  preview: document.getElementById("preview"),
  overlay: document.getElementById("overlay"),
  results: document.getElementById("results"),
  count: document.getElementById("count"),
  classes: document.getElementById("classes"),
  timing: document.getElementById("timing"),
  error: document.getElementById("error"),
};

const BOX_COLORS = [
  "#4cc2ff", "#35d07f", "#ffc857", "#ff6b6b", "#c58fff",
  "#7ee8fa", "#ffb0e6", "#a3e635", "#f97316", "#e879f9",
];
const nameToColor = new Map();
function colorFor(name) {
  if (!nameToColor.has(name)) {
    const hash = [...name].reduce((h, ch) => (h * 31 + ch.charCodeAt(0)) >>> 0, 7);
    nameToColor.set(name, BOX_COLORS[hash % BOX_COLORS.length]);
  }
  return nameToColor.get(name);
}

let seq = 0;                 // request/image generation counter
let controller = null;       // AbortController of the in-flight request
let lastResult = null;       // {detections, image} for redraw on resize

// ---------------------------------------------------------------- status

function setStatus(text, kind) {
  els.status.textContent = text;
  els.status.className = "status" + (kind ? " " + kind : "");
}

function setServiceEnabled(enabled) {
  els.detect.disabled = !enabled;
}

async function checkHealth() {
  try {
    const res = await fetch(API.health, { cache: "no-store" });
    const data = await res.json().catch(() => null);
    if (res.ok && data && data.status === "ready") {
      setStatus(`READY · ${data.model} · CPU · imgsz ${data.image_size}`, "ok");
      setServiceEnabled(true);
      return true;
    }
    const msg = (data && data.error && data.error.message) || "service not ready";
    setStatus("NOT READY", "bad");
    setServiceEnabled(false);
    showError("Backend not ready: " + msg + " — retrying…");
    return false;
  } catch (err) {
    setStatus("UNREACHABLE", "bad");
    setServiceEnabled(false);
    showError("Could not reach the detection service. Is the server running?");
    return false;
  }
}

// Retry while the model is still loading on first start (bounded).
async function waitUntilReady(attempts = 20, delayMs = 3000) {
  for (let i = 0; i < attempts; i += 1) {
    if (await checkHealth()) return;
    await new Promise((resolve) => setTimeout(resolve, delayMs));
  }
}

// ---------------------------------------------------------------- errors

function showError(message) {
  if (!message) {
    els.error.classList.add("hidden");
    return;
  }
  els.error.textContent = message;
  els.error.classList.remove("hidden");
}

// ---------------------------------------------------------------- preview

function showPreview(file) {
  els.empty.classList.add("hidden");
  els.preview.classList.remove("hidden");
  const previous = els.preview.src;
  const url = URL.createObjectURL(file);
  els.preview.onload = () => {
    if (previous.startsWith("blob:")) URL.revokeObjectURL(previous);
    layoutOverlay(); // size canvas after display size is known
  };
  els.preview.src = url;
}

// ---------------------------------------------------------------- detect

function invalidateCurrent() {
  seq += 1;
  if (controller) {
    controller.abort();
    controller = null;
  }
  clearOverlay();
  lastResult = null;
  els.results.classList.add("hidden");
  showError(null);
}

async function runDetection() {
  const file = els.file.files[0];
  if (!file) return;
  invalidateCurrent();
  const mySeq = seq;
  controller = new AbortController();
  els.loading.classList.remove("hidden");
  els.detect.disabled = true;

  const form = new FormData();
  form.append("file", file);

  try {
    const res = await fetch(API.detect, {
      method: "POST",
      body: form,
      signal: controller.signal,
    });
    const data = await res.json().catch(() => null);
    if (mySeq !== seq) return; // replaced by a newer image — ignore
    if (!res.ok) {
      const msg =
        (data && data.error && data.error.message) ||
        "Request failed (HTTP " + res.status + ")";
      if (res.status === 429) {
        showError("Server is busy (429). Waiting a moment, retry…");
        setTimeout(() => { if (mySeq === seq) runDetection(); }, 1500);
        return;
      }
      showError(msg);
      return;
    }
    lastResult = data;
    drawOverlay(data.detections, data.image);
    showResults(data);
  } catch (err) {
    if (mySeq !== seq) return;
    if (err && err.name === "AbortError") return;
    showError("Could not reach the detection service.");
  } finally {
    if (mySeq === seq) {
      els.loading.classList.add("hidden");
      setServiceEnabled(true);
    }
  }
}

// ---------------------------------------------------------------- overlay

function layoutOverlay() {
  // Size the canvas backing store to the displayed image, scaled for the
  // device pixel ratio. The canvas is centered in the stage exactly like
  // the image, so CSS size == displayed image size (no letterbox offsets).
  const preview = els.preview;
  if (!lastResult || preview.classList.contains("hidden")) return;
  const dpr = window.devicePixelRatio || 1;
  const cssW = preview.clientWidth;
  const cssH = preview.clientHeight;
  if (!cssW || !cssH) return;
  const canvas = els.overlay;
  canvas.width = Math.round(cssW * dpr);
  canvas.height = Math.round(cssH * dpr);
  canvas.style.width = cssW + "px";
  canvas.style.height = cssH + "px";
  drawOverlay(lastResult.detections, lastResult.image);
}

function drawOverlay(detections, apiImage) {
  const canvas = els.overlay;
  if (canvas.classList.contains("hidden")) canvas.classList.remove("hidden");
  const dpr = window.devicePixelRatio || 1;
  const cssW = els.preview.clientWidth;
  const cssH = els.preview.clientHeight;
  if (!cssW || !cssH) return;
  if (canvas.width !== Math.round(cssW * dpr) || canvas.height !== Math.round(cssH * dpr)) {
    canvas.width = Math.round(cssW * dpr);
    canvas.height = Math.round(cssH * dpr);
    canvas.style.width = cssW + "px";
    canvas.style.height = cssH + "px";
  }
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, cssW, cssH);

  const srcW = (apiImage && apiImage.width) || els.preview.naturalWidth;
  const srcH = (apiImage && apiImage.height) || els.preview.naturalHeight;
  if (!srcW || !srcH) return;
  const sx = cssW / srcW; // displayed CSS px per source pixel
  const sy = cssH / srcH;

  for (const det of detections) {
    const [bx1, by1, bx2, by2] = det.box;
    const x = Math.max(0, Math.min(bx1, srcW)) * sx;
    const y = Math.max(0, Math.min(by1, srcH)) * sy;
    const x2 = Math.max(0, Math.min(bx2, srcW)) * sx;
    const y2 = Math.max(0, Math.min(by2, srcH)) * sy;
    const w = x2 - x;
    const h = y2 - y;
    if (w <= 0 || h <= 0) continue;

    const color = colorFor(det.class_name);
    ctx.lineWidth = 2;
    ctx.strokeStyle = color;
    ctx.strokeRect(x, y, w, h);

    const label = det.class_name + " " + (det.confidence * 100).toFixed(1) + "%";
    ctx.font = "600 13px system-ui, sans-serif";
    const tw = ctx.measureText(label).width;
    const padX = 5;
    const boxH = 18;
    // Keep the label inside the image: flip below the top edge when needed.
    let labelY = y - boxH;
    if (labelY < 1) labelY = Math.min(y + 2, cssH - boxH - 1);
    const labelX = Math.max(1, Math.min(x, cssW - tw - padX * 2 - 1));
    ctx.globalAlpha = 0.92;
    ctx.fillStyle = color;
    ctx.fillRect(labelX, labelY, tw + padX * 2, boxH);
    ctx.globalAlpha = 1;
    ctx.fillStyle = "#08131b";
    ctx.fillText(label, labelX + padX, labelY + 13);
  }
}

function clearOverlay() {
  const ctx = els.overlay.getContext("2d");
  ctx.save();
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.clearRect(0, 0, els.overlay.width, els.overlay.height);
  ctx.restore();
  els.overlay.classList.add("hidden");
}

// ---------------------------------------------------------------- results

function showResults(data) {
  const n = data.detections.length;
  els.count.textContent = n === 0
    ? "No objects detected"
    : n + (n === 1 ? " object detected" : " objects detected");
  els.classes.innerHTML = "";
  for (const det of data.detections) {
    const li = document.createElement("li");
    li.textContent = det.class_name + " " + (det.confidence * 100).toFixed(1) + "%";
    els.classes.appendChild(li);
  }
  const t = data.timing_ms || {};
  els.timing.textContent =
    "Server timing — inference: " + (t.inference != null ? t.inference.toFixed(1) : "?") +
    " ms · total: " + (t.total != null ? t.total.toFixed(1) : "?") + " ms";
  els.results.classList.remove("hidden");
}

// ---------------------------------------------------------------- wiring

els.file.addEventListener("change", () => {
  const file = els.file.files && els.file.files[0];
  if (!file) return;
  if (!/^image\/(jpeg|png)$/.test(file.type)) {
    showError("Please choose a JPEG or PNG image file.");
    return;
  }
  if (file.size > MAX_UPLOAD_BYTES) {
    showError("The selected file is larger than the 8 MiB upload limit.");
    return;
  }
  invalidateCurrent(); // never annotate image A over image B
  showPreview(file);
  runDetection();
});

els.detect.addEventListener("click", () => {
  if (els.preview.classList.contains("hidden")) return;
  runDetection();
});

window.addEventListener("resize", () => layoutOverlay());
els.preview.addEventListener("load", () => layoutOverlay());

checkHealth().then((ready) => { if (!ready) waitUntilReady(); });
