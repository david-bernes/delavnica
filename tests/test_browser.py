"""Real-browser verification (Playwright + headless Chromium).

WO-001V: these tests drive the *actual* web application served by the real
FastAPI/Uvicorn process over HTTP with a real browser. They complement the
mocked unit/API tests; a passing CSS source inspection or HTTP test is not a
substitute for rendered-geometry and interaction evidence.

Setup (once, in the guest)::

    .venv/bin/python -m pip install -r requirements-dev.txt
    .venv/bin/python -m playwright install --with-deps chromium

Run::

    .venv/bin/python -m pytest tests/test_browser.py -v

The session fixture reuses an already-running service at
``http://127.0.0.1:8000`` (override with ``BROWSER_TEST_URL``) and starts
``python -m backend.main`` only if none is reachable. End-to-end detection
tests skip (not fail) when the official weights have not been downloaded.

Screenshots and generated fixtures are written to ``artifacts/browser/``
(git-ignored) and never committed.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import httpx
import pytest
import pytest_asyncio
from PIL import Image
from playwright.async_api import Page, async_playwright

pytestmark = pytest.mark.asyncio

REPO_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = REPO_ROOT / "artifacts" / "browser"
APP_URL = os.environ.get("BROWSER_TEST_URL", "http://127.0.0.1:8000")
BUS_SOURCE_URL = "https://ultralytics.com/images/bus.jpg"
RECT_TOLERANCE_PX = 1.5

# --------------------------------------------------------------------------
# JS helpers evaluated inside the real page
# --------------------------------------------------------------------------

DONE_JS = """
() => {
  const results = document.getElementById("results");
  const loading = document.getElementById("loading");
  const error = document.getElementById("error");
  return (
    !results.classList.contains("hidden") &&
    loading.classList.contains("hidden") &&
    error.classList.contains("hidden")
  );
}
"""

RECTS_JS = """
() => {
  const r = (sel) => {
    const b = document.querySelector(sel).getBoundingClientRect();
    return {x: b.x, y: b.y, width: b.width, height: b.height};
  };
  return {preview: r("#preview"), overlay: r("#overlay"), frame: r("#frame")};
}
"""

INK_JS = """
() => {
  const c = document.getElementById("overlay");
  if (!c.width || !c.height) return 0;
  const d = c.getContext("2d").getImageData(0, 0, c.width, c.height).data;
  let n = 0;
  for (let i = 3; i < d.length; i += 4) if (d[i] > 0) n++;
  return n;
}
"""

# Simulates the internal race behind F-002: start a real detection for the
# currently selected image, invalidate it (as selecting a new file would),
# then let the delayed response arrive. The seq guard must discard it.
SEQ_GUARD_JS = """
async (stale) => {
  const origFetch = window.fetch;
  let responded = false;
  window.fetch = (input, init) => {
    const url = String(typeof input === "string" ? input : input.url);
    if (url.endsWith("/detect")) {
      return new Promise((resolve) => setTimeout(() => {
        responded = true;
        resolve(new Response(JSON.stringify(stale), {
          status: 200,
          headers: {"Content-Type": "application/json"},
        }));
      }, 300));
    }
    return origFetch(input, init);
  };
  const ink = () => {
    const c = document.getElementById("overlay");
    const d = c.getContext("2d").getImageData(0, 0, c.width, c.height).data;
    let n = 0;
    for (let i = 3; i < d.length; i += 4) if (d[i] > 0) n++;
    return n;
  };
  setTimeout(() => { invalidateCurrent(); }, 50);
  await runDetection();
  window.fetch = origFetch;
  return {
    responded: responded,
    lastResult: lastResult,
    countHidden: document.getElementById("results").classList.contains("hidden"),
    ink: ink(),
    loadingHidden: document.getElementById("loading").classList.contains("hidden"),
    detectEnabled: !document.getElementById("detect").disabled,
  };
}
"""

CHIP_RE = re.compile(r"^[a-z_]+ \d{1,3}\.\d%$")
TIMING_RE = re.compile(r"inference: \d+\.\d ms \u00b7 total: \d+\.\d ms")


# --------------------------------------------------------------------------
# Server lifecycle
# --------------------------------------------------------------------------


def _health_ready(url: str) -> bool:
    try:
        with urllib.request.urlopen(url + "/health", timeout=3) as res:
            data = json.loads(res.read())
            return res.status == 200 and data.get("status") == "ready"
    except Exception:
        return False


@pytest.fixture(scope="session")
def app_url():
    """Reuse a running service, or start one for the session."""
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    if _health_ready(APP_URL):
        yield APP_URL
        return
    log_path = ARTIFACTS_DIR / "server.log"
    log = open(log_path, "ab")
    proc = subprocess.Popen(
        [sys.executable, "-m", "backend.main"],
        cwd=REPO_ROOT,
        stdout=log,
        stderr=subprocess.STDOUT,
        env=dict(os.environ),
    )
    try:
        deadline = time.time() + 240
        ready = False
        while time.time() < deadline:
            if _health_ready(APP_URL):
                ready = True
                break
            if proc.poll() is not None:
                break
            time.sleep(2)
        if not ready:
            pytest.fail(f"app did not become ready at {APP_URL}; see {log_path}")
        yield APP_URL
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
        log.close()


# --------------------------------------------------------------------------
# Test images
# --------------------------------------------------------------------------


@pytest.fixture(scope="session")
def images(app_url):
    """Legal, deterministic test images + a real /detect response for bus.jpg."""
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    bus = ARTIFACTS_DIR / "bus.jpg"
    if not bus.exists():
        candidate = Path("/tmp/bus.jpg")
        if candidate.exists():
            shutil.copyfile(candidate, bus)
        else:  # official Ultralytics sample image (documented provenance)
            with urllib.request.urlopen(BUS_SOURCE_URL, timeout=120) as res, open(bus, "wb") as fh:
                shutil.copyfileobj(res, fh)
    with Image.open(bus) as im:
        im.verify()
        with Image.open(bus) as im:
            bus_size = im.size

    # Landscape fixture: the scene upright, letterboxed into a 1280x800
    # canvas (a neutral-gray fill that yields no detections). Rotating the
    # photo would tilt the objects and degrade the real model's detections.
    landscape = ARTIFACTS_DIR / "landscape.jpg"
    portrait = ARTIFACTS_DIR / "portrait.jpg"
    target_w, target_h = 1280, 800
    with Image.open(bus) as im:
        if im.size[0] >= im.size[1]:
            im.save(landscape, format="JPEG", quality=90)
            im.transpose(Image.Transpose.ROTATE_90).save(portrait, format="JPEG", quality=90)
        else:
            im.save(portrait, format="JPEG", quality=90)
            scale = min(target_w / im.width, target_h / im.height)
            resized = im.resize((int(im.width * scale), int(im.height * scale)), Image.LANCZOS)
            canvas = Image.new("RGB", (target_w, target_h), (120, 120, 120))
            offset = ((target_w - resized.width) // 2, (target_h - resized.height) // 2)
            canvas.paste(resized, offset)
            canvas.save(landscape, format="JPEG", quality=90)
    with Image.open(landscape) as im:
        landscape_size = im.size

    blank = ARTIFACTS_DIR / "blank.jpg"
    Image.new("RGB", (800, 600), (120, 120, 120)).save(blank, format="JPEG", quality=90)

    unsupported = ARTIFACTS_DIR / "notes.txt"
    unsupported.write_text("This is not an image.\n")

    oversized = ARTIFACTS_DIR / "big.jpg"
    if not oversized.exists():
        with open(oversized, "wb") as fh:
            fh.truncate(9 * 1024 * 1024)  # 9 MiB > 8 MiB client limit

    bus_json = None
    try:
        with open(bus, "rb") as fh:
            res = httpx.post(
                APP_URL + "/detect",
                files={"file": ("bus.jpg", fh, "image/jpeg")},
                timeout=120,
            )
        if res.status_code == 200:
            bus_json = res.json()
    except Exception:
        bus_json = None

    return {
        "dir": ARTIFACTS_DIR,
        "bus": bus,
        "landscape": landscape,
        "landscape_size": landscape_size,
        "portrait": portrait,
        "blank": blank,
        "unsupported": unsupported,
        "oversized": oversized,
        "bus_size": bus_size,
        "bus_json": bus_json,
    }


def require_model(images):
    """Skip e2e detection tests when the real checkpoint is unavailable."""
    if images["bus_json"] is None:
        pytest.skip(
            "real model not available at /detect (official weights missing?); "
            "browser end-to-end detection test skipped"
        )
    return images["bus_json"]


# --------------------------------------------------------------------------
# Page harness
# --------------------------------------------------------------------------


@dataclass
class PageUnderTest:
    page: Page
    events: dict = field(repr=False, default_factory=dict)

    async def upload(self, path):
        await self.page.set_input_files("#file", str(path))

    async def wait_done(self, timeout: int = 30000):
        await self.page.wait_for_function(DONE_JS, timeout=timeout)

    async def rects(self):
        return await self.page.evaluate(RECTS_JS)

    async def ink(self):
        return await self.page.evaluate(INK_JS)

    async def text(self, selector: str):
        js = f"document.querySelector('{selector}').textContent"
        return (await self.page.evaluate(js)).strip()

    async def chips(self):
        expr = "els => els.map(e => e.textContent)"
        return await self.page.eval_on_selector_all("#classes li", expr)

    async def hidden(self, element_id: str) -> bool:
        js = f"document.getElementById('{element_id}').classList.contains('hidden')"
        return await self.page.evaluate(js)

    async def screenshot(self, name: str):
        await self.page.screenshot(path=str(ARTIFACTS_DIR / name), full_page=False)


def assert_rects_aligned(rects, tol: float = RECT_TOLERANCE_PX):
    """Image, canvas and frame must share the same displayed rectangle."""
    for a, b in (("preview", "overlay"), ("preview", "frame"), ("overlay", "frame")):
        ra, rb = rects[a], rects[b]
        for key in ("x", "y", "width", "height"):
            assert abs(ra[key] - rb[key]) <= tol, (
                f"{a}.{key}={ra[key]:.2f} vs {b}.{key}={rb[key]:.2f} (tolerance {tol} px)"
            )


def assert_static_assets(env: PageUnderTest):
    statuses = {}
    for url, status in env.events["responses"]:
        name = url.rsplit("/", 1)[-1]
        if name in ("styles.css", "app.js"):
            statuses[name] = status
    for name in ("styles.css", "app.js"):
        assert statuses.get(name) == 200, f"{name} was not loaded with HTTP 200: {statuses}"


def assert_page_clean(env: PageUnderTest):
    if env.events["pageerrors"]:
        raise AssertionError(f"uncaught JavaScript exceptions: {env.events['pageerrors']}")
    assert not env.events["console_errors"], f"console errors: {env.events['console_errors']}"
    bad = [(u, s) for u, s in env.events["responses"] if s >= 400]
    assert not bad, f"unexpected HTTP error responses: {bad}"


@pytest_asyncio.fixture
async def page(app_url):  # noqa: B008 - pytest-asyncio fixture
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        context = await browser.new_context(
            viewport={"width": 1280, "height": 800}, device_scale_factor=1
        )
        page = await context.new_page()
        page.set_default_timeout(20000)
        events: dict = {
            "pageerrors": [],
            "console_errors": [],
            "responses": [],
            "detect_requests": [],
            "failed_requests": [],
        }
        page.on("pageerror", lambda err: events["pageerrors"].append(str(err)))
        page.on(
            "console",
            lambda msg: events["console_errors"].append(msg.text) if msg.type == "error" else None,
        )
        page.on("response", lambda resp: events["responses"].append((resp.url, resp.status)))
        page.on(
            "request",
            lambda req: events["detect_requests"].append(req.url)
            if req.url.rsplit("/", 1)[-1] == "detect"
            else None,
        )
        page.on(
            "requestfailed", lambda req: events["failed_requests"].append((req.url, req.failure))
        )
        await page.goto(app_url + "/", wait_until="domcontentloaded")
        await page.wait_for_function(
            "document.getElementById('detect').disabled === false", timeout=60000
        )
        await page.evaluate("window.__reloadSentinel = 1")
        yield PageUnderTest(page, events)
        await context.close()
        await browser.close()


# --------------------------------------------------------------------------
# TEST-01 .. TEST-12
# --------------------------------------------------------------------------


async def test_01_landscape_detection(page, images):
    require_model(images)
    await page.upload(images["landscape"])
    await page.wait_done()
    natural = await page.page.evaluate(
        "({w: document.getElementById('preview').naturalWidth,"
        " h: document.getElementById('preview').naturalHeight})"
    )
    assert natural == {
        "w": images["landscape_size"][0],
        "h": images["landscape_size"][1],
    }
    assert natural["w"] > natural["h"], f"landscape fixture is not landscape: {natural}"
    chips = await page.chips()
    assert len(chips) >= 2, f"expected multiple detections on bus.jpg, got {chips}"
    assert (await page.text("#count")) == f"{len(chips)} objects detected"
    assert any(c.startswith("person") for c in chips)
    assert await page.ink() > 0, "bounding boxes were not drawn on the canvas"
    assert_rects_aligned(await page.rects())
    assert "READY" in await page.text("#status")
    await page.screenshot("test01_landscape_detection.png")
    assert_static_assets(page)
    assert_page_clean(page)


async def test_02_portrait_detection(page, images):
    require_model(images)
    await page.upload(images["portrait"])
    await page.wait_done()
    natural = await page.page.evaluate(
        "({w: document.getElementById('preview').naturalWidth,"
        " h: document.getElementById('preview').naturalHeight})"
    )
    assert natural["w"] < natural["h"], f"portrait fixture is not portrait: {natural}"
    assert await page.ink() > 0
    assert_rects_aligned(await page.rects())
    await page.screenshot("test02_portrait_detection.png")
    assert_page_clean(page)


async def test_03_viewport_resizing(page, images):
    require_model(images)
    await page.upload(images["bus"])
    await page.wait_done()
    for width, height in ((900, 700), (1400, 900), (1280, 800)):
        await page.page.set_viewport_size({"width": width, "height": height})
        await page.page.wait_for_timeout(250)  # let the resize handler run
        assert_rects_aligned(await page.rects())
        assert await page.ink() > 0, f"boxes disappeared after resize to {width}x{height}"


async def test_04_narrow_viewport(page, images):
    require_model(images)
    await page.upload(images["bus"])
    await page.wait_done()
    await page.page.set_viewport_size({"width": 360, "height": 740})
    await page.page.wait_for_timeout(250)
    assert_rects_aligned(await page.rects())
    assert await page.ink() > 0
    file_box = await page.page.locator("#file").bounding_box()
    detect_box = await page.page.locator("#detect").bounding_box()
    assert file_box is not None and file_box["width"] > 50, "file input unusable in narrow viewport"
    assert detect_box is not None, "Detect button not visible in narrow viewport"
    await page.screenshot("test04_narrow_viewport.png")
    assert_page_clean(page)


async def test_05_zero_detections(page, images):
    require_model(images)
    await page.upload(images["blank"])
    await page.wait_done()
    assert (await page.text("#count")) == "No objects detected"
    assert await page.ink() == 0, "stale or phantom boxes drawn on a zero-detection image"
    assert not await page.hidden("results"), "results section must be visible for zero detections"
    assert_rects_aligned(await page.rects())
    await page.screenshot("test05_zero_detections.png")
    assert_page_clean(page)


async def test_06_multiple_detections_displayed(page, images):
    require_model(images)
    await page.upload(images["bus"])
    await page.wait_done()
    chips = await page.chips()
    assert len(chips) >= 2
    for chip in chips:
        assert CHIP_RE.match(chip), f"chip not in 'class NN.N%' form: {chip!r}"
    assert TIMING_RE.search(await page.text("#timing")), "server timing not displayed"
    assert await page.ink() > 0
    assert_rects_aligned(await page.rects())
    await page.screenshot("test06_multiple_detections.png")
    assert_page_clean(page)


async def test_07_unsupported_file_rejected(page, images):
    require_model(images)
    await page.upload(images["bus"])
    await page.wait_done()
    assert await page.ink() > 0
    requests_before = len(page.events["detect_requests"])
    await page.upload(images["unsupported"])
    await page.page.wait_for_selector("#error:not(.hidden)")
    assert "JPEG or PNG" in await page.text("#error")
    assert await page.ink() == 0, "previous annotations survived an unsupported file"
    assert await page.hidden("results"), "results panel survived an unsupported file"
    assert await page.hidden("loading"), "loading indicator stuck after rejection"
    assert await page.page.evaluate("document.getElementById('file').value") == ""
    assert await page.page.evaluate("document.getElementById('detect').disabled") is False
    assert len(page.events["detect_requests"]) == requests_before, "rejected file was submitted"
    await page.screenshot("test07_error_rejected_file.png")
    assert_page_clean(page)


async def test_08_oversized_file_rejected(page, images):
    require_model(images)
    await page.upload(images["bus"])
    await page.wait_done()
    requests_before = len(page.events["detect_requests"])
    await page.upload(images["oversized"])
    await page.page.wait_for_selector("#error:not(.hidden)")
    assert "8 MiB" in await page.text("#error")
    assert await page.ink() == 0
    assert await page.hidden("results")
    assert await page.hidden("loading")
    assert await page.page.evaluate("document.getElementById('file').value") == ""
    assert len(page.events["detect_requests"]) == requests_before, "oversized file was submitted"
    await page.screenshot("test08_oversized_file.png")
    assert_page_clean(page)


async def test_09_stale_response_never_annotates(page, images):
    """Select a second image while the first request is still pending."""
    bus_json = require_model(images)
    state = {"n": 0, "fulfilled": False}

    async def handle(route):
        state["n"] += 1
        if state["n"] == 1:
            await asyncio.sleep(2.0)  # hold the first response back
            try:
                await route.fulfill(
                    status=200,
                    content_type="application/json",
                    body=json.dumps(bus_json),
                )
                state["fulfilled"] = True
            except Exception:
                pass  # request was aborted before it could be fulfilled
        else:
            await route.continue_()

    await page.page.route("**/detect", handle)
    await page.upload(images["bus"])
    await page.page.wait_for_event(
        "request", predicate=lambda r: r.url.rsplit("/", 1)[-1] == "detect"
    )
    await page.page.wait_for_timeout(300)
    await page.upload(images["blank"])
    await page.wait_done()
    assert (await page.text("#count")) == "No objects detected"
    # Let the deliberately delayed first response land (or be discarded).
    await page.page.wait_for_timeout(2300)
    assert (await page.text("#count")) == "No objects detected", (
        "stale response from the previous image annotated the new image"
    )
    assert await page.ink() == 0, "stale bounding boxes drawn over the new image"
    assert await page.chips() == []
    assert await page.hidden("error")
    aborted = any(
        "abort" in (failure or "").lower() for _, failure in page.events["failed_requests"]
    ) or not state["fulfilled"]
    assert aborted, "the pending request was not cancelled when the image changed"
    await page.screenshot("test09_stale_response_discarded.png")
    assert_page_clean(page)


async def test_09b_seq_guard_discards_late_response(page, images):
    """Deterministic exercise of the in-app stale-response (seq) guard."""
    bus_json = require_model(images)
    await page.upload(images["blank"])
    await page.wait_done()
    result = await page.page.evaluate(SEQ_GUARD_JS, bus_json)
    assert result["responded"] is True, "fake stale response never arrived"
    assert result["lastResult"] is None, "stale detection result was stored"
    assert result["countHidden"] is True, "stale results were displayed"
    assert result["ink"] == 0, "stale boxes were drawn"
    assert result["loadingHidden"] is True
    assert result["detectEnabled"] is True
    assert_page_clean(page)


async def test_10_detect_after_rejection_sends_nothing(page, images):
    require_model(images)
    await page.upload(images["bus"])
    await page.wait_done()
    await page.upload(images["unsupported"])
    await page.page.wait_for_selector("#error:not(.hidden)")
    requests_before = len(page.events["detect_requests"])
    await page.page.click("#detect")
    await page.page.wait_for_timeout(400)
    assert len(page.events["detect_requests"]) == requests_before, (
        "Detect submitted a rejected file"
    )
    assert await page.hidden("results")
    assert await page.ink() == 0
    assert (await page.text("#error")) != ""


async def test_11_recovery_after_rejection(page, images):
    require_model(images)
    await page.upload(images["bus"])
    await page.wait_done()
    await page.upload(images["unsupported"])
    await page.page.wait_for_selector("#error:not(.hidden)")
    await page.upload(images["bus"])
    await page.wait_done()
    chips = await page.chips()
    assert len(chips) >= 2, "detection did not recover after a rejected file"
    assert await page.hidden("error"), "stale error message survived recovery"
    assert await page.ink() > 0
    assert (await page.page.evaluate("window.__reloadSentinel")) == 1, "page was reloaded"
    await page.screenshot("test11_recovery_after_rejection.png")
    assert_page_clean(page)


async def test_12_repeated_interactions(page, images):
    require_model(images)
    sequence = [
        (images["bus"], True),
        (images["blank"], False),
        (images["bus"], True),
        (images["portrait"], True),
    ]
    for path, expect_detections in sequence:
        await page.upload(path)
        await page.wait_done()
        chips = await page.chips()
        if expect_detections:
            assert len(chips) >= 1, f"no detections for {path.name}"
            assert await page.ink() > 0
        else:
            assert (await page.text("#count")) == "No objects detected"
            assert await page.ink() == 0
        assert await page.hidden("loading"), "loading indicator stuck"
        assert await page.hidden("error"), "error appeared during repeat uploads"
        assert await page.page.evaluate("document.getElementById('detect').disabled") is False
    assert_page_clean(page)
