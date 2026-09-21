"""Source-level regression guards for the browser demo (WO-001R: F-001/F-002/F-003).

No browser or JS runtime exists in the approved execution environment, so
real-browser geometry and state behavior cannot be automated here
(limitation reported in the agent report). These tests pin the corrective
invariants at the source level so a regression cannot silently return:

F-001: the overlay canvas lives inside the same position:relative wrapper as
       the image and is pinned with inset:0, and hidden elements use
       display (never the visibility property) so hidden elements can never
       occupy layout space or displace the displayed image.
F-002: file-change invalidation runs BEFORE client-side validation returns,
       so selecting an invalid or oversized file cannot leave the previous
       image's detections visible.
F-003: invalidating an in-flight request always clears the loading indicator
       (the aborted request's finally block is seq-guarded and will not),
       and a rejected file is cleared from the input so the Detect button
       can never submit it over the previous image's preview.
"""

from __future__ import annotations

import re
from pathlib import Path

FRONTEND = Path(__file__).resolve().parent.parent / "frontend"
HTML = (FRONTEND / "index.html").read_text(encoding="utf-8")
CSS = (FRONTEND / "styles.css").read_text(encoding="utf-8")
JS = (FRONTEND / "app.js").read_text(encoding="utf-8")


def _block(text: str, start: str, end: str) -> str:
    i = text.index(start)
    return text[i : text.index(end, i) + len(end)]


def _js_code(source: str) -> str:
    """Strip JS comments so guards match executable code, not prose.

    app.js contains no string literal with '//' or '/*' (its URLs are
    same-origin relative paths), so naive comment removal is safe here.
    """
    no_block_comments = re.sub(r"/\*.*?\*/", "", source, flags=re.S)
    return re.sub(r"//[^\n]*", "", no_block_comments)


# Executable code only: an assertion may not be satisfied by an explanatory
# comment that merely mentions a function name.
JS_CODE = _js_code(JS)


def _function(name: str) -> str:
    return _block(JS_CODE, f"function {name}(", "\n}")


def _change_handler() -> str:
    return _block(JS_CODE, 'els.file.addEventListener("change"', "});")


def _run_detection() -> str:
    return _block(JS_CODE, "async function runDetection(", "\n}")


# ------------------------------------------------------------------- F-001


def test_canvas_and_image_share_one_wrapper():
    """The canvas must be a sibling of the image inside a common container."""
    frame = _block(HTML, '<div id="frame"', "</div>")
    assert 'id="preview"' in frame
    assert 'id="overlay"' in frame


def test_wrapper_is_position_relative():
    frame_rule = _block(CSS, ".frame {", "}")
    assert "position: relative" in frame_rule
    assert "max-width: 100%" in frame_rule


def test_canvas_pinned_to_wrapper_with_inset_zero():
    overlay_rule = _block(CSS, "#overlay {", "}")
    assert "position: absolute" in overlay_rule
    assert re.search(r"^\s*inset:\s*0\s*;", overlay_rule, re.M)
    # No independent centering (the old left/top/transform scheme) remains.
    assert "transform" not in overlay_rule


def test_hiding_never_uses_visibility_property():
    """visibility keeps layout space; display:none removes the element."""
    assert re.search(r"\bvisibility\s*:", CSS) is None
    hidden_rule = _block(CSS, ".hidden {", "}")
    assert "display: none" in hidden_rule


# ------------------------------------------------------------------- F-002


def test_invalidation_runs_before_validation_in_change_handler():
    """The change handler must contain an EXECUTABLE invalidateCurrent()
    statement (a comment is not a call), positioned before both validations."""
    handler = _change_handler()
    call = re.search(r"^\s*invalidateCurrent\(\)\s*;", handler, re.M)
    assert call, "change handler must contain an executable invalidateCurrent() call"
    invalidate_pos = call.start()
    assert invalidate_pos < handler.index("file.type"), (
        "invalidateCurrent() must run before the media-type validation"
    )
    assert invalidate_pos < handler.index("MAX_UPLOAD_BYTES"), (
        "invalidateCurrent() must run before the size validation"
    )


def test_rejected_file_keeps_recoverable_error_message():
    handler = _change_handler()
    assert handler.count("rejectFile(") == 2  # type and size rejection branches
    assert "Please choose a JPEG or PNG image file." in handler
    assert "8 MiB upload limit" in handler
    assert "showError(" in _function("rejectFile")


# ------------------------------------------------------------------- F-003


def test_invalidation_clears_loading_state():
    """The aborted request's finally block is seq-guarded and will not run,
    so invalidateCurrent() must clear the loading indicator itself."""
    invalidate = _function("invalidateCurrent")
    assert re.search(r"^\s*stopLoading\(\)\s*;", invalidate, re.M), (
        "invalidateCurrent() must call stopLoading()"
    )
    stop = _function("stopLoading")
    assert re.search(r"^\s*els\.loading\.classList\.add\(\"hidden\"\)\s*;", stop, re.M)
    # Detect button restored to the state /health last reported, never
    # force-enabled against a not-ready service.
    assert "setServiceEnabled(serviceReady)" in stop


def test_completed_request_still_cleans_up_loading_state():
    """Successful/failed-but-completed requests keep their own cleanup,
    still guarded by the seq check so an aborted request cannot double-run
    it (or clear a newer request's state)."""
    finally_block = _block(_run_detection(), "finally {", "}")
    assert re.search(r"stopLoading\(\)\s*;", finally_block)
    assert "mySeq === seq" in finally_block


def test_rejected_file_is_cleared_from_input():
    """Detect reads els.file.files[0]; a rejected file must not survive in
    the input, or it could be submitted over the previous image's preview."""
    handler = _change_handler()
    assert handler.count("rejectFile(") == 2
    reject = _function("rejectFile")
    assert re.search(r"^\s*els\.file\.value\s*=\s*\"\"\s*;", reject, re.M)


def test_detect_ignores_absent_file():
    """With the input cleared after a rejection, Detect must be a no-op
    rather than submitting anything stale."""
    detect = _run_detection()
    assert "const file = els.file.files[0];" in detect
    assert re.search(r"^\s*if \(!file\) return;\s*$", detect, re.M)
