"""Source-level regression guards for the browser demo (WO-001R, F-001/F-002).

No browser or JS runtime exists in the approved execution environment, so
real-browser geometry cannot be automated here (limitation reported in the
agent report). These tests pin the two corrective invariants at the source
level so a regression cannot silently return:

F-001: the overlay canvas lives inside the same position:relative wrapper as
       the image and is pinned with inset:0, and hidden elements use
       display (never the visibility property) so hidden elements can never
       occupy layout space or displace the displayed image.
F-002: file-change invalidation runs BEFORE client-side validation returns,
       so selecting an invalid or oversized file cannot leave the previous
       image's detections visible.
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
    handler = _block(JS, 'els.file.addEventListener("change"', "});")
    invalidate_pos = handler.index("invalidateCurrent()")
    assert invalidate_pos < handler.index("file.type"), (
        "invalidateCurrent() must run before the media-type validation"
    )
    assert invalidate_pos < handler.index("MAX_UPLOAD_BYTES"), (
        "invalidateCurrent() must run before the size validation"
    )


def test_rejected_file_keeps_recoverable_error_message():
    handler = _block(JS, 'els.file.addEventListener("change"', "});")
    assert handler.count("showError(") >= 2  # type and size rejection messages
