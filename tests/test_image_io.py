"""Unit tests for upload size limits and untrusted-image decoding."""

from __future__ import annotations

import asyncio
import io

import pytest
from PIL import Image
from starlette.datastructures import UploadFile

from backend.image_io import (
    DecodedImageTooLarge,
    InvalidImage,
    UnsupportedFormat,
    UploadTooLarge,
    decode_image,
    read_upload,
)


def _save(img: Image.Image, fmt: str) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


def _run(coro):
    return asyncio.run(coro)


# ------------------------------------------------------------------ decoding


def test_valid_jpeg_decoded_to_rgb():
    img = Image.new("RGB", (64, 32), (1, 2, 3))
    data = _save(img, "JPEG")
    out = decode_image(data, max_pixels=10_000)
    assert out.mode == "RGB"
    assert out.size == (64, 32)


def test_valid_png_decoded_to_rgb():
    img = Image.new("RGB", (40, 20), (9, 9, 9))
    data = _save(img, "PNG")
    out = decode_image(data, max_pixels=10_000)
    assert out.mode == "RGB"
    assert out.size == (40, 20)


def test_rgba_alpha_composited_over_white():
    img = Image.new("RGBA", (8, 8), (255, 0, 0, 0))  # fully transparent red
    data = _save(img, "PNG")
    out = decode_image(data, max_pixels=1000)
    assert out.mode == "RGB"
    # Alpha 0 => white background shows through.
    assert out.getpixel((0, 0)) == (255, 255, 255)


def test_greyscale_converted_to_rgb():
    img = Image.new("L", (16, 16), 128)
    data = _save(img, "PNG")
    out = decode_image(data, max_pixels=1000)
    assert out.mode == "RGB"


def test_exif_orientation_applied():
    # A 200x100 image tagged to be rotated 90deg CW (orientation 6) must be
    # returned as 100x200 so coordinates match the displayed image.
    img = Image.new("RGB", (200, 100), (200, 20, 20))
    exif = img.getexif()
    exif[0x0112] = 6  # Orientation = "rotate 90 CW"
    data = _save(img, "JPEG")
    data = _inject_exif(data, exif)
    out = decode_image(data, max_pixels=100_000)
    assert out.size == (100, 200)


def _inject_exif(jpeg_bytes: bytes, exif) -> bytes:
    buf = io.BytesIO()
    Image.open(io.BytesIO(jpeg_bytes)).save(buf, format="JPEG", exif=exif.tobytes())
    return buf.getvalue()


@pytest.mark.parametrize("fmt", ["GIF", "WEBP", "TIFF", "BMP"])
def test_unsupported_formats_rejected(fmt):
    img = Image.new("RGB", (16, 16), (0, 0, 0))
    data = _save(img, fmt)
    with pytest.raises(UnsupportedFormat):
        decode_image(data, max_pixels=10_000)


def test_garbage_bytes_rejected():
    with pytest.raises(InvalidImage):
        decode_image(b"not an image at all", max_pixels=10_000)


def test_empty_bytes_rejected():
    with pytest.raises(InvalidImage):
        decode_image(b"", max_pixels=10_000)


def test_truncated_jpeg_rejected():
    img = Image.new("RGB", (300, 300), (10, 20, 30))
    data = _save(img, "JPEG")
    with pytest.raises(InvalidImage):
        decode_image(data[: int(len(data) * 0.3)], max_pixels=1_000_000)


def test_animated_gif_rejected():
    frames = [Image.new("RGB", (16, 16), (i * 40, 0, 0)) for i in range(3)]
    buf = io.BytesIO()
    frames[0].save(buf, format="GIF", save_all=True, append_images=frames[1:])
    data = buf.getvalue()
    # GIF is not in the allowlist, so it is rejected as unsupported either way.
    with pytest.raises(UnsupportedFormat):
        decode_image(data, max_pixels=10_000)


def test_decoded_pixel_limit_enforced():
    img = Image.new("RGB", (400, 400), (0, 0, 0))  # 160_000 px
    data = _save(img, "PNG")
    with pytest.raises(DecodedImageTooLarge):
        decode_image(data, max_pixels=100_000)


def test_decoded_pixel_limit_allows_under():
    img = Image.new("RGB", (300, 300), (0, 0, 0))  # 90_000 px
    data = _save(img, "PNG")
    out = decode_image(data, max_pixels=100_000)
    assert out.size == (300, 300)


# ---------------------------------------------------------------- upload read


def test_read_upload_within_limit():
    async def _go():
        f = UploadFile(io.BytesIO(b"x" * 100), filename="a.jpg")
        return await read_upload(f, max_bytes=1000)

    assert _run(_go()) == b"x" * 100


def test_read_upload_enforces_limit():
    async def _go():
        f = UploadFile(io.BytesIO(b"x" * 5000), filename="a.jpg")
        return await read_upload(f, max_bytes=1000)

    with pytest.raises(UploadTooLarge):
        _run(_go())
