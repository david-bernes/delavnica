"""Upload reading, size limits and image decoding for untrusted input.

All uploaded bytes are treated as untrusted: size is enforced while
streaming, the *decoded* format is allowlisted (never the filename
extension), orientation is applied via EXIF, and the result is normalized
to 8-bit RGB before inference.
"""

from __future__ import annotations

import io
import logging

from PIL import Image, ImageOps, UnidentifiedImageError

logger = logging.getLogger(__name__)

ALLOWED_FORMATS = {"JPEG", "PNG"}
_READ_CHUNK_SIZE = 1024 * 1024  # 1 MiB streaming reads


class UploadTooLarge(Exception):
    """Compressed upload exceeds the configured byte limit."""


class UnsupportedFormat(Exception):
    """Decoded image format is not in the allowlist."""


class InvalidImage(Exception):
    """Image is corrupt, truncated, animated or otherwise undecodable."""


class DecodedImageTooLarge(Exception):
    """Decoded image exceeds the configured pixel limit."""


async def read_upload(file, max_bytes: int) -> bytes:
    """Read an ASGI ``UploadFile`` in bounded chunks, enforcing ``max_bytes``.

    The limit is enforced while streaming, so the application never holds
    more than ``max_bytes + 1 chunk`` of upload data in memory even if
    ``Content-Length`` is missing or lies.
    """
    data = bytearray()
    while True:
        chunk = await file.read(_READ_CHUNK_SIZE)
        if not chunk:
            break
        data += chunk
        if len(data) > max_bytes:
            raise UploadTooLarge(
                f"Upload of at least {len(data)} bytes exceeds the "
                f"{max_bytes} byte limit."
            )
    return bytes(data)


def _to_rgb(img: Image.Image) -> Image.Image:
    """Deterministically normalize any supported mode to 8-bit RGB.

    Alpha is composited over white so transparent regions are stable across
    requests; palettes and greyscale are expanded to RGB.
    """
    if img.mode == "P":
        img = img.convert("RGBA" if "transparency" in img.info else "RGB")
    if img.mode in ("RGBA", "LA"):
        background = Image.new("RGB", img.size, (255, 255, 255))
        background.paste(img, mask=img.getchannel("A"))
        return background
    if img.mode != "RGB":
        return img.convert("RGB")
    return img


def decode_image(data: bytes, max_pixels: int) -> Image.Image:
    """Decode and normalize an untrusted image to orientation-corrected RGB.

    Raises
    ------
    InvalidImage
        Empty, corrupt, truncated, unidentified or animated input.
    UnsupportedFormat
        Decoded format is not JPEG or PNG (filename is never trusted).
    DecodedImageTooLarge
        The decoded image exceeds ``max_pixels`` (checked from the header
        *before* full decoding).
    """
    if not data:
        raise InvalidImage("The uploaded file is empty.")
    # Defense in depth: Pillow raises DecompressionBombError above
    # 2 * MAX_IMAGE_PIXELS; the explicit pixel check below is the strict
    # application policy.
    Image.MAX_IMAGE_PIXELS = max_pixels
    try:
        with Image.open(io.BytesIO(data)) as img:
            if img.format not in ALLOWED_FORMATS:
                raise UnsupportedFormat(
                    f"Unsupported image format {img.format!r}; "
                    "only JPEG and PNG are accepted."
                )
            if getattr(img, "n_frames", 1) > 1:
                raise InvalidImage("Animated or multi-frame images are not supported.")
            # Strict pixel limit from the header, before any decoding cost.
            if img.width * img.height > max_pixels:
                raise DecodedImageTooLarge(
                    f"Decoded image {img.width}x{img.height} "
                    f"({img.width * img.height} pixels) exceeds the "
                    f"{max_pixels} pixel limit."
                )
            img.load()  # force full decode; truncation/corruption raises here
            oriented = ImageOps.exif_transpose(img)
            return _to_rgb(oriented)
    except (UnsupportedFormat, InvalidImage, DecodedImageTooLarge):
        raise
    except Image.DecompressionBombError as exc:
        # Pillow trips this when the image is far beyond the limit (2x);
        # semantically it is a size violation, not a corrupt image.
        raise DecodedImageTooLarge(
            f"Decoded image exceeds the {max_pixels} pixel limit."
        ) from exc
    except (
        UnidentifiedImageError,
        OSError,
        ValueError,
        SyntaxError,
    ) as exc:
        raise InvalidImage("The uploaded file is not a valid JPEG or PNG image.") from exc
