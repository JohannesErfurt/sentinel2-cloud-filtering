"""Band reading, reflectance conversion and resampling (F1, F2).

Two rules from SPEC.md 1.2 are enforced here rather than left to callers:

* **Down is a block mean.** ``cv2.INTER_AREA`` and an exact reshape-and-mean
  agree to float32 rounding. GDAL's decimated ``Resampling.average`` does *not*
  -- it reads the JPEG-2000 codestream's own reduced-resolution levels, which
  differ from a block mean of the full-resolution pixels by up to 8481 DN on
  this product. It is fast and wrong, so it is not used.
* **Up is nearest for labels, and nearest by default for everything.** Nearest
  upsampling is window-independent: a tile read on its own is bit-identical to
  the same tile cut out of a scene-wide read. Linear interpolation is not,
  unless a halo is read around the window, which ``read_on_10m_grid`` does when
  asked for it.
"""
from __future__ import annotations

import math

import cv2
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.windows import Window

from .constants import BAND_RESOLUTION
from .metadata import ProductMetadata

#: Rows of the source raster to process at a time when downsampling a whole
#: band. Keeps peak memory at a few tens of MB instead of ~1 GB for a 10980^2
#: float32 array.
_STRIPE_ROWS = 1098


# ----------------------------------------------------------------- resampling
def block_mean(array: np.ndarray, factor: int) -> np.ndarray:
    """Average non-overlapping ``factor`` x ``factor`` blocks. Exact.

    The three Sentinel-2 grids share an upper-left corner, so a 60 m pixel
    covers exactly a 6 x 6 block of 10 m pixels and this is the correct
    downsample, not an approximation of one.
    """
    if factor == 1:
        return array
    height, width = array.shape[:2]
    if height % factor or width % factor:
        raise ValueError(
            "block_mean needs a shape divisible by %d, got %dx%d" % (factor, height, width)
        )
    working = array.astype(np.float32, copy=False)
    return working.reshape(height // factor, factor, width // factor, factor).mean(
        axis=(1, 3), dtype=np.float32
    )


def upsample_nearest(array: np.ndarray, factor: int) -> np.ndarray:
    """Repeat every pixel ``factor`` times in both axes, preserving dtype.

    This is exactly ``np.repeat(np.repeat(a, f, 0), f, 1)``. It never invents a
    value that was not in the input, which is what makes it the only correct
    choice for a class label or a boolean mask.
    """
    if factor == 1:
        return array
    return np.repeat(np.repeat(array, factor, axis=0), factor, axis=1)


def resize_linear(array: np.ndarray, out_shape: tuple[int, int]) -> np.ndarray:
    """Bilinear resize for continuous data. Not window-independent."""
    height, width = out_shape
    return cv2.resize(
        array.astype(np.float32, copy=False), (width, height), interpolation=cv2.INTER_LINEAR
    )


# --------------------------------------------------------------- band reading
def _scale_to_10m(band: str) -> int:
    resolution = BAND_RESOLUTION[band]
    if resolution % 10:
        raise ValueError("band %s has resolution %d m, not a multiple of 10" % (band, resolution))
    return resolution // 10


def read_dn(
    meta: ProductMetadata,
    band: str,
    window: Window | None = None,
) -> np.ndarray:
    """Raw digital numbers from one band, on the band's own native grid."""
    with rasterio.open(meta.band_path(band)) as src:
        return src.read(1, window=window)


def to_reflectance(
    dn: np.ndarray,
    meta: ProductMetadata,
    band: str,
    apply_offset: bool = True,
) -> np.ndarray:
    """Convert DN to top-of-atmosphere reflectance.

    ``reflectance = (dn + RADIO_ADD_OFFSET) / QUANTIFICATION_VALUE``

    ``apply_offset=False`` exists only so tests can demonstrate what the
    omission costs (F10); no production path uses it. Genuinely dark surfaces
    can land slightly below zero after the offset -- that is sensor noise, and
    values are not clipped here.
    """
    offset = meta.offset(band) if apply_offset else 0.0
    return (dn.astype(np.float32) + np.float32(offset)) / np.float32(meta.quantification_value)


def read_reflectance(
    meta: ProductMetadata,
    band: str,
    window: Window | None = None,
    out_shape: tuple[int, int] | None = None,
    apply_offset: bool = True,
) -> np.ndarray:
    """Offset-corrected float32 reflectance on the band's native grid.

    With ``out_shape`` smaller than the source, the result is block-averaged
    (SPEC.md 1.2); the reduction factor must be integral. Whole-band
    downsampling is streamed in stripes to keep peak memory low.
    """
    if out_shape is None:
        return to_reflectance(read_dn(meta, band, window), meta, band, apply_offset)

    with rasterio.open(meta.band_path(band)) as src:
        source_height = src.height if window is None else int(window.height)
        source_width = src.width if window is None else int(window.width)
        out_height, out_width = out_shape
        if out_height > source_height or out_width > source_width:
            raise ValueError(
                "out_shape %s is larger than the source %s; use read_on_10m_grid to upsample"
                % (out_shape, (source_height, source_width))
            )
        if source_height % out_height or source_width % out_width:
            raise ValueError(
                "out_shape %s does not divide the source %s evenly"
                % (out_shape, (source_height, source_width))
            )
        factor_y = source_height // out_height
        factor_x = source_width // out_width
        if factor_y != factor_x:
            raise ValueError("anisotropic downsample %dx%d is not supported" % (factor_y, factor_x))

        if window is not None:
            dn = src.read(1, window=window)
            return block_mean(to_reflectance(dn, meta, band, apply_offset), factor_y)

        # Stream the whole band in stripes aligned to the block factor.
        stripe_rows = max(factor_y, (_STRIPE_ROWS // factor_y) * factor_y)
        result = np.empty(out_shape, dtype=np.float32)
        for top in range(0, source_height, stripe_rows):
            rows = min(stripe_rows, source_height - top)
            chunk = src.read(1, window=Window(0, top, source_width, rows))
            reduced = block_mean(to_reflectance(chunk, meta, band, apply_offset), factor_y)
            result[top // factor_y : top // factor_y + reduced.shape[0]] = reduced
        return result


def read_on_10m_grid(
    meta: ProductMetadata,
    band: str,
    window: Window | None = None,
    interpolation: str = "nearest",
    apply_offset: bool = True,
) -> np.ndarray:
    """Reflectance for ``window`` expressed on the **10 m** grid.

    ``window`` is always in 10 m pixel coordinates, whatever the band's native
    resolution. A 549-pixel tile is 274.5 pixels at 20 m and 91.5 at 60 m, so
    for coarse bands the window does not land on pixel boundaries: the source
    window is expanded outwards to integers, read, upsampled, and cropped back
    to the exact requested extent.

    ``interpolation='nearest'`` (the default) makes a windowed read bit-identical
    to the same crop of a scene-wide read. ``'linear'`` reads a one-pixel halo so
    that it stays window-independent too, but it is still smoother than a scene
    edge, where OpenCV replicates.
    """
    scale = _scale_to_10m(band)
    if scale == 1:
        return read_reflectance(meta, band, window, apply_offset=apply_offset)

    native_height, native_width = meta.shape(BAND_RESOLUTION[band])
    if window is None:
        col_off, row_off = 0, 0
        width, height = meta.shape(10)[1], meta.shape(10)[0]
    else:
        col_off, row_off = int(window.col_off), int(window.row_off)
        width, height = int(window.width), int(window.height)

    halo = 1 if interpolation == "linear" else 0
    src_col0 = max(0, math.floor(col_off / scale) - halo)
    src_row0 = max(0, math.floor(row_off / scale) - halo)
    src_col1 = min(native_width, math.ceil((col_off + width) / scale) + halo)
    src_row1 = min(native_height, math.ceil((row_off + height) / scale) + halo)

    native_window = Window(src_col0, src_row0, src_col1 - src_col0, src_row1 - src_row0)
    patch = read_reflectance(meta, band, native_window, apply_offset=apply_offset)

    out_shape = (patch.shape[0] * scale, patch.shape[1] * scale)
    if interpolation == "linear":
        enlarged = resize_linear(patch, out_shape)
    elif interpolation == "nearest":
        enlarged = upsample_nearest(patch, scale)
    else:
        raise ValueError("interpolation must be 'nearest' or 'linear', got %r" % interpolation)

    top = row_off - src_row0 * scale
    left = col_off - src_col0 * scale
    return enlarged[top : top + height, left : left + width]


def read_nodata_mask(
    meta: ProductMetadata,
    band: str = "B02",
    window: Window | None = None,
) -> np.ndarray:
    """Boolean no-data mask on the 10 m grid.

    The product declares ``SPECIAL_VALUE_INDEX`` 0 for NODATA. B02 is the
    default probe because it is a native 10 m band, so the mask needs no
    resampling at all.
    """
    scale = _scale_to_10m(band)
    if scale == 1:
        return read_dn(meta, band, window) == meta.nodata_value

    native_window = None
    if window is not None:
        native_window = Window(
            math.floor(int(window.col_off) / scale),
            math.floor(int(window.row_off) / scale),
            math.ceil(int(window.width) / scale),
            math.ceil(int(window.height) / scale),
        )
    patch = read_dn(meta, band, native_window) == meta.nodata_value
    enlarged = upsample_nearest(patch, scale)
    if window is None:
        return enlarged
    top = int(window.row_off) - math.floor(int(window.row_off) / scale) * scale
    left = int(window.col_off) - math.floor(int(window.col_off) / scale) * scale
    return enlarged[top : top + int(window.height), left : left + int(window.width)]


def read_tci(
    meta: ProductMetadata,
    window: Window | None = None,
    out_shape: tuple[int, int] | None = None,
) -> np.ndarray:
    """True-colour image as ``(height, width, 3)`` uint8 in **RGB** order.

    TCI is a display product: it is already 8-bit and contrast-stretched. It is
    the source for the output JPEGs and must never be used for detection
    (SPEC.md 1.6).

    With ``out_shape`` smaller than the source, GDAL's fast decimated read is
    used rather than the exact block mean in :func:`read_reflectance`. That
    approximation matters for a *quantitative* band feeding a detector -- it
    can be off by thousands of DN (SPEC.md 1.2) -- but this output is a
    background image in a viewer, never a detection input, so the speed is
    worth it and a few grey levels of difference are invisible.
    """
    with rasterio.open(meta.tci_path) as src:
        out = (3,) + tuple(out_shape) if out_shape else None
        return np.transpose(
            src.read(window=window, out_shape=out, resampling=Resampling.average), (1, 2, 0)
        )


__all__ = [
    "block_mean",
    "upsample_nearest",
    "resize_linear",
    "read_dn",
    "to_reflectance",
    "read_reflectance",
    "read_on_10m_grid",
    "read_nodata_mask",
    "read_tci",
]
