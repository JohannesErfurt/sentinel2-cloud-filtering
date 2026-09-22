"""The ``esa`` backend: the product's own ``MSK_CLASSI_B00.jp2`` (60 m). Task B1.

**This is not a detector.** It is the reference every other backend is scored
against, and the cheapest correct answer to R1 and R3-R6: R2 asks the task to
*detect* clouds, and this reads a mask ESA already computed. See SPEC.md
section 2.
"""
from __future__ import annotations

import numpy as np
import rasterio

from ..io import read_reflectance, upsample_nearest
from ..metadata import ProductMetadata
from . import Detector, register


class ClassiMaskError(RuntimeError):
    """``MSK_CLASSI_B00.jp2`` does not look like what this loader expects."""


def read_classi_mask(meta: ProductMetadata) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load ``MSK_CLASSI_B00.jp2`` and split it into ``(opaque, cirrus, snow)``.

    File (== rasterio band) order is opaque cloud, cirrus, snow/ice
    (SPEC.md 2.1). Four structural invariants are checked so a mis-parsed or
    malformed file fails loudly instead of silently mislabelling every tile:

    * three bands at the mask's native 60 m resolution;
    * opaque and cirrus never overlap (`MSK_CLASSI` never double-labels a pixel);
    * ``(opaque | cirrus)`` matches the product's own ``Cloud_Coverage_Assessment``;
    * ``snow`` matches ``Snow_Coverage_Assessment``.

    These catch a channel swapped with a *different* class -- e.g. opaque with
    snow, which is not a symmetric swap and shows up immediately in both
    aggregates. They do **not** catch opaque swapped with cirrus: that swap
    preserves every check here, because the two channels are only ever used
    together as their union. :func:`verify_channel_order_by_brightness`
    catches that case instead.
    """
    with rasterio.open(meta.mask_classi_path) as src:
        if src.count != 3:
            raise ClassiMaskError(
                "%s has %d band(s), expected 3 (opaque, cirrus, snow)"
                % (meta.mask_classi_path, src.count)
            )
        expected_shape = meta.shape(60)
        if (src.height, src.width) != expected_shape:
            raise ClassiMaskError(
                "%s is %dx%d, expected %s (the 60 m grid)"
                % (meta.mask_classi_path, src.height, src.width, expected_shape)
            )
        array = src.read()

    opaque = array[0] > 0
    cirrus = array[1] > 0
    snow = array[2] > 0

    overlap = int(np.count_nonzero(opaque & cirrus))
    if overlap:
        raise ClassiMaskError(
            "%d pixels are flagged both opaque and cirrus; MSK_CLASSI's channels "
            "should never overlap -- this may not be the file this loader expects" % overlap
        )

    cloud_percent = 100.0 * float((opaque | cirrus).mean())
    if abs(cloud_percent - meta.cloud_coverage_assessment) > 5e-3:
        raise ClassiMaskError(
            "opaque|cirrus is %.4f%% of the scene but Cloud_Coverage_Assessment says "
            "%.4f%% -- the channels may be swapped with snow, or this is the wrong file"
            % (cloud_percent, meta.cloud_coverage_assessment)
        )

    snow_percent = 100.0 * float(snow.mean())
    if abs(snow_percent - meta.snow_coverage_assessment) > 5e-3:
        raise ClassiMaskError(
            "the snow channel is %.4f%% of the scene but Snow_Coverage_Assessment says %.4f%%"
            % (snow_percent, meta.snow_coverage_assessment)
        )

    return opaque, cirrus, snow


def verify_channel_order_by_brightness(
    opaque: np.ndarray, cirrus: np.ndarray, brightness: np.ndarray
) -> None:
    """Confirm opaque cloud is visibly brighter than cirrus.

    The checks in :func:`read_classi_mask` cannot tell opaque and cirrus apart
    from *each other*: swapping them preserves every aggregate those checks
    look at. This is the second, independent confirmation SPEC.md 2.1
    describes: measured on the real product, mean visible brightness
    ((B02+B03+B04)/3, offset-corrected reflectance -- never the display-stretched
    TCI) is 0.415 inside opaque against 0.189 inside cirrus. A swap would
    reverse that inequality.
    """
    if not opaque.any() or not cirrus.any():
        return  # nothing to compare; not itself a sign of a swap
    opaque_mean = float(brightness[opaque].mean())
    cirrus_mean = float(brightness[cirrus].mean())
    if opaque_mean <= cirrus_mean:
        raise ClassiMaskError(
            "mean brightness inside 'opaque' (%.3f) is not above 'cirrus' (%.3f) -- "
            "the two channels look swapped" % (opaque_mean, cirrus_mean)
        )


@register
class EsaDetector(Detector):
    """ESA's shipped classification mask, upsampled to 10 m and cut into tiles."""

    name = "esa"
    description = "ESA's shipped MSK_CLASSI_B00.jp2 (60 m) -- reference baseline, not a detector."

    def __init__(self, verify_channels: bool = True, **kwargs):
        if kwargs:
            raise TypeError("esa detector takes no parameters, got %s" % sorted(kwargs))
        #: Cross-check opaque/cirrus by brightness (SPEC.md 2.1). Reads three
        #: extra bands, so it is exposed as a constructor flag rather than
        #: hard-wired: production leaves it on; unit tests that only exercise
        #: the mask-cutting logic on a bare synthetic mask -- with no band
        #: files behind it -- turn it off rather than fabricating imagery.
        self._verify_channels = verify_channels
        self._opaque60: np.ndarray | None = None
        self._cirrus60: np.ndarray | None = None
        self._snow60: np.ndarray | None = None
        self._cloud10: np.ndarray | None = None

    def _ensure_loaded(self, meta: ProductMetadata) -> None:
        if self._cloud10 is not None:
            return

        opaque, cirrus, snow = read_classi_mask(meta)

        if self._verify_channels:
            shape60 = meta.shape(60)
            brightness = (
                read_reflectance(meta, "B02", out_shape=shape60)
                + read_reflectance(meta, "B03", out_shape=shape60)
                + read_reflectance(meta, "B04", out_shape=shape60)
            ) / np.float32(3.0)
            verify_channel_order_by_brightness(opaque, cirrus, brightness)

        self._opaque60, self._cirrus60, self._snow60 = opaque, cirrus, snow
        # Upsample *before* cutting tiles (SPEC.md 2.1): a 549 px tile is
        # 91.5 px at 60 m, so cutting there puts a tile edge half-way through a
        # pixel and shifts the counts (108 invalid tiles instead of 107).
        self._cloud10 = upsample_nearest(opaque | cirrus, 6)

    def tile_mask(self, meta: ProductMetadata, row: int, col: int) -> np.ndarray:
        self._ensure_loaded(meta)
        return self.cut(self._cloud10, row, col)

    def scene_layers(self, meta: ProductMetadata) -> tuple[dict[str, np.ndarray], int]:
        """Opaque and cirrus as separate 60 m layers, for the GeoJSON bonus.

        Kept at the mask's native 60 m rather than upsampled to 10 m: they
        never overlap, so their areas sum to the union's area exactly, and a
        10 m vectorisation of the same 602 polygons would add nothing but file
        size.
        """
        self._ensure_loaded(meta)
        return {"opaque": self._opaque60, "cirrus": self._cirrus60}, 60

    def parameters(self) -> dict:
        return {}

    def close(self) -> None:
        self._opaque60 = self._cirrus60 = self._snow60 = self._cloud10 = None


__all__ = ["ClassiMaskError", "read_classi_mask", "verify_channel_order_by_brightness", "EsaDetector"]
