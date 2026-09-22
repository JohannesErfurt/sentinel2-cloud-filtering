"""The ``threshold`` backend: brightness / NDSI / B10 band tests. Task B2.

A detector with documented physics and three thresholds chosen by looking,
not by fitting to ESA's mask (SPEC.md 3.2's whole point). See
``pipeline/masks.py`` for the formulas themselves; this module wires them to
the product, to the resolution decision (D2), and to ``config/thresholds.json``.
"""
from __future__ import annotations

import json
import os

import numpy as np
from rasterio.windows import Window

from ..constants import SCENE_PX, TILE_PX
from ..io import read_on_10m_grid, read_reflectance, upsample_nearest
from ..masks import combined_cloud_mask
from ..metadata import ProductMetadata
from . import Detector, register

MASK_60M_SIZE = SCENE_PX // 6  # 1830

#: Starting values SPEC.md 3.2 recommends, used only if config/thresholds.json
#: is absent *and* no CLI override was given -- day-one usability, not a
#: substitute for actually choosing (B2.2).
FALLBACK_THRESHOLDS = {"t_bright": 0.18, "t_ndsi": -0.20, "t_cirrus": 0.005}

_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "config",
    "thresholds.json",
)


class ThresholdConfigError(RuntimeError):
    """``config/thresholds.json`` is missing a required key or is malformed."""


def load_config(path: str | None = None) -> dict:
    """Read ``config/thresholds.json``, or the fallback values if it is absent.

    A missing file is not an error -- it lets the detector run before B2.2's
    choice is recorded -- but a *malformed* one is: silently falling back on a
    typo would be worse than failing loudly.
    """
    path = path or _CONFIG_PATH
    if not os.path.isfile(path):
        return dict(FALLBACK_THRESHOLDS)
    with open(path, encoding="utf8") as handle:
        try:
            data = json.load(handle)
        except json.JSONDecodeError as error:
            raise ThresholdConfigError("%s is not valid JSON: %s" % (path, error)) from None
    missing = [k for k in ("t_bright", "t_ndsi", "t_cirrus") if k not in data]
    if missing:
        raise ThresholdConfigError("%s is missing %s" % (path, missing))
    return {k: float(data[k]) for k in ("t_bright", "t_ndsi", "t_cirrus")}


@register
class ThresholdDetector(Detector):
    """Brightness + NDSI veto + B10 cirrus, combined per SPEC.md 3.1."""

    name = "threshold"
    description = "Brightness/NDSI/B10 band tests with thresholds chosen by inspection."

    def __init__(
        self,
        t_bright: float | None = None,
        t_ndsi: float | None = None,
        t_cirrus: float | None = None,
        detection_resolution: int = 10,
        config_path: str | None = None,
    ):
        if detection_resolution not in (10, 60):
            raise ValueError("detection_resolution must be 10 or 60, got %r" % detection_resolution)
        config = load_config(config_path)
        self.t_bright = t_bright if t_bright is not None else config["t_bright"]
        self.t_ndsi = t_ndsi if t_ndsi is not None else config["t_ndsi"]
        self.t_cirrus = t_cirrus if t_cirrus is not None else config["t_cirrus"]
        self.detection_resolution = detection_resolution
        self._cloud10: np.ndarray | None = None  # only populated at 60 m (cached scene mask)

    # ------------------------------------------------------------- the rule
    def _mask_10m_grid(self, meta: ProductMetadata, window=None) -> np.ndarray:
        """The combined rule evaluated directly on the 10 m grid.

        Brightness uses 10 m bands natively; NDSI (20 m) and B10 (60 m) are
        upsampled with nearest neighbour first -- which is why the cirrus
        branch is blocky in 6x6 squares across its whole extent, not just at
        edges (SPEC.md 3.3). ``window`` is in 10 m pixel coordinates; omitting
        it reads the whole scene, which is expensive (five ~482 MB arrays) and
        exists only for the cross-check in tests and for ``scene_mask``, never
        for the per-tile streaming path ``tile_mask`` uses.
        """
        b02 = read_on_10m_grid(meta, "B02", window)
        b03 = read_on_10m_grid(meta, "B03", window)
        b04 = read_on_10m_grid(meta, "B04", window)
        b11 = read_on_10m_grid(meta, "B11", window, interpolation="nearest")
        b10 = read_on_10m_grid(meta, "B10", window, interpolation="nearest")
        return combined_cloud_mask(b02, b03, b04, b11, b10, self.t_bright, self.t_ndsi, self.t_cirrus)

    def _mask_60m_grid(self, meta: ProductMetadata) -> np.ndarray:
        """The combined rule evaluated once on the whole scene at 60 m.

        The alternative D2 answer: simpler, reads every band on one grid, and
        cheap enough (five block-mean reads of a 1830x1830 array) to hold in
        memory whole rather than streaming. Used when
        ``detection_resolution=60``, and always for :meth:`scene_layers`
        regardless of that setting -- the GeoJSON bonus is vectorised at 60 m
        either way (SPEC.md 1.7), so there is no reason to pay for the 10 m
        computation just to immediately downsample it again.
        """
        shape = (MASK_60M_SIZE, MASK_60M_SIZE)
        b02 = read_reflectance(meta, "B02", out_shape=shape)
        b03 = read_reflectance(meta, "B03", out_shape=shape)
        b04 = read_reflectance(meta, "B04", out_shape=shape)
        b11 = read_reflectance(meta, "B11", out_shape=shape)
        b10 = read_reflectance(meta, "B10", out_shape=shape)
        return combined_cloud_mask(b02, b03, b04, b11, b10, self.t_bright, self.t_ndsi, self.t_cirrus)

    def _ensure_60m_loaded(self, meta: ProductMetadata) -> None:
        if self._cloud10 is not None:
            return
        cloud60 = self._mask_60m_grid(meta)
        self._cloud10 = upsample_nearest(cloud60, 6)

    # --------------------------------------------------------- Detector API
    def tile_mask(self, meta: ProductMetadata, row: int, col: int) -> np.ndarray:
        if self.detection_resolution == 60:
            self._ensure_60m_loaded(meta)
            return self.cut(self._cloud10, row, col)

        window = Window(col * TILE_PX, row * TILE_PX, TILE_PX, TILE_PX)
        return self._mask_10m_grid(meta, window)

    def scene_mask(self, meta: ProductMetadata, window=None) -> np.ndarray:
        """The combined rule computed directly over ``window`` (default: the
        whole scene), independent of the per-tile streaming path.

        Not used by the pipeline's default run -- it exists so the streaming
        path can be checked against an independently-computed reference
        (SPEC.md B2.3) and to back :meth:`scene_layers`. At full scene extent
        with ``detection_resolution=10`` this needs five ~482 MB arrays; call
        it with a bounded ``window`` outside of that specific cross-check.
        """
        if self.detection_resolution == 60:
            cloud60 = self._mask_60m_grid(meta)
            return upsample_nearest(cloud60, 6)
        return self._mask_10m_grid(meta, window)

    def scene_layers(self, meta: ProductMetadata) -> tuple[dict[str, np.ndarray], int]:
        """The mask at 60 m, for the GeoJSON bonus (SPEC.md 1.7)."""
        return {"cloud": self._mask_60m_grid(meta)}, 60

    def parameters(self) -> dict:
        return {
            "t_bright": self.t_bright,
            "t_ndsi": self.t_ndsi,
            "t_cirrus": self.t_cirrus,
            "detection_resolution": self.detection_resolution,
        }

    def close(self) -> None:
        self._cloud10 = None


__all__ = ["ThresholdDetector", "ThresholdConfigError", "load_config", "FALLBACK_THRESHOLDS"]
