"""The ``s2cloudless`` backend: a pretrained LightGBM classifier. Task B3.

The ML bonus (R7). **Gradient-boosted trees, not a deep neural network** --
it satisfies "Machine Learning", not strictly "Deep Learning" (SPEC.md 4.1).

The model ships inside the ``s2cloudless`` package (no download step) and
takes all 13 Sentinel-2 bands in ``S2_BANDS`` order, which is exactly
``pipeline.constants.BANDS`` -- asserted below rather than assumed, since a
silent reordering would misfeed every band into the wrong model slot.
"""
from __future__ import annotations

import json
import os

import cv2
import numpy as np

from ..constants import BANDS, SCENE_PX
from ..io import read_reflectance, upsample_nearest
from ..metadata import ProductMetadata
from . import Detector, register

MASK_60M_SIZE = SCENE_PX // 6  # 1830

#: The library's own constructor defaults (S2PixelCloudDetector), used only
#: if config/s2cloudless.json is absent *and* no CLI override was given.
#: Day-one usability, not the deliberate choice B3.3 asks for -- that choice
#: is at 0.6 / none / none, recorded in config/s2cloudless.json (§4.3, D6).
LIBRARY_DEFAULTS = {"prob_threshold": 0.4, "average_over": 1, "dilation_size": 1}

_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "config",
    "s2cloudless.json",
)


class S2CloudlessConfigError(RuntimeError):
    """``config/s2cloudless.json`` is missing a required key or is malformed."""


def load_config(path: str | None = None) -> dict:
    """Read ``config/s2cloudless.json``, or the library defaults if absent.

    Mirrors ``pipeline.detectors.threshold.load_config``: a missing file is
    fine (day-one usability), a malformed one raises rather than silently
    falling back on a typo.
    """
    path = path or _CONFIG_PATH
    if not os.path.isfile(path):
        return dict(LIBRARY_DEFAULTS)
    with open(path, encoding="utf8") as handle:
        try:
            data = json.load(handle)
        except json.JSONDecodeError as error:
            raise S2CloudlessConfigError("%s is not valid JSON: %s" % (path, error)) from None
    missing = [k for k in ("prob_threshold", "average_over", "dilation_size") if k not in data]
    if missing:
        raise S2CloudlessConfigError("%s is missing %s" % (path, missing))
    out = {"prob_threshold": float(data["prob_threshold"])}
    for key in ("average_over", "dilation_size"):
        out[key] = None if data[key] is None else int(data[key])
    return out


def _none_if_zero(value: int | None) -> int | None:
    """CLI flags are plain ints (``argparse`` has no clean way to pass
    ``None``), so ``0`` is the command-line spelling of "off" -- matching the
    ``--average-over``/``--dilation-size`` help text in ``pipeline.run``."""
    return None if value == 0 else value


def mask_from_probability(
    prob: np.ndarray, threshold: float, average_over: int | None, dilation_size: int | None
) -> np.ndarray:
    """Threshold, then optional averaging and dilation -- the same rule
    ``S2PixelCloudDetector.get_mask_from_prob`` implements, reimplemented to
    route around a real bug in it.

    The library builds its intermediate mask as ``int8`` when ``average_over``
    is falsy (``(cloud_probs > threshold).astype(np.int8)``), and on this
    OpenCV build (5.0.0) ``cv2.dilate`` refuses a signed-integer input
    ("Unsupported data type (=1)"). Any call with ``average_over`` off and a
    real ``dilation_size`` crashes -- reachable from the CLI via
    ``--average-over 0 --dilation-size 2``, and required by B3.3's own sweep
    grid, which must cover exactly that combination. Verified to give
    identical output to the library's own method wherever the library does
    not crash (a uint8 mask throughout is the only difference).
    """
    from s2cloudless.utils import cv2_disk

    if average_over:
        disk = cv2_disk(average_over)
        conv_filter = disk / np.sum(disk)
        mask = (cv2.filter2D(prob, -1, conv_filter, borderType=cv2.BORDER_REFLECT) > threshold).astype(np.uint8)
    else:
        mask = (prob > threshold).astype(np.uint8)
    if dilation_size:
        mask = cv2.dilate(mask, cv2_disk(dilation_size))
    return mask.astype(bool)


def build_reflectance_stack(meta: ProductMetadata, shape: tuple[int, int] = (MASK_60M_SIZE, MASK_60M_SIZE)) -> np.ndarray:
    """All 13 bands, offset-corrected, block-averaged to ``shape``, stacked in
    ``S2_BANDS`` order -- the layout ``all_bands=True`` expects.

    Reflectance is clipped at 0: the model was trained on values that never
    went negative, and this scene's own sensor noise can dip slightly below
    zero after the offset (SPEC.md 1.1).
    """
    stack = np.stack([read_reflectance(meta, band, out_shape=shape) for band in BANDS], axis=-1)
    return np.clip(stack, 0.0, None)


@register
class S2CloudlessDetector(Detector):
    """A pretrained gradient-boosted cloud classifier, run at 60 m."""

    name = "s2cloudless"
    description = "s2cloudless: pretrained LightGBM classifier (ML bonus, not a neural network)."

    def __init__(
        self,
        prob_threshold: float | None = None,
        average_over: int | None = None,
        dilation_size: int | None = None,
        config_path: str | None = None,
    ):
        config = load_config(config_path)
        self.prob_threshold = prob_threshold if prob_threshold is not None else config["prob_threshold"]
        self.average_over = _none_if_zero(average_over) if average_over is not None else config["average_over"]
        self.dilation_size = _none_if_zero(dilation_size) if dilation_size is not None else config["dilation_size"]

        self._prob60: np.ndarray | None = None
        self._cloud60: np.ndarray | None = None
        self._cloud10: np.ndarray | None = None

    def _ensure_loaded(self, meta: ProductMetadata) -> None:
        if self._cloud10 is not None:
            return

        # Imported lazily: s2cloudless pulls in lightgbm and sentinelhub, and
        # nothing else in this project needs either unless this backend runs.
        from s2cloudless import S2PixelCloudDetector
        from s2cloudless.utils import S2_BANDS

        if list(S2_BANDS) != list(BANDS):
            raise AssertionError(
                "s2cloudless.utils.S2_BANDS %r no longer matches pipeline.constants.BANDS %r -- "
                "the 13-band stack would be fed to the model in the wrong order" % (S2_BANDS, BANDS)
            )

        stack = build_reflectance_stack(meta)
        # Only all_bands=True matters for inference; threshold/average_over/
        # dilation_size are applied afterward via mask_from_probability, not
        # through this instance (see that function's docstring for why).
        model = S2PixelCloudDetector(all_bands=True)
        self._prob60 = model.get_cloud_probability_maps(stack[None, ...])[0]
        self._cloud60 = mask_from_probability(
            self._prob60, self.prob_threshold, self.average_over, self.dilation_size
        )
        # Upsample *before* cutting tiles, exactly as the esa backend does:
        # cutting the 60 m mask directly would put a tile edge half-way
        # through a pixel (91.5 px) and shift the counts.
        self._cloud10 = upsample_nearest(self._cloud60, 6)

    def tile_mask(self, meta: ProductMetadata, row: int, col: int) -> np.ndarray:
        self._ensure_loaded(meta)
        return self.cut(self._cloud10, row, col)

    def tile_extra(self, row: int, col: int) -> dict:
        """Mean cloud probability per tile (decision D8): ``cloud_cover_percent``
        in report.csv is the hard-mask fraction; the soft alternative lives
        here, in tile_stats.csv, so both definitions can be compared."""
        if self._prob60 is None:
            return {}
        from ..constants import TILE_METRES

        tile_px_60m = TILE_METRES / 60.0  # 91.5
        y0, y1 = round(row * tile_px_60m), round((row + 1) * tile_px_60m)
        x0, x1 = round(col * tile_px_60m), round((col + 1) * tile_px_60m)
        return {"mean_probability": float(self._prob60[y0:y1, x0:x1].mean())}

    def scene_layers(self, meta: ProductMetadata) -> tuple[dict[str, np.ndarray], int]:
        """The hard mask at 60 m, for the GeoJSON bonus (SPEC.md 4.6/B3.4)."""
        self._ensure_loaded(meta)
        return {"cloud": self._cloud60}, 60

    def write_scene_artifacts(self, meta: ProductMetadata, out_dir: str) -> None:
        """``cloud_probability_60m.npy`` (B3.2): the raw probability map, for
        anyone who wants a threshold or morphology this backend did not ship
        with, without re-running the (comparatively expensive) model."""
        self._ensure_loaded(meta)
        os.makedirs(out_dir, exist_ok=True)
        np.save(os.path.join(out_dir, "cloud_probability_60m.npy"), self._prob60.astype(np.float32))

    def parameters(self) -> dict:
        return {
            "prob_threshold": self.prob_threshold,
            "average_over": self.average_over,
            "dilation_size": self.dilation_size,
        }

    def close(self) -> None:
        self._prob60 = self._cloud60 = self._cloud10 = None


__all__ = [
    "S2CloudlessDetector",
    "S2CloudlessConfigError",
    "load_config",
    "LIBRARY_DEFAULTS",
    "build_reflectance_stack",
    "mask_from_probability",
]
