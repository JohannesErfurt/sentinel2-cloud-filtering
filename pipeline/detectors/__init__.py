"""Detector backends and the registry the pipeline dispatches through.

The foundations (F1-F10) know nothing about how a cloud mask is produced. A
backend supplies exactly one thing -- R2, "detect clouds" -- and everything
downstream is shared:

* ``tile_mask`` returns a 549 x 549 boolean mask on the 10 m grid;
* ``scene_layers`` returns named scene-wide masks for the GeoJSON bonus;
* ``parameters`` returns whatever went into ``run_summary.json``.

The three backends named in SPEC.md 0.4 -- ``esa`` (B1), ``threshold`` (B2) and
``s2cloudless`` (B3) -- register themselves here when implemented. Until then
the registry is empty and ``build`` says so rather than failing obscurely.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from ..constants import TILE_PX
from ..metadata import ProductMetadata


class DetectorError(RuntimeError):
    """A detector was requested that does not exist, or cannot run."""


class Detector(ABC):
    """Base class for a cloud-mask backend."""

    #: Registry key, e.g. "esa".
    name: str = ""
    #: Short line for the README and run_summary.json.
    description: str = ""

    @abstractmethod
    def tile_mask(self, meta: ProductMetadata, row: int, col: int) -> np.ndarray:
        """Boolean cloud mask for one tile, shape ``(549, 549)`` on the 10 m grid.

        Must return ``bool``. A float probability or an interpolated label
        reaching this point is a silent error, and ``compute_tile_stats``
        rejects it.
        """

    def scene_layers(self, meta: ProductMetadata) -> tuple[dict[str, np.ndarray], int]:
        """Named scene-wide boolean masks plus the resolution they are on.

        Returning coarse masks is normal and preferred for the GeoJSON bonus: a
        10 m mask of this scene vectorises to roughly 31 000 polygons.
        """
        raise NotImplementedError("%s does not provide scene layers" % self.name)

    def tile_extra(self, row: int, col: int) -> dict:
        """Extra per-tile values for ``tile_stats.csv``. Empty by default."""
        return {}

    def parameters(self) -> dict:
        """Everything that would change the output, for ``run_summary.json``."""
        return {}

    def write_scene_artifacts(self, meta: ProductMetadata, out_dir: str) -> None:
        """Extra scene-level files a backend wants saved beside the standard
        five outputs -- e.g. s2cloudless's raw probability map (B3.2). A
        no-op by default; called once per run, after the tile loop."""
        return

    def close(self) -> None:
        """Release anything held open. Called when a run finishes."""

    # Convenience for subclasses that build a scene mask once and cut from it.
    @staticmethod
    def cut(scene_mask_10m: np.ndarray, row: int, col: int) -> np.ndarray:
        top, left = row * TILE_PX, col * TILE_PX
        return scene_mask_10m[top : top + TILE_PX, left : left + TILE_PX]


_REGISTRY: dict[str, type[Detector]] = {}


def register(cls: type[Detector]) -> type[Detector]:
    """Class decorator that adds a backend to the registry."""
    if not cls.name:
        raise DetectorError("%s has no name" % cls.__name__)
    _REGISTRY[cls.name] = cls
    return cls


def available() -> list[str]:
    """Registered backend names, sorted."""
    _load_backends()
    return sorted(_REGISTRY)


def build(name: str, **params) -> Detector:
    """Instantiate a registered backend."""
    _load_backends()
    try:
        cls = _REGISTRY[name]
    except KeyError:
        known = ", ".join(sorted(_REGISTRY)) or "none yet"
        raise DetectorError(
            "unknown detector %r (registered: %s). The backends are implemented "
            "as tasks B1 (esa), B2 (threshold) and B3 (s2cloudless) in SPEC.md." % (name, known)
        ) from None
    return cls(**params)


def _load_backends() -> None:
    """Import backend modules if they exist. Absent ones are simply not offered."""
    for module in ("esa", "threshold", "s2cloudless"):
        try:
            __import__("%s.%s" % (__name__, module))
        except ImportError:
            continue


__all__ = ["Detector", "DetectorError", "register", "available", "build"]
