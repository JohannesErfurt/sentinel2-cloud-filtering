"""The three band tests behind the ``threshold`` backend (B2.1, SPEC.md 3.1).

All three take offset-corrected reflectance, already on a common grid -- this
module does no reading or resampling of its own, so it can be tested against
plain synthetic arrays with no product involved.
"""
from __future__ import annotations

import numpy as np


def brightness(b02: np.ndarray, b03: np.ndarray, b04: np.ndarray) -> np.ndarray:
    """Mean visible reflectance. Thick cloud is bright across the spectrum."""
    return (b02 + b03 + b04) / np.float32(3.0)


def ndsi(b03: np.ndarray, b11: np.ndarray) -> np.ndarray:
    """Normalised difference between green (B03) and shortwave IR (B11).

    Not a snow test here (this scene has none): a spectral-slope measure.
    Cloud is spectrally flat, so its NDSI sits near zero; vegetation and soil
    get brighter toward the SWIR (negative); water and shadow absorb it
    (positive). See SPEC.md 3.1 for the class-mean reference table.

    The zero denominator (both bands exactly 0) is guarded to return 0 rather
    than raising or emitting ``inf`` -- it happens only off the edge of the
    valid footprint, never on real cloud or ground.
    """
    b03 = b03.astype(np.float32, copy=False)
    b11 = b11.astype(np.float32, copy=False)
    denominator = b03 + b11
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.where(denominator != 0, (b03 - b11) / denominator, np.float32(0.0))
    return result.astype(np.float32, copy=False)


def cirrus_flag(b10: np.ndarray, t_cirrus: float) -> np.ndarray:
    """B10 sits in a water-vapour absorption band: blind to the ground, so
    anything it sees is high in the atmosphere. The ratio to a clear-sky
    floor is the signal, not the absolute reflectance (SPEC.md 3.1)."""
    return b10 > np.float32(t_cirrus)


def combined_cloud_mask(
    b02: np.ndarray,
    b03: np.ndarray,
    b04: np.ndarray,
    b11: np.ndarray,
    b10: np.ndarray,
    t_bright: float,
    t_ndsi: float,
    t_cirrus: float,
) -> np.ndarray:
    """``cloud = (brightness > T_bright AND NDSI > T_ndsi) OR (B10 > T_cirrus)``.

    The NDSI veto removes bright pixels whose spectrum is too SWIR-heavy to be
    a clean cloud pixel ("whiteness veto", not a snow test); B10 adds cirrus
    the other two miss. Returns ``bool``.
    """
    bright_mask = brightness(b02, b03, b04) > np.float32(t_bright)
    ndsi_mask = ndsi(b03, b11) > np.float32(t_ndsi)
    cirrus_mask = cirrus_flag(b10, t_cirrus)
    return (bright_mask & ndsi_mask) | cirrus_mask


__all__ = ["brightness", "ndsi", "cirrus_flag", "combined_cloud_mask"]
