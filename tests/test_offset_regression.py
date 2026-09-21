"""F10 -- the test that catches the pipeline's one silent, total failure mode.

Baseline 04.00 added ``RADIO_ADD_OFFSET = -1000``. Drop it and every reflectance
is 0.1 too high. Nothing raises, nothing looks wrong, and s2cloudless -- trained
before the offset existed -- flags the entire scene as cloud. All 400 tiles are
then discarded, ``tiles/`` is written empty, and the pipeline exits 0.

This runs only with ``--safe-dir`` and takes about a minute.
"""
from __future__ import annotations

import numpy as np
import pytest

from pipeline.constants import BANDS
from pipeline.io import read_reflectance

pytestmark = pytest.mark.needs_product


@pytest.fixture(scope="module")
def reflectance_60m(product):
    """All 13 bands, offset-corrected, block-averaged to the 60 m grid."""
    stack = [read_reflectance(product, band, out_shape=(1830, 1830)) for band in BANDS]
    return np.stack(stack, axis=-1)


def test_the_offset_decides_whether_anything_survives(reflectance_60m):
    s2cloudless = pytest.importorskip("s2cloudless")

    detector = s2cloudless.S2PixelCloudDetector(
        threshold=0.4, all_bands=True, average_over=1, dilation_size=1
    )
    corrected = np.clip(reflectance_60m, 0, None)[None, ...]
    # Skipping the offset is exactly a +0.1 shift (see test_metadata.py).
    naive = corrected + np.float32(0.1)

    with_offset = detector.get_cloud_masks(corrected)[0].mean() * 100
    without_offset = detector.get_cloud_masks(naive)[0].mean() * 100

    assert with_offset == pytest.approx(39.3, abs=1.0), (
        "SPEC.md 4.3: library defaults flag 39.3 %% of this scene, got %.2f %%" % with_offset
    )
    assert without_offset >= 99.0, (
        "without the offset the model should flag essentially everything, got %.2f %%"
        % without_offset
    )


def test_a_genuinely_clear_tile_stays_clear(product):
    """Tile 11,2 is clear; a detector that flags it is broken, not conservative."""
    s2cloudless = pytest.importorskip("s2cloudless")

    from pipeline.io import read_on_10m_grid
    from pipeline.tiling import tile_window

    window = tile_window(11, 2)
    stack = np.stack(
        [read_on_10m_grid(product, band, window=window) for band in BANDS], axis=-1
    )
    reduced = stack[::6, ::6][None, ...]
    detector = s2cloudless.S2PixelCloudDetector(
        threshold=0.4, all_bands=True, average_over=1, dilation_size=1
    )
    flagged = detector.get_cloud_masks(np.clip(reduced, 0, None))[0].mean() * 100
    assert flagged < 5.0, "tile 11,2 is clear sky, got %.2f %% flagged" % flagged
