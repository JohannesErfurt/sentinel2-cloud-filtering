"""F2 -- resampling. Down is a block mean, up is nearest for anything discrete."""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from pipeline.io import block_mean, read_on_10m_grid, resize_linear, upsample_nearest
from pipeline.tiling import tile_window


def test_upsample_nearest_equals_np_repeat():
    rng = np.random.default_rng(0)
    mask = rng.integers(0, 2, size=(37, 41), dtype=np.uint8).astype(bool)
    expected = np.repeat(np.repeat(mask, 6, axis=0), 6, axis=1)
    assert np.array_equal(upsample_nearest(mask, 6), expected)


def test_upsample_nearest_never_invents_a_value():
    """The reason a label must never be interpolated, as a property test."""
    rng = np.random.default_rng(1)
    labels = rng.integers(0, 4, size=(20, 20)).astype(np.uint8)
    enlarged = upsample_nearest(labels, 6)
    assert enlarged.dtype == labels.dtype
    assert set(np.unique(enlarged)) <= set(np.unique(labels))


def test_block_mean_equals_inter_area():
    rng = np.random.default_rng(2)
    data = rng.random((180, 240), dtype=np.float32)
    exact = block_mean(data, 6)
    area = cv2.resize(data, (240 // 6, 180 // 6), interpolation=cv2.INTER_AREA)
    assert exact.shape == (30, 40)
    assert np.allclose(exact, area, atol=1e-5)


def test_block_mean_is_exact_on_a_known_block():
    data = np.arange(36, dtype=np.float32).reshape(6, 6)
    assert block_mean(data, 6)[0, 0] == pytest.approx(17.5)


def test_block_mean_rejects_a_ragged_shape():
    with pytest.raises(ValueError, match="divisible"):
        block_mean(np.zeros((10, 10), dtype=np.float32), 6)


def test_uint8_linear_resize_does_not_produce_fractions():
    """Documented so nobody "fixes" it later (SPEC.md 1.2).

    Bilinear on a uint8 mask looks like it should give soft edges. It does not:
    it rounds straight back to 0/1 and quietly moves the boundary instead.
    """
    rng = np.random.default_rng(3)
    mask = (rng.random((100, 100)) > 0.5).astype(np.uint8)
    enlarged = cv2.resize(mask, (600, 600), interpolation=cv2.INTER_LINEAR)
    assert set(np.unique(enlarged)) <= {0, 1}, "no intermediate values appear"

    shrunk = cv2.resize(enlarged, (100, 100), interpolation=cv2.INTER_NEAREST)
    moved = int((shrunk != mask).sum())
    assert moved > 0, "edge pixels are silently relocated, which is the point"


def test_resize_linear_returns_float32():
    data = np.arange(16, dtype=np.uint8).reshape(4, 4)
    out = resize_linear(data, (8, 8))
    assert out.dtype == np.float32 and out.shape == (8, 8)


# ------------------------------------------------------------ against the real product
@pytest.mark.needs_product
def test_coarse_band_window_matches_the_scene_wide_read(product):
    """A 549 px tile is 274.5 px at 20 m, so the window straddles pixels.

    Reading one tile must still give exactly what cutting it out of a
    scene-wide read would, or streaming and scene-wide assembly disagree.
    """
    from pipeline.io import read_reflectance

    scene20 = read_reflectance(product, "B11")
    scene10 = upsample_nearest(scene20, 2)
    for row, col in [(0, 0), (6, 5), (12, 19), (19, 19)]:
        window = tile_window(row, col)
        streamed = read_on_10m_grid(product, "B11", window=window)
        top, left = row * 549, col * 549
        assert np.array_equal(streamed, scene10[top : top + 549, left : left + 549])


@pytest.mark.needs_product
def test_60m_band_window_matches_the_scene_wide_read(product):
    from pipeline.io import read_reflectance

    scene60 = read_reflectance(product, "B10")
    scene10 = upsample_nearest(scene60, 6)
    for row, col in [(0, 0), (5, 16), (19, 0)]:
        window = tile_window(row, col)
        streamed = read_on_10m_grid(product, "B10", window=window)
        top, left = row * 549, col * 549
        assert np.array_equal(streamed, scene10[top : top + 549, left : left + 549])


@pytest.mark.needs_product
def test_whole_band_downsample_is_an_exact_block_mean(product):
    """GDAL's decimated 'average' reads the JP2 overviews and is not this."""
    from pipeline.io import read_reflectance

    native = read_reflectance(product, "B04", window=tile_window(6, 5))
    reduced = read_reflectance(
        product, "B04", window=tile_window(6, 5), out_shape=(61, 61)
    )
    # 549 = 61 * 9, so the tile reduces exactly by 9.
    assert reduced.shape == (61, 61)
    assert np.allclose(reduced, block_mean(native, 9), atol=1e-6)


@pytest.mark.needs_product
def test_this_product_has_essentially_no_nodata(product):
    """SPEC.md 0.2: 4 pixels of DN == 0 in B02, out of 120.6 million."""
    from pipeline.io import read_nodata_mask

    mask = read_nodata_mask(product, "B02")
    assert mask.shape == (10980, 10980)
    assert mask.mean() < 1e-4
