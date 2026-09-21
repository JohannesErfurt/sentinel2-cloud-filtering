"""F3 and F4 -- the grid, four-corner geocoding, and the 30 % rule.

These are the two places in the pipeline where a wrong answer looks completely
plausible: a bounding box that is 130 m short, and a tile that flips validity
because a percentage was rounded before it was compared.
"""
from __future__ import annotations

import numpy as np
import pytest
from pyproj import Transformer

from pipeline.constants import CLOUD_THRESHOLD_PERCENT, GRID, SCENE_PX, TILE_PIXELS, TILE_PX
from pipeline.tiling import (
    GridError,
    compute_tile_stats,
    is_valid,
    iter_tiles,
    scene_cloud_percent,
    tile_latlon_bbox,
    tile_utm_bounds,
    tile_window,
)

# SPEC.md 1.3: the four-corner box of tile 6,5.
TILE_6_5_BBOX = (49.29258, 10.75286, 49.34311, 10.83015)


# ------------------------------------------------------------------------- F3
def test_the_grid_has_exactly_400_tiles():
    tiles = list(iter_tiles())
    assert len(tiles) == 400
    assert tiles[0] == (0, 0) and tiles[-1] == (GRID - 1, GRID - 1)
    assert tiles[1] == (0, 1), "row-major: the column moves fastest"


def test_windows_partition_the_scene_exactly_once():
    cover = np.zeros((SCENE_PX, SCENE_PX), dtype=np.uint8)
    for row, col in iter_tiles():
        window = tile_window(row, col)
        assert (int(window.width), int(window.height)) == (TILE_PX, TILE_PX)
        cover[
            int(window.row_off) : int(window.row_off) + TILE_PX,
            int(window.col_off) : int(window.col_off) + TILE_PX,
        ] += 1
    assert cover.min() == 1 and cover.max() == 1


def test_tiles_outside_the_grid_are_rejected():
    for row, col in [(-1, 0), (0, 20), (20, 20)]:
        with pytest.raises(GridError):
            tile_window(row, col)


def test_tile_utm_bounds_tile_zero_is_the_scene_origin(meta):
    east_min, north_min, east_max, north_max = tile_utm_bounds(meta, 0, 0)
    assert (east_min, north_max) == meta.origin(10)
    assert east_max - east_min == 5490 and north_max - north_min == 5490


def test_tile_6_5_bbox_matches_the_reference(meta):
    got = tile_latlon_bbox(meta, 6, 5)
    assert max(abs(a - b) for a, b in zip(got, TILE_6_5_BBOX)) < 2e-5


def test_a_two_corner_box_is_visibly_too_small(meta):
    """Grid north is not true north, so SW+NE alone miss ~130 m of latitude.

    This is the mistake the four-corner rule exists to prevent, so it is
    asserted rather than merely described.
    """
    east_min, north_min, east_max, north_max = tile_utm_bounds(meta, 6, 5)
    transformer = Transformer.from_crs(meta.epsg, 4326, always_xy=True)
    _, lat_sw = transformer.transform(east_min, north_min)
    _, lat_ne = transformer.transform(east_max, north_max)

    four_corner = tile_latlon_bbox(meta, 6, 5)
    assert min(lat_sw, lat_ne) > four_corner[0] + 5e-4
    assert max(lat_sw, lat_ne) < four_corner[2] - 5e-4

    short_by_metres = (min(lat_sw, lat_ne) - four_corner[0]) * 111_000
    assert 100 < short_by_metres < 160, "about 130 m, per SPEC.md 1.3"


def test_the_union_of_all_boxes_is_the_scene_footprint(meta):
    boxes = [tile_latlon_bbox(meta, row, col) for row, col in iter_tiles()]
    union = (
        min(b[0] for b in boxes), min(b[1] for b in boxes),
        max(b[2] for b in boxes), max(b[3] for b in boxes),
    )
    assert max(abs(a - b) for a, b in zip(union, meta.footprint_bbox())) < 1e-4


# ------------------------------------------------------------------------- F4
def test_the_integer_rule_equals_the_float_cut_for_every_possible_count():
    """90 420.3 pixels is 30 %. No count can land on it, so <= and < agree."""
    cut = CLOUD_THRESHOLD_PERCENT * TILE_PIXELS / 100.0
    assert cut == pytest.approx(90420.3)
    counts = np.arange(0, TILE_PIXELS + 1)
    expected = counts < cut
    actual = np.array([is_valid(int(c)) for c in counts[::97]])
    assert np.array_equal(actual, expected[::97])
    # and the two operators agree at the boundary itself
    assert is_valid(90420) is True
    assert is_valid(90421) is False


def test_the_percentages_either_side_of_the_cut():
    assert 100.0 * 90420 / TILE_PIXELS == pytest.approx(29.99990, abs=1e-4)
    assert 100.0 * 90421 / TILE_PIXELS == pytest.approx(30.00023, abs=1e-4)


def test_a_tile_at_29_97_percent_is_valid_and_rounding_would_break_it(meta):
    count = int(round(0.2997 * TILE_PIXELS))
    stats = compute_tile_stats(
        meta, 6, 5, _mask_with(count), None
    )
    assert stats.valid is True
    assert stats.cloud_cover_percent == pytest.approx(29.97, abs=0.01)
    # Rounding first turns 29.97 into "30.0", where <= and < disagree.
    rounded = round(stats.cloud_cover_percent, 1)
    assert rounded == 30.0
    assert (rounded < 30.0) is not stats.valid, "rounding first flips the verdict"


def test_an_all_nodata_tile_is_invalid_despite_zero_cloud(meta):
    stats = compute_tile_stats(
        meta, 0, 0, np.zeros((TILE_PX, TILE_PX), bool), np.ones((TILE_PX, TILE_PX), bool)
    )
    assert stats.cloud_cover_percent == 0.0
    assert stats.nodata_fraction == 1.0
    assert stats.valid is False, "a black tile must not ship as valid"


def test_a_lightly_clipped_tile_is_still_valid(meta):
    nodata = np.zeros((TILE_PX, TILE_PX), bool)
    nodata[:10] = True
    stats = compute_tile_stats(meta, 0, 0, np.zeros((TILE_PX, TILE_PX), bool), nodata)
    assert stats.valid is True


def test_a_float_mask_is_rejected(meta):
    """A thresholded probability that never got thresholded is a silent error."""
    with pytest.raises(GridError, match="dtype"):
        compute_tile_stats(meta, 0, 0, np.zeros((TILE_PX, TILE_PX), np.float32))


def test_a_wrongly_shaped_mask_is_rejected(meta):
    with pytest.raises(GridError, match="shape"):
        compute_tile_stats(meta, 0, 0, np.zeros((100, 100), bool))


def test_the_mean_of_tile_percentages_is_the_scene_percentage(meta):
    rng = np.random.default_rng(5)
    counts = rng.integers(0, TILE_PIXELS, size=400)
    stats = [
        compute_tile_stats(meta, i // GRID, i % GRID, _mask_with(int(c)))
        for i, c in enumerate(counts)
    ]
    expected = 100.0 * counts.sum() / (400.0 * TILE_PIXELS)
    assert scene_cloud_percent(stats) == pytest.approx(expected, abs=1e-9)


def _mask_with(count: int) -> np.ndarray:
    mask = np.zeros(TILE_PIXELS, dtype=bool)
    mask[:count] = True
    return mask.reshape(TILE_PX, TILE_PX)
