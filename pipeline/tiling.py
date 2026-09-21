"""The 20x20 tile grid, four-corner geocoding, and the 30 % rule (F3, F4)."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from pyproj import Transformer
from rasterio.windows import Window

from .constants import (
    CLOUD_THRESHOLD_PERCENT,
    GRID,
    MAX_NODATA_FRACTION,
    SCENE_PX,
    TILE_METRES,
    TILE_PIXELS,
    TILE_PX,
    WGS84_EPSG,
)
from .metadata import ProductError, ProductMetadata


class GridError(RuntimeError):
    """The product does not match the 20x20 grid of 549-pixel tiles."""


@lru_cache(maxsize=8)
def _transformer(epsg: int) -> Transformer:
    # always_xy keeps the argument order (easting, northing) -> (lon, lat)
    # regardless of what the CRS definition says the axis order is.
    return Transformer.from_crs(epsg, WGS84_EPSG, always_xy=True)


def check_grid(meta: ProductMetadata) -> None:
    """Fail loudly unless the granule is exactly 20 x 20 tiles of 549 px."""
    height, width = meta.shape(10)
    if (height, width) != (SCENE_PX, SCENE_PX):
        raise GridError(
            "expected a %d x %d granule at 10 m (20 x 20 tiles of %d px), got %d x %d"
            % (SCENE_PX, SCENE_PX, TILE_PX, height, width)
        )


def iter_tiles():
    """Yield ``(row, col)`` for all 400 tiles in row-major order.

    Row-major is the order ``report.csv`` uses: line ``i + 2`` of the file is
    tile ``(i // 20, i % 20)`` (SPEC.md 1.5).
    """
    for row in range(GRID):
        for col in range(GRID):
            yield row, col


def tile_window(row: int, col: int) -> Window:
    """The tile's pixel window on the **10 m** grid."""
    if not (0 <= row < GRID and 0 <= col < GRID):
        raise GridError("tile (%d, %d) is outside the %dx%d grid" % (row, col, GRID, GRID))
    return Window(col * TILE_PX, row * TILE_PX, TILE_PX, TILE_PX)


def tile_utm_bounds(meta: ProductMetadata, row: int, col: int) -> tuple[float, float, float, float]:
    """``(east_min, north_min, east_max, north_max)`` in the granule's own CRS."""
    if not (0 <= row < GRID and 0 <= col < GRID):
        raise GridError("tile (%d, %d) is outside the %dx%d grid" % (row, col, GRID, GRID))
    ulx, uly = meta.origin(10)
    east_min = ulx + col * TILE_METRES
    north_max = uly - row * TILE_METRES
    return east_min, north_max - TILE_METRES, east_min + TILE_METRES, north_max


def tile_latlon_bbox(meta: ProductMetadata, row: int, col: int) -> tuple[float, float, float, float]:
    """``(min_latitude, min_longitude, max_latitude, max_longitude)`` in WGS84.

    The bounding box of **all four** corners, not two opposite ones. Grid north
    is not true north, so a UTM square is a slightly rotated quadrilateral in
    lat/lon: for tile 6,5 a box built from the SW and NE corners alone is
    0.00117 deg -- about 130 m -- short at each end in latitude.
    """
    east_min, north_min, east_max, north_max = tile_utm_bounds(meta, row, col)
    transformer = _transformer(meta.epsg)
    eastings = [east_min, east_max, east_max, east_min]
    northings = [north_max, north_max, north_min, north_min]
    lons, lats = transformer.transform(eastings, northings)
    return min(lats), min(lons), max(lats), max(lons)


def tile_transform(meta: ProductMetadata, row: int, col: int):
    """An affine transform mapping the tile's 10 m pixels to the granule CRS."""
    from rasterio.transform import Affine

    east_min, _, _, north_max = tile_utm_bounds(meta, row, col)
    return Affine(10.0, 0.0, east_min, 0.0, -10.0, north_max)


# ------------------------------------------------------------- tile statistics
def is_valid(cloud_pixels: int, nodata_fraction: float = 0.0) -> bool:
    """The 30 % rule, computed on integers (SPEC.md 1.4).

    30 % of a tile is 90 420.3 pixels, so no tile can sit exactly on the cut and
    ``<=`` and ``<`` agree -- but only on the unrounded count. Comparing a value
    that has already been rounded to one decimal makes a 29.97 % tile display as
    "30.0" and its verdict depends on which operator was written. Hence the
    integer form: no float ever reaches the comparison.
    """
    below_cloud_cut = 100 * int(cloud_pixels) <= CLOUD_THRESHOLD_PERCENT * TILE_PIXELS
    return bool(below_cloud_cut and nodata_fraction <= MAX_NODATA_FRACTION)


@dataclass(frozen=True)
class TileStats:
    """Per-tile numbers. ``cloud_cover_percent`` is never rounded before use."""

    row: int
    col: int
    cloud_pixels: int
    cloud_cover_percent: float
    nodata_fraction: float
    valid: bool
    min_latitude: float
    min_longitude: float
    max_latitude: float
    max_longitude: float
    extra: dict

    @property
    def name(self) -> str:
        return "tile_r%02d_c%02d" % (self.row, self.col)


def compute_tile_stats(
    meta: ProductMetadata,
    row: int,
    col: int,
    cloud_tile: np.ndarray,
    nodata_tile: np.ndarray | None = None,
    extra: dict | None = None,
) -> TileStats:
    """Cloud percentage, no-data fraction and the valid flag for one tile."""
    if cloud_tile.shape != (TILE_PX, TILE_PX):
        raise GridError(
            "tile (%d, %d) mask has shape %s, expected (%d, %d)"
            % (row, col, cloud_tile.shape, TILE_PX, TILE_PX)
        )
    if cloud_tile.dtype != np.bool_:
        raise GridError(
            "tile (%d, %d) mask has dtype %s, expected bool -- a thresholded "
            "probability or an interpolated label would be a silent error"
            % (row, col, cloud_tile.dtype)
        )

    cloud_pixels = int(np.count_nonzero(cloud_tile))
    if nodata_tile is None:
        nodata_fraction = 0.0
    else:
        if nodata_tile.shape != (TILE_PX, TILE_PX):
            raise GridError("tile (%d, %d) no-data mask has shape %s" % (row, col, nodata_tile.shape))
        nodata_fraction = float(np.count_nonzero(nodata_tile)) / TILE_PIXELS

    min_lat, min_lon, max_lat, max_lon = tile_latlon_bbox(meta, row, col)
    return TileStats(
        row=row,
        col=col,
        cloud_pixels=cloud_pixels,
        cloud_cover_percent=100.0 * cloud_pixels / TILE_PIXELS,
        nodata_fraction=nodata_fraction,
        valid=is_valid(cloud_pixels, nodata_fraction),
        min_latitude=min_lat,
        min_longitude=min_lon,
        max_latitude=max_lat,
        max_longitude=max_lon,
        extra=dict(extra or {}),
    )


def scene_cloud_percent(stats: list[TileStats]) -> float:
    """Scene cloud percentage as the mean of the per-tile percentages.

    The tiles partition the scene into 400 equal areas, so their mean is exactly
    the scene figure -- which is what check CK6 asserts.
    """
    if not stats:
        return 0.0
    return float(np.mean([s.cloud_cover_percent for s in stats]))


__all__ = [
    "GridError",
    "check_grid",
    "iter_tiles",
    "tile_window",
    "tile_utm_bounds",
    "tile_latlon_bbox",
    "tile_transform",
    "is_valid",
    "TileStats",
    "compute_tile_stats",
    "scene_cloud_percent",
    "ProductError",
]
