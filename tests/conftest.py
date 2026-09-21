"""Shared fixtures.

Command-line options and markers live in the rootdir ``conftest.py``; pytest
only honours ``pytest_addoption`` there.

Most tests run on synthetic arrays and a stand-in :class:`ProductMetadata`, so
``pytest`` passes on a machine that has never seen the 801 MB .SAFE folder
(F10). Tests that genuinely need the product are marked ``needs_product`` and
skip unless ``--safe-dir`` is given:

    pytest
    pytest --safe-dir S2C_MSIL1C_..._T32UPV_....SAFE
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.constants import BANDS, GRID, TILE_PIXELS  # noqa: E402
from pipeline.metadata import ProductMetadata  # noqa: E402

# The real geocoding of T32UPV, so geodesy can be checked without the imagery.
T32UPV_ORIGIN = (600000.0, 5500020.0)
T32UPV_EPSG = 32632
T32UPV_FOOTPRINT = (
    (49.64443670244057, 10.3851737331767),
    (49.61627372192671, 11.904572762902152),
    (48.62982157570405, 11.847478451946776),
    (48.657027178567795, 10.357910769851939),
    (49.64443670244057, 10.3851737331767),
)


@pytest.fixture(scope="session")
def safe_dir(request):
    path = request.config.getoption("--safe-dir")
    if not path:
        pytest.skip("needs --safe-dir")
    return path


@pytest.fixture(scope="session")
def product(safe_dir):
    from pipeline.metadata import read_product

    return read_product(safe_dir)


def make_metadata(**overrides) -> ProductMetadata:
    """A stand-in product with T32UPV's real geocoding and no image files."""
    defaults = dict(
        safe_dir="/nonexistent/S2C_TEST.SAFE",
        granule_dir="/nonexistent/S2C_TEST.SAFE/GRANULE/L1C_T32UPV",
        product_uri="S2C_MSIL1C_TEST_T32UPV.SAFE",
        processing_level="Level-1C",
        processing_baseline="05.11",
        spacecraft="Sentinel-2C",
        sensing_time="2025-10-02T10:27:28.089482Z",
        generation_time="2025-10-02T14:31:20.000000Z",
        epsg=T32UPV_EPSG,
        sizes={10: (10980, 10980), 20: (5490, 5490), 60: (1830, 1830)},
        geoposition={
            10: (T32UPV_ORIGIN[0], T32UPV_ORIGIN[1], 10.0, -10.0),
            20: (T32UPV_ORIGIN[0], T32UPV_ORIGIN[1], 20.0, -20.0),
            60: (T32UPV_ORIGIN[0], T32UPV_ORIGIN[1], 60.0, -60.0),
        },
        quantification_value=10000.0,
        radio_add_offset={band: -1000.0 for band in BANDS},
        has_radiometric_offset=True,
        nodata_value=0,
        saturated_value=65535,
        cloud_coverage_assessment=20.953686285049,
        snow_coverage_assessment=0.0,
        mean_sun_zenith=53.5212818451206,
        mean_sun_azimuth=168.397438235809,
        footprint=T32UPV_FOOTPRINT,
    )
    defaults.update(overrides)
    return ProductMetadata(**defaults)


@pytest.fixture
def meta() -> ProductMetadata:
    return make_metadata()


def make_stats(meta, cloud_pixels_for, extra_for=None):
    """400 :class:`TileStats` from a function of ``(row, col) -> cloud pixels``."""
    from pipeline.tiling import TileStats, iter_tiles, tile_latlon_bbox

    stats = []
    for row, col in iter_tiles():
        count = int(cloud_pixels_for(row, col))
        min_lat, min_lon, max_lat, max_lon = tile_latlon_bbox(meta, row, col)
        nodata = 0.0
        stats.append(
            TileStats(
                row=row,
                col=col,
                cloud_pixels=count,
                cloud_cover_percent=100.0 * count / TILE_PIXELS,
                nodata_fraction=nodata,
                valid=100 * count <= 30 * TILE_PIXELS,
                min_latitude=min_lat,
                min_longitude=min_lon,
                max_latitude=max_lat,
                max_longitude=max_lon,
                extra=(extra_for or (lambda r, c: {}))(row, col),
            )
        )
    return stats


@pytest.fixture
def synthetic_run(tmp_path, meta):
    """A complete, valid output folder built without any imagery.

    Most tiles are deliberately cloudy so only a handful of JPEGs get written --
    enough for CK5 and CK6 to mean something, few enough to stay fast.
    """
    import json

    import cv2

    from pipeline.report import write_report_csv, write_tile_stats_csv
    from pipeline.tiling import scene_cloud_percent, tile_utm_bounds
    from pipeline.export import write_world_file

    def cloud_for(row, col):
        # 16 clear tiles, the rest solidly over the cut.
        return 0 if (row % 5 == 0 and col % 5 == 0) else int(0.62 * TILE_PIXELS)

    stats = make_stats(meta, cloud_for)
    out_dir = tmp_path / "run"
    tiles_dir = out_dir / "tiles"
    tiles_dir.mkdir(parents=True)

    rng = np.random.default_rng(7)
    for item in stats:
        if not item.valid:
            continue
        image = rng.integers(40, 210, size=(549, 549, 3), dtype=np.uint8)
        cv2.imwrite(str(tiles_dir / (item.name + ".jpg")), image[:, :, ::-1],
                    [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        east_min, _, _, north_max = tile_utm_bounds(meta, item.row, item.col)
        write_world_file(str(tiles_dir / (item.name + ".jgw")), east_min, north_max)

    write_report_csv(str(out_dir / "report.csv"), stats)
    write_tile_stats_csv(str(out_dir / "tile_stats.csv"), stats)

    valid = sum(1 for s in stats if s.valid)
    summary = {
        "detector": "synthetic",
        "parameters": {},
        "scene": {
            "product_uri": meta.product_uri,
            "epsg": meta.epsg,
            "origin_easting": meta.origin(10)[0],
            "origin_northing": meta.origin(10)[1],
            "shape_10m": list(meta.shape(10)),
            "footprint_bbox": list(meta.footprint_bbox()),
        },
        "results": {
            "scene_cloud_percent": scene_cloud_percent(stats),
            "valid_tiles": valid,
            "invalid_tiles": GRID * GRID - valid,
            "jpegs_written": valid,
        },
    }
    (out_dir / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf8")
    return out_dir
