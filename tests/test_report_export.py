"""F5, F6, F7 -- the CSV schema, JPEG output, and GeoJSON."""
from __future__ import annotations

import json

import cv2
import numpy as np
import pytest
from rasterio.transform import Affine

from pipeline.constants import GRID, REPORT_COLUMNS, TILE_PIXELS, TILE_PX
from pipeline.export import (
    build_geojson,
    geojson_utm_area,
    write_geojson,
    write_tile_jpeg,
)
from pipeline.report import read_report_csv, write_report_csv
from tests.conftest import make_stats


# ------------------------------------------------------------------------- F5
def test_report_has_exactly_the_six_columns_and_400_rows(tmp_path, meta):
    stats = make_stats(meta, lambda r, c: (r * GRID + c) * 700)
    path = write_report_csv(str(tmp_path / "report.csv"), stats)
    header, rows = read_report_csv(path)
    assert header == REPORT_COLUMNS
    assert len(rows) == 400


def test_report_rows_are_row_major(tmp_path, meta):
    stats = make_stats(meta, lambda r, c: 0)
    path = write_report_csv(str(tmp_path / "report.csv"), stats)
    _, rows = read_report_csv(path)
    for index, row in enumerate(rows):
        expected = stats[index]
        assert (expected.row, expected.col) == (index // GRID, index % GRID)
        assert float(row["min_latitude"]) == pytest.approx(expected.min_latitude, abs=1e-8)


def test_percentages_keep_at_least_four_decimals(tmp_path, meta):
    stats = make_stats(meta, lambda r, c: 1)
    _, rows = read_report_csv(write_report_csv(str(tmp_path / "r.csv"), stats))
    decimals = rows[0]["cloud_cover_percent"].split(".")[1]
    assert len(decimals) >= 4
    assert float(rows[0]["cloud_cover_percent"]) == pytest.approx(100.0 / TILE_PIXELS, abs=5e-7)


def test_the_written_percentage_recovers_the_exact_pixel_count(tmp_path, meta):
    """CK3 re-derives the count from the percentage, so the format must not lose it.

    One pixel is 100 / 301401 = 0.000332 %, three times coarser than the sixth
    decimal place, so every count from 0 to 301401 round-trips exactly.
    """
    counts = [0, 1, 2, 90419, 90420, 90421, TILE_PIXELS - 1, TILE_PIXELS]
    stats = make_stats(meta, lambda r, c: counts[(r * GRID + c) % len(counts)])
    _, rows = read_report_csv(write_report_csv(str(tmp_path / "r.csv"), stats))
    for index, row in enumerate(rows):
        recovered = round(float(row["cloud_cover_percent"]) / 100.0 * TILE_PIXELS)
        assert recovered == stats[index].cloud_pixels


def test_valid_is_written_as_true_or_false(tmp_path, meta):
    stats = make_stats(meta, lambda r, c: 0 if r else TILE_PIXELS)
    _, rows = read_report_csv(write_report_csv(str(tmp_path / "r.csv"), stats))
    assert {row["valid"] for row in rows} == {"True", "False"}


def test_writing_twice_gives_identical_bytes(tmp_path, meta):
    """CK8 compares bytes, so formatting must not drift between runs."""
    stats = make_stats(meta, lambda r, c: r * c * 13)
    first = (tmp_path / "a.csv")
    second = (tmp_path / "b.csv")
    write_report_csv(str(first), stats)
    write_report_csv(str(second), stats)
    assert first.read_bytes() == second.read_bytes()
    assert b"\r\n" not in first.read_bytes(), "line endings must not depend on the platform"


# ------------------------------------------------------------------------- F6
def test_jpeg_round_trips_without_a_channel_swap(tmp_path):
    """A red/blue swap survives every shape check, so compare channel means."""
    rng = np.random.default_rng(11)
    rgb = np.stack(
        [
            np.full((TILE_PX, TILE_PX), 200, np.uint8),
            rng.integers(90, 110, (TILE_PX, TILE_PX), dtype=np.uint8),
            np.full((TILE_PX, TILE_PX), 40, np.uint8),
        ],
        axis=-1,
    )
    path = write_tile_jpeg(str(tmp_path / "tile.jpg"), rgb)
    decoded = cv2.imread(path, cv2.IMREAD_UNCHANGED)[:, :, ::-1]
    assert decoded.shape == (TILE_PX, TILE_PX, 3)
    gaps = np.abs(rgb.mean(axis=(0, 1)) - decoded.mean(axis=(0, 1)))
    assert gaps.max() < 2.0
    assert decoded[..., 0].mean() > decoded[..., 2].mean(), "red stays red"


def test_jpeg_rejects_the_wrong_dtype_or_shape(tmp_path):
    with pytest.raises(ValueError):
        write_tile_jpeg(str(tmp_path / "a.jpg"), np.zeros((10, 10), np.uint8))
    with pytest.raises(ValueError):
        write_tile_jpeg(str(tmp_path / "b.jpg"), np.zeros((10, 10, 3), np.float32))


# ------------------------------------------------------------------------- F7
def _square_mask(size=20):
    mask = np.zeros((size, size), dtype=bool)
    mask[4:12, 4:12] = True
    return mask


def test_geojson_is_longitude_latitude(tmp_path, meta):
    transform = Affine(60.0, 0.0, 600000.0, 0.0, -60.0, 5500020.0)
    collection, areas = build_geojson({"cloud": _square_mask()}, transform, meta.epsg)
    coordinates = collection["features"][0]["geometry"]["coordinates"][0]
    lons = [point[0] for point in coordinates]
    lats = [point[1] for point in coordinates]
    assert 10 < min(lons) < 12, "first value must be longitude"
    assert 49 < min(lats) < 50, "second value must be latitude"
    assert areas["cloud"] == pytest.approx(8 * 8 * 60 * 60)


def test_geojson_coordinates_are_rounded_to_six_decimals(tmp_path, meta):
    transform = Affine(60.0, 0.0, 600000.0, 0.0, -60.0, 5500020.0)
    collection, _ = build_geojson({"cloud": _square_mask()}, transform, meta.epsg)
    for point in collection["features"][0]["geometry"]["coordinates"][0]:
        for value in point:
            assert round(value, 6) == value


def test_geojson_area_survives_the_round_trip(tmp_path, meta):
    """Rounding to 6 decimals must not move the area by more than CK7 allows."""
    transform = Affine(60.0, 0.0, 600000.0, 0.0, -60.0, 5500020.0)
    mask = _square_mask()
    collection, _ = build_geojson({"cloud": mask}, transform, meta.epsg)
    pixel_area = int(mask.sum()) * 60 * 60
    actual = geojson_utm_area(collection, meta.epsg)
    assert abs(actual - pixel_area) / pixel_area < 1e-3


def test_geojson_separates_named_classes(tmp_path, meta):
    transform = Affine(60.0, 0.0, 600000.0, 0.0, -60.0, 5500020.0)
    opaque = _square_mask()
    cirrus = np.zeros_like(opaque)
    cirrus[15:18, 15:18] = True
    collection, areas = build_geojson({"opaque": opaque, "cirrus": cirrus}, transform, meta.epsg)
    classes = sorted(f["properties"]["class"] for f in collection["features"])
    assert classes == ["cirrus", "opaque"]
    assert set(areas) == {"opaque", "cirrus"}


def test_geojson_file_is_written_as_a_feature_collection(tmp_path, meta):
    transform = Affine(60.0, 0.0, 600000.0, 0.0, -60.0, 5500020.0)
    collection, _ = build_geojson({"cloud": _square_mask()}, transform, meta.epsg)
    path = write_geojson(str(tmp_path / "cloud_mask.geojson"), collection)
    reloaded = json.loads(open(path, encoding="utf8").read())
    assert reloaded["type"] == "FeatureCollection"
    assert reloaded["features"][0]["properties"]["class"] == "cloud"


def test_geojson_rejects_a_non_boolean_mask(meta):
    from pipeline.export import mask_to_features

    transform = Affine(60.0, 0.0, 600000.0, 0.0, -60.0, 5500020.0)
    with pytest.raises(ValueError, match="boolean"):
        mask_to_features(np.zeros((8, 8), np.uint8), transform, "cloud")
