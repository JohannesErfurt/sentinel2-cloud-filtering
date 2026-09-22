"""Deliverable checks CK1-CK8 (F9).

CK1-CK6 run as assertions at the end of every pipeline run, so a bad run fails
loudly instead of producing quiet nonsense. CK7 and CK8 run from
``scripts/check_deliverables.py``.

Every check that can work without the 801 MB .SAFE folder does: the geocoding
and area references it needs are written into ``run_summary.json``, so an
unzipped deliverable can be validated on its own.
"""
from __future__ import annotations

import glob
import json
import os
import re
from dataclasses import dataclass

import numpy as np

from .constants import (
    CLOUD_THRESHOLD_PERCENT,
    GRID,
    MAX_NODATA_FRACTION,
    REPORT_COLUMNS,
    SCENE_PX,
    TILE_PIXELS,
    TILE_PX,
)
from .report import read_report_csv

ALL_CHECKS = ["CK1", "CK2", "CK3", "CK4", "CK5", "CK6", "CK7", "CK8"]
IN_PIPELINE_CHECKS = ["CK1", "CK2", "CK3", "CK4", "CK5", "CK6"]

_DECIMALS = re.compile(r"\.(\d+)$")


class CheckFailure(AssertionError):
    """One or more deliverable checks failed."""


@dataclass
class CheckResult:
    id: str
    ok: bool
    detail: str
    skipped: bool = False

    def __str__(self) -> str:
        status = "SKIP" if self.skipped else ("PASS" if self.ok else "FAIL")
        return "%-4s %-5s %s" % (self.id, status, self.detail)


def _summary(out_dir: str) -> dict:
    path = os.path.join(out_dir, "run_summary.json")
    if not os.path.isfile(path):
        raise CheckFailure("run_summary.json missing from %s" % out_dir)
    with open(path, encoding="utf8") as handle:
        return json.load(handle)


def _rows(out_dir: str) -> tuple[list[str], list[dict]]:
    path = os.path.join(out_dir, "report.csv")
    if not os.path.isfile(path):
        raise CheckFailure("report.csv missing from %s" % out_dir)
    return read_report_csv(path)


def _tile_stats(out_dir: str) -> dict[tuple[int, int], dict]:
    path = os.path.join(out_dir, "tile_stats.csv")
    if not os.path.isfile(path):
        return {}
    _, rows = read_report_csv(path)
    return {(int(r["tile_row"]), int(r["tile_col"])): r for r in rows}


# --------------------------------------------------------------------- CK1-CK6
def check_ck1(out_dir: str, meta=None) -> CheckResult:
    """Grid: 400 tiles, each a 549 x 549 window, covering 10980^2 exactly once."""
    from .tiling import iter_tiles, tile_window

    cover = np.zeros((SCENE_PX, SCENE_PX), dtype=np.uint8)
    tiles = list(iter_tiles())
    if len(tiles) != GRID * GRID:
        return CheckResult("CK1", False, "grid has %d tiles, expected %d" % (len(tiles), GRID * GRID))
    for row, col in tiles:
        window = tile_window(row, col)
        if (int(window.width), int(window.height)) != (TILE_PX, TILE_PX):
            return CheckResult("CK1", False, "tile (%d,%d) window is not %d px" % (row, col, TILE_PX))
        cover[
            int(window.row_off) : int(window.row_off) + TILE_PX,
            int(window.col_off) : int(window.col_off) + TILE_PX,
        ] += 1
    if not np.array_equal(cover, np.ones_like(cover)):
        return CheckResult(
            "CK1",
            False,
            "windows do not partition the scene: %d pixels uncovered, %d covered twice"
            % (int((cover == 0).sum()), int((cover > 1).sum())),
        )
    return CheckResult("CK1", True, "400 tiles of %d px cover %d^2 exactly once" % (TILE_PX, SCENE_PX))


def check_ck2(out_dir: str, meta=None) -> CheckResult:
    """CSV schema: exactly the six named columns and exactly 400 data rows."""
    header, rows = _rows(out_dir)
    if header != REPORT_COLUMNS:
        return CheckResult("CK2", False, "header is %s, expected %s" % (header, REPORT_COLUMNS))
    if len(rows) != GRID * GRID:
        return CheckResult("CK2", False, "%d data rows, expected %d" % (len(rows), GRID * GRID))
    return CheckResult("CK2", True, "header exact, %d data rows" % len(rows))


def check_ck3(out_dir: str, meta=None) -> CheckResult:
    """CSV values: range, precision, and the valid flag matching the rules."""
    _, rows = _rows(out_dir)
    stats = _tile_stats(out_dir)
    for index, row in enumerate(rows):
        raw = row["cloud_cover_percent"]
        try:
            percent = float(raw)
        except ValueError:
            return CheckResult("CK3", False, "row %d: cloud_cover_percent %r is not a number" % (index, raw))
        if not 0.0 <= percent <= 100.0:
            return CheckResult("CK3", False, "row %d: cloud_cover_percent %s out of [0, 100]" % (index, raw))
        match = _DECIMALS.search(raw)
        if not match or len(match.group(1)) < 4:
            return CheckResult("CK3", False, "row %d: %r has fewer than 4 decimals" % (index, raw))
        if row["valid"] not in ("True", "False"):
            return CheckResult("CK3", False, "row %d: valid is %r, expected True or False" % (index, row["valid"]))

        key = (index // GRID, index % GRID)
        if key in stats:
            cloud_pixels = int(stats[key]["cloud_pixels"])
            nodata_fraction = float(stats[key]["nodata_fraction"])
        else:
            cloud_pixels = int(round(percent / 100.0 * TILE_PIXELS))
            nodata_fraction = 0.0
        expected = (
            100 * cloud_pixels <= CLOUD_THRESHOLD_PERCENT * TILE_PIXELS
            and nodata_fraction <= MAX_NODATA_FRACTION
        )
        if (row["valid"] == "True") != expected:
            return CheckResult(
                "CK3",
                False,
                "row %d (tile %d,%d): valid=%s but the rule gives %s at %s %% / %.4f no-data"
                % (index, key[0], key[1], row["valid"], expected, raw, nodata_fraction),
            )
    return CheckResult("CK3", True, "400 rows in range, >=4 decimals, valid flag matches the rules")


def check_ck4(out_dir: str, meta=None) -> CheckResult:
    """Geography: four-corner boxes, and their union equal to the footprint."""
    from .tiling import tile_latlon_bbox

    _, rows = _rows(out_dir)
    summary = _summary(out_dir)
    scene = summary.get("scene", {})
    reference = _GridReference.from_summary(scene, meta)
    if reference is None:
        return CheckResult("CK4", True, "no scene geocoding available; range checks only", skipped=True)

    worst = 0.0
    for index, row in enumerate(rows):
        values = [float(row[name]) for name in REPORT_COLUMNS[:4]]
        min_lat, min_lon, max_lat, max_lon = values
        if not (min_lat < max_lat and min_lon < max_lon):
            return CheckResult("CK4", False, "row %d: degenerate bounding box %s" % (index, values))
        expected = tile_latlon_bbox(reference, index // GRID, index % GRID)
        worst = max(worst, max(abs(a - b) for a, b in zip(values, expected)))
        if worst > 1e-6:
            return CheckResult(
                "CK4", False, "row %d: box %s differs from the grid by %.3g deg" % (index, values, worst)
            )

    union = (
        min(float(r["min_latitude"]) for r in rows),
        min(float(r["min_longitude"]) for r in rows),
        max(float(r["max_latitude"]) for r in rows),
        max(float(r["max_longitude"]) for r in rows),
    )
    footprint = scene.get("footprint_bbox")
    if footprint:
        gap = max(abs(a - b) for a, b in zip(union, footprint))
        if gap > 1e-4:
            return CheckResult(
                "CK4", False, "union %s differs from footprint %s by %.3g deg" % (union, footprint, gap)
            )
        return CheckResult(
            "CK4", True, "boxes match the grid to %.1e deg; union matches the footprint to %.1e deg" % (worst, gap)
        )
    return CheckResult("CK4", True, "boxes match the grid to %.1e deg" % worst)


def check_ck5(out_dir: str, meta=None) -> CheckResult:
    """JPEGs: exactly the valid rows, correct size, world files, TCI fidelity."""
    import cv2

    from .export import read_prj_epsg, read_world_file
    from .tiling import tile_utm_bounds

    _, rows = _rows(out_dir)
    tiles_dir = os.path.join(out_dir, "tiles")
    found = sorted(os.path.basename(p) for p in glob.glob(os.path.join(tiles_dir, "*.jpg")))
    expected = sorted(
        "tile_r%02d_c%02d.jpg" % (index // GRID, index % GRID)
        for index, row in enumerate(rows)
        if row["valid"] == "True"
    )
    if found != expected:
        missing = sorted(set(expected) - set(found))
        extra = sorted(set(found) - set(expected))
        return CheckResult(
            "CK5",
            False,
            "tiles/ does not match the valid rows: %d missing (%s), %d unexpected (%s)"
            % (len(missing), ", ".join(missing[:3]) or "-", len(extra), ", ".join(extra[:3]) or "-"),
        )

    summary = _summary(out_dir)
    reference = _GridReference.from_summary(summary.get("scene", {}), meta)
    worst_jgw = 0.0
    for name in found:
        image = cv2.imread(os.path.join(tiles_dir, name), cv2.IMREAD_UNCHANGED)
        if image is None or image.shape != (TILE_PX, TILE_PX, 3):
            shape = None if image is None else image.shape
            return CheckResult("CK5", False, "%s decodes to %s, expected (%d, %d, 3)" % (name, shape, TILE_PX, TILE_PX))
        world = os.path.join(tiles_dir, name[:-4] + ".jgw")
        if not os.path.isfile(world):
            return CheckResult("CK5", False, "%s has no world file" % name)
        prj = os.path.join(tiles_dir, name[:-4] + ".prj")
        if not os.path.isfile(prj):
            return CheckResult(
                "CK5", False,
                "%s has no .prj -- a .jgw alone has no unit or CRS attached, so a GIS "
                "tool guesses one (usually its own project CRS) and reads UTM metres "
                "as degrees; this is exactly what silently misplaces a tile" % name,
            )
        if reference is not None:
            declared_epsg = read_prj_epsg(prj)
            if declared_epsg != reference.epsg:
                return CheckResult(
                    "CK5", False,
                    "%s's .prj names EPSG:%s, expected EPSG:%d" % (name, declared_epsg, reference.epsg),
                )
            row, col = int(name[6:8]), int(name[10:12])
            x_size, y_size, east, north = read_world_file(world)
            east_min, _, _, north_max = tile_utm_bounds(reference, row, col)
            worst_jgw = max(worst_jgw, abs(east - (east_min + 5.0)), abs(north - (north_max - 5.0)))
            if (x_size, y_size) != (10.0, -10.0) or worst_jgw > 1e-6:
                return CheckResult("CK5", False, "%s world file does not match the grid" % name)

    if meta is None:
        return CheckResult(
            "CK5", True, "%d JPEGs and world files match the valid rows (no TCI comparison)" % len(found), skipped=True
        )

    from .io import read_tci
    from .tiling import tile_window

    rng = np.random.default_rng(0)
    sample = [found[i] for i in rng.choice(len(found), size=min(12, len(found)), replace=False)] if found else []
    worst = 0.0
    for name in sample:
        row, col = int(name[6:8]), int(name[10:12])
        source = read_tci(meta, tile_window(row, col)).astype(np.int16)
        decoded = cv2.imread(os.path.join(tiles_dir, name), cv2.IMREAD_UNCHANGED)[:, :, ::-1].astype(np.int16)
        worst = max(worst, float(np.abs(source - decoded).mean()))
        channel_gap = float(np.abs(source.mean(axis=(0, 1)) - decoded.mean(axis=(0, 1))).max())
        if channel_gap > 2.0:
            return CheckResult(
                "CK5", False, "%s per-channel mean differs by %.2f grey levels (channel swap?)" % (name, channel_gap)
            )
    if worst > 3.0:
        return CheckResult("CK5", False, "worst mean absolute difference to TCI is %.2f grey levels" % worst)
    return CheckResult(
        "CK5", True, "%d JPEGs match the valid rows; worst mean |diff| to TCI %.2f grey levels over %d sampled"
        % (len(found), worst, len(sample))
    )


def check_ck6(out_dir: str, meta=None) -> CheckResult:
    """Consistency between the CSV, the summary and the folder of JPEGs."""
    _, rows = _rows(out_dir)
    summary = _summary(out_dir)
    mean_percent = float(np.mean([float(r["cloud_cover_percent"]) for r in rows]))
    reported = summary.get("results", {}).get("scene_cloud_percent")
    if reported is None:
        return CheckResult("CK6", False, "run_summary.json has no results.scene_cloud_percent")
    if abs(mean_percent - float(reported)) > 1e-3:
        return CheckResult(
            "CK6", False, "mean of tile percentages %.6f != scene %.6f" % (mean_percent, float(reported))
        )
    valid_rows = sum(1 for r in rows if r["valid"] == "True")
    jpegs = len(glob.glob(os.path.join(out_dir, "tiles", "*.jpg")))
    if valid_rows != jpegs:
        return CheckResult("CK6", False, "%d valid rows but %d JPEGs" % (valid_rows, jpegs))
    return CheckResult(
        "CK6", True, "mean of tiles %.4f %% == scene %.4f %%; %d valid rows == %d JPEGs"
        % (mean_percent, float(reported), valid_rows, jpegs)
    )


# ------------------------------------------------------------------- CK7, CK8
def check_ck7(out_dir: str, meta=None) -> CheckResult:
    """GeoJSON: structure, coordinate order, extent, validity and area."""
    from shapely.geometry import shape

    from .export import geojson_utm_area

    path = os.path.join(out_dir, "cloud_mask.geojson")
    if not os.path.isfile(path):
        return CheckResult("CK7", True, "no cloud_mask.geojson (optional bonus)", skipped=True)
    with open(path, encoding="utf8") as handle:
        collection = json.load(handle)
    if collection.get("type") != "FeatureCollection":
        return CheckResult("CK7", False, "not a FeatureCollection")

    summary = _summary(out_dir)
    scene = summary.get("scene", {})
    footprint = scene.get("footprint_bbox")
    invalid = 0
    for feature in collection.get("features", []):
        geometry = shape(feature["geometry"])
        if not geometry.is_valid:
            invalid += 1
        if footprint:
            min_lat, min_lon, max_lat, max_lon = footprint
            lon0, lat0, lon1, lat1 = geometry.bounds
            margin = 1e-3
            if not (
                min_lon - margin <= lon0 and lon1 <= max_lon + margin
                and min_lat - margin <= lat0 and lat1 <= max_lat + margin
            ):
                return CheckResult(
                    "CK7",
                    False,
                    "a geometry with bounds %s falls outside the footprint %s -- "
                    "coordinates may be (lat, lon) instead of (lon, lat)"
                    % ((lon0, lat0, lon1, lat1), footprint),
                )
    if invalid:
        return CheckResult("CK7", False, "%d invalid geometries" % invalid)

    geojson_info = summary.get("results", {}).get("geojson", {})
    expected_area = geojson_info.get("mask_area_m2")
    tolerance = float(geojson_info.get("area_tolerance", 1e-6))
    epsg = scene.get("epsg")
    if expected_area and epsg:
        actual = geojson_utm_area(collection, int(epsg))
        relative = abs(actual - float(expected_area)) / float(expected_area)
        if relative > tolerance:
            return CheckResult(
                "CK7",
                False,
                "polygon area %.1f m2 differs from the mask's %.1f m2 by %.3g (tolerance %.3g)"
                % (actual, float(expected_area), relative, tolerance),
            )
        return CheckResult(
            "CK7", True, "%d features, valid, in (lon, lat); area within %.1e of the mask"
            % (len(collection.get("features", [])), relative)
        )
    return CheckResult(
        "CK7", True, "%d features, valid, in (lon, lat); no area reference recorded"
        % len(collection.get("features", []))
    )


def check_ck8(out_dir: str, other_dir: str) -> CheckResult:
    """Determinism: two runs give a byte-identical report and the same tile list."""
    first = os.path.join(out_dir, "report.csv")
    second = os.path.join(other_dir, "report.csv")
    for path in (first, second):
        if not os.path.isfile(path):
            return CheckResult("CK8", False, "report.csv missing from %s" % os.path.dirname(path))
    with open(first, "rb") as a, open(second, "rb") as b:
        if a.read() != b.read():
            return CheckResult("CK8", False, "report.csv differs between the two runs")
    names_a = sorted(os.path.basename(p) for p in glob.glob(os.path.join(out_dir, "tiles", "*.jpg")))
    names_b = sorted(os.path.basename(p) for p in glob.glob(os.path.join(other_dir, "tiles", "*.jpg")))
    if names_a != names_b:
        return CheckResult("CK8", False, "tile lists differ (%d vs %d files)" % (len(names_a), len(names_b)))
    return CheckResult("CK8", True, "report.csv byte-identical and %d tiles identical across two runs" % len(names_a))


# ------------------------------------------------------------------- plumbing
class _GridReference:
    """The minimum geocoding CK4 and CK5 need: origin, CRS and raster size.

    Either the real :class:`ProductMetadata`, or a stand-in rebuilt from
    ``run_summary.json`` so an unzipped deliverable validates without the .SAFE
    folder.
    """

    def __init__(self, epsg: int, ulx: float, uly: float, shape10: tuple[int, int]):
        self.epsg = epsg
        self._ulx, self._uly = ulx, uly
        self._shape = shape10

    def origin(self, resolution: int = 10):
        return self._ulx, self._uly

    def shape(self, resolution: int = 10):
        return self._shape

    @classmethod
    def from_summary(cls, scene: dict, meta=None):
        if meta is not None:
            return meta
        try:
            return cls(
                int(scene["epsg"]),
                float(scene["origin_easting"]),
                float(scene["origin_northing"]),
                tuple(scene["shape_10m"]),
            )
        except (KeyError, TypeError, ValueError):
            return None


_CHECKS = {
    "CK1": check_ck1,
    "CK2": check_ck2,
    "CK3": check_ck3,
    "CK4": check_ck4,
    "CK5": check_ck5,
    "CK6": check_ck6,
    "CK7": check_ck7,
}


def run_checks(out_dir: str, meta=None, ids: list[str] | None = None) -> list[CheckResult]:
    """Run the named checks (default: everything except CK8) over an output folder."""
    results = []
    for check_id in ids or [c for c in ALL_CHECKS if c != "CK8"]:
        function = _CHECKS.get(check_id)
        if function is None:
            continue
        try:
            results.append(function(out_dir, meta))
        except CheckFailure as error:
            results.append(CheckResult(check_id, False, str(error)))
        except Exception as error:  # a check must never mask a real failure
            results.append(CheckResult(check_id, False, "%s: %s" % (type(error).__name__, error)))
    return results


def assert_checks(out_dir: str, meta=None, ids: list[str] | None = None) -> list[CheckResult]:
    """Run checks and raise :class:`CheckFailure` if any failed."""
    results = run_checks(out_dir, meta, ids)
    failures = [r for r in results if not r.ok]
    if failures:
        raise CheckFailure(
            "deliverable checks failed:\n  " + "\n  ".join(str(r) for r in failures)
        )
    return results


__all__ = [
    "ALL_CHECKS",
    "IN_PIPELINE_CHECKS",
    "CheckFailure",
    "CheckResult",
    "run_checks",
    "assert_checks",
    "check_ck8",
]
