"""CSV output: the task's report and the sidecar (F5).

``report.csv`` carries exactly the six columns the brief names, in that order,
with one row per tile including the discarded ones. It has no identity column,
so row order, the JPEG file names and the world files are the only link from a
row back to a place on the ground -- which is why everything else, including
the tile row and column, goes to ``tile_stats.csv`` instead of being bolted on.

All formatting is fixed-width so two runs produce byte-identical files (CK8).
"""
from __future__ import annotations

import csv
import os

from .constants import REPORT_COLUMNS
from .tiling import TileStats

#: 8 decimals of latitude is about 1 mm -- far tighter than CK4's 1e-6 deg.
_LATLON = "%.8f"
#: The brief asks for a percentage; 6 decimals is more than the "at least 4"
#: SPEC.md 1.5 requires, and 100 * n / 301401 never needs more to round-trip.
_PERCENT = "%.6f"

TILE_STATS_COLUMNS = [
    "tile_row",
    "tile_col",
    "tile_name",
    "cloud_pixels",
    "cloud_cover_percent",
    "nodata_fraction",
    "valid",
    "min_latitude",
    "min_longitude",
    "max_latitude",
    "max_longitude",
]


def _open(path: str):
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    # newline="" plus an explicit terminator keeps the bytes identical on
    # Windows and POSIX, which CK8 compares.
    return open(path, "w", encoding="utf8", newline="")


def write_report_csv(path: str, stats: list[TileStats]) -> str:
    """Write the deliverable CSV: header plus one row per tile, row-major."""
    ordered = sorted(stats, key=lambda s: (s.row, s.col))
    with _open(path) as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(REPORT_COLUMNS)
        for item in ordered:
            writer.writerow(
                [
                    _LATLON % item.min_latitude,
                    _LATLON % item.min_longitude,
                    _LATLON % item.max_latitude,
                    _LATLON % item.max_longitude,
                    _PERCENT % item.cloud_cover_percent,
                    "True" if item.valid else "False",
                ]
            )
    return path


def write_tile_stats_csv(path: str, stats: list[TileStats]) -> str:
    """Write the sidecar: identity columns plus whatever the detector added."""
    ordered = sorted(stats, key=lambda s: (s.row, s.col))
    extra_keys = sorted({key for item in ordered for key in item.extra})
    with _open(path) as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(TILE_STATS_COLUMNS + extra_keys)
        for item in ordered:
            row = [
                item.row,
                item.col,
                item.name,
                item.cloud_pixels,
                _PERCENT % item.cloud_cover_percent,
                "%.8f" % item.nodata_fraction,
                "True" if item.valid else "False",
                _LATLON % item.min_latitude,
                _LATLON % item.min_longitude,
                _LATLON % item.max_latitude,
                _LATLON % item.max_longitude,
            ]
            for key in extra_keys:
                value = item.extra.get(key, "")
                row.append(_PERCENT % value if isinstance(value, float) else value)
            writer.writerow(row)
    return path


def read_report_csv(path: str) -> tuple[list[str], list[dict]]:
    """Read back a report as ``(header, rows)``; used by the validator (F9)."""
    with open(path, encoding="utf8", newline="") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration:
            raise ValueError("%s is empty" % path) from None
        rows = [dict(zip(header, values)) for values in reader if values]
    return header, rows


__all__ = [
    "TILE_STATS_COLUMNS",
    "write_report_csv",
    "write_tile_stats_csv",
    "read_report_csv",
]
