"""The pipeline command (F8).

    python -m pipeline.run --safe-dir <SAFE> --detector {esa,threshold,s2cloudless} --out output/<name>

One pipeline, three interchangeable detector backends. Everything except the
cloud mask itself is shared, so adding a backend never touches this file.

Tiles are streamed: each of the 400 windows is read, masked, measured and
written before the next one starts. A whole 10 m band is only 1.9 s to read, so
there is no speed argument for holding the scene in memory -- but a single
10980^2 float32 array is 482 MB, and the threshold detector would need five of
them. Memory is the constraint, and streaming removes it.
"""
from __future__ import annotations

import argparse
import importlib.metadata as importlib_metadata
import json
import os
import sys
import time

import numpy as np

from . import detectors
from .checks import IN_PIPELINE_CHECKS, CheckFailure, assert_checks
from .constants import TILE_PIXELS
from .export import (
    DEFAULT_JPEG_QUALITY,
    build_geojson,
    write_geojson,
    write_prj_file,
    write_tile_jpeg,
    write_world_file,
)
from .io import read_nodata_mask, read_tci
from .metadata import ProductError, read_product
from .report import write_report_csv, write_tile_stats_csv
from .tiling import (
    check_grid,
    compute_tile_stats,
    iter_tiles,
    scene_cloud_percent,
    tile_utm_bounds,
    tile_window,
)
from .util import peak_memory_mb

_VERSION_PACKAGES = [
    "numpy", "scipy", "opencv-python-headless", "rasterio", "pyproj",
    "shapely", "s2cloudless", "lightgbm",
]

#: Relative tolerance CK7 applies to the GeoJSON area. Coordinates are rounded
#: to 6 decimals (about 0.1 m) before writing, which perturbs polygon areas
#: slightly; simplification perturbs them a great deal more.
_AREA_TOLERANCE_ROUNDED = 1e-3
_AREA_TOLERANCE_SIMPLIFIED = 5e-3


def _versions() -> dict:
    found = {"python": sys.version.split()[0]}
    for name in _VERSION_PACKAGES:
        try:
            found[name] = importlib_metadata.version(name)
        except importlib_metadata.PackageNotFoundError:
            continue
    return found


def build_parser() -> argparse.ArgumentParser:
    known = detectors.available()
    parser = argparse.ArgumentParser(
        prog="python -m pipeline.run",
        description=(
            "Split a Sentinel-2 L1C scene into 549x549 tiles, estimate cloud "
            "coverage per tile, discard tiles above 30 %% cloud, and write the "
            "valid tiles as JPEGs plus a CSV report."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "registered detectors: %s\n" % (", ".join(known) or "none yet -- see tasks B1-B3 in SPEC.md")
        ),
    )
    parser.add_argument("--safe-dir", required=True, help="path to the .SAFE product folder")
    parser.add_argument(
        "--detector",
        required=True,
        help="cloud-mask backend: %s" % (", ".join(known) or "none registered yet"),
    )
    parser.add_argument("--out", required=True, help="output folder; created if absent")

    parser.add_argument(
        "--detection-resolution", type=int, choices=(10, 60), default=None,
        help="grid the detector's band tests run on (decision D2); backend default if omitted",
    )
    parser.add_argument("--jpeg-quality", type=int, default=DEFAULT_JPEG_QUALITY, help="default 90")
    parser.add_argument(
        "--save-tile-masks", action="store_true",
        help="also write tile_masks/tile_rRR_cCC.png for all 400 tiles",
    )
    parser.add_argument("--no-geojson", action="store_true", help="skip the GeoJSON bonus output")
    parser.add_argument(
        "--geojson-simplify", type=float, default=None, metavar="METRES",
        help="simplify polygons before writing; prefer coordinate rounding (see SPEC.md 1.7)",
    )
    parser.add_argument("--skip-checks", action="store_true", help="do not run CK1-CK6 after the run")
    parser.add_argument("--quiet", action="store_true", help="only print the final summary line")

    group = parser.add_argument_group("threshold detector (B2)")
    group.add_argument("--t-bright", type=float, default=None, help="brightness threshold")
    group.add_argument("--t-ndsi", type=float, default=None, help="NDSI veto threshold")
    group.add_argument("--t-cirrus", type=float, default=None, help="B10 cirrus threshold")

    group = parser.add_argument_group("s2cloudless detector (B3)")
    group.add_argument("--prob-threshold", type=float, default=None, help="cloud probability threshold")
    group.add_argument("--average-over", type=int, default=None, help="averaging disk radius; 0 for none")
    group.add_argument("--dilation-size", type=int, default=None, help="dilation disk radius; 0 for none")

    return parser


def _detector_params(args: argparse.Namespace) -> dict:
    names = [
        "detection_resolution", "t_bright", "t_ndsi", "t_cirrus",
        "prob_threshold", "average_over", "dilation_size",
    ]
    return {name: getattr(args, name) for name in names if getattr(args, name) is not None}


def run(args: argparse.Namespace) -> dict:
    started = time.time()
    log = (lambda *a: None) if args.quiet else (lambda *a: print(*a, flush=True))

    meta = read_product(args.safe_dir)
    check_grid(meta)
    log("product  %s" % meta.product_uri)
    log("baseline %s  |  offset %s  |  quantification %g"
        % (meta.processing_baseline,
           "%+g" % meta.offset("B02") if meta.has_radiometric_offset else "absent (pre-04.00)",
           meta.quantification_value))

    params = _detector_params(args)
    detector = detectors.build(args.detector, **params)
    log("detector %s%s" % (detector.name, "  " + detector.description if detector.description else ""))

    out_dir = os.path.abspath(args.out)
    tiles_dir = os.path.join(out_dir, "tiles")
    os.makedirs(tiles_dir, exist_ok=True)
    masks_dir = os.path.join(out_dir, "tile_masks")
    if args.save_tile_masks:
        os.makedirs(masks_dir, exist_ok=True)

    with open(os.path.join(out_dir, "scene_metadata.json"), "w", encoding="utf8") as handle:
        json.dump(meta.to_dict(), handle, indent=2, sort_keys=True)
        handle.write("\n")

    stats = []
    written = 0
    try:
        for index, (row, col) in enumerate(iter_tiles()):
            window = tile_window(row, col)
            cloud = detector.tile_mask(meta, row, col)
            nodata = read_nodata_mask(meta, window=window)
            item = compute_tile_stats(meta, row, col, cloud, nodata, detector.tile_extra(row, col))
            stats.append(item)

            if item.valid:
                rgb = read_tci(meta, window)
                write_tile_jpeg(os.path.join(tiles_dir, item.name + ".jpg"), rgb, args.jpeg_quality)
                east_min, _, _, north_max = tile_utm_bounds(meta, row, col)
                write_world_file(os.path.join(tiles_dir, item.name + ".jgw"), east_min, north_max)
                # A .jgw alone is numbers with no unit or CRS attached, so a GIS
                # tool has to guess one -- usually its own project CRS, which
                # turns UTM metres into nonsense degrees. The .prj sidecar is
                # what lets a plain drag-and-drop land in the right place.
                write_prj_file(os.path.join(tiles_dir, item.name + ".prj"), meta.epsg)
                written += 1

            if args.save_tile_masks:
                import cv2

                cv2.imwrite(
                    os.path.join(masks_dir, item.name + ".png"),
                    (cloud.astype(np.uint8) * 255),
                )
            if not args.quiet and (index + 1) % 50 == 0:
                log("  %3d/400 tiles  |  %d valid so far" % (index + 1, written))

        valid = sum(1 for s in stats if s.valid)
        scene_percent = scene_cloud_percent(stats)

        write_report_csv(os.path.join(out_dir, "report.csv"), stats)
        write_tile_stats_csv(os.path.join(out_dir, "tile_stats.csv"), stats)

        geojson_info: dict = {}
        if not args.no_geojson:
            geojson_info = _write_geojson(meta, detector, out_dir, args, log)

        # After the tile loop and the GeoJSON, not before: a detector that
        # caches a scene-wide array (esa, threshold at 60 m, s2cloudless) can
        # reuse it here for free. Closing the detector first -- as an earlier
        # version of this function did -- forced scene_layers() to reload and
        # re-verify the whole mask from scratch on every single run.
        detector.write_scene_artifacts(meta, out_dir)
    finally:
        detector.close()

    summary = {
        "detector": detector.name,
        "parameters": detector.parameters(),
        "command": " ".join(sys.argv),
        "resampling": {
            "downsample": "exact block mean (equals cv2.INTER_AREA)",
            "upsample_labels": "nearest (np.repeat)",
            "upsample_continuous": "nearest by default; linear reads a 1 px halo",
        },
        "scene": {
            "product_uri": meta.product_uri,
            "processing_baseline": meta.processing_baseline,
            "sensing_time": meta.sensing_time,
            "epsg": meta.epsg,
            "origin_easting": meta.origin(10)[0],
            "origin_northing": meta.origin(10)[1],
            "shape_10m": list(meta.shape(10)),
            "footprint_bbox": list(meta.footprint_bbox()),
            "cloud_coverage_assessment": meta.cloud_coverage_assessment,
        },
        "results": {
            "scene_cloud_percent": scene_percent,
            "valid_tiles": valid,
            "invalid_tiles": len(stats) - valid,
            "jpegs_written": written,
            "cloud_pixels_total": int(sum(s.cloud_pixels for s in stats)),
            "max_nodata_fraction": max((s.nodata_fraction for s in stats), default=0.0),
            "geojson": geojson_info,
        },
        "versions": _versions(),
        "runtime_seconds": round(time.time() - started, 2),
        "peak_memory_mb": round(peak_memory_mb() or 0.0, 1),
    }
    with open(os.path.join(out_dir, "run_summary.json"), "w", encoding="utf8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")

    if not args.skip_checks:
        results = assert_checks(out_dir, meta, IN_PIPELINE_CHECKS)
        for result in results:
            log("  %s" % result)

    log(
        "%s: %.4f %% scene cloud  |  %d valid, %d invalid  |  %d JPEGs  |  %.1f s  |  peak %.0f MB"
        % (detector.name, scene_percent, valid, len(stats) - valid, written,
           summary["runtime_seconds"], summary["peak_memory_mb"])
    )
    return summary


def _write_geojson(meta, detector, out_dir, args, log) -> dict:
    from rasterio.transform import Affine

    try:
        layers, resolution = detector.scene_layers(meta)
    except NotImplementedError:
        log("  (detector provides no scene layers; skipping the GeoJSON bonus)")
        return {}

    ulx, uly = meta.origin(10)
    transform = Affine(float(resolution), 0.0, ulx, 0.0, -float(resolution), uly)
    collection, areas = build_geojson(
        layers, transform, meta.epsg, simplify_m=args.geojson_simplify
    )
    path = write_geojson(os.path.join(out_dir, "cloud_mask.geojson"), collection)
    pixel_area = sum(
        float(np.count_nonzero(mask)) * resolution * resolution for mask in layers.values()
    )
    return {
        "resolution_m": resolution,
        "classes": sorted(layers),
        "features": len(collection["features"]),
        "mask_area_m2": pixel_area,
        "vector_area_m2": sum(areas.values()),
        "simplify_m": args.geojson_simplify,
        "coordinate_decimals": 6,
        "area_tolerance": (
            _AREA_TOLERANCE_SIMPLIFIED if args.geojson_simplify else _AREA_TOLERANCE_ROUNDED
        ),
        "file_bytes": os.path.getsize(path),
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        run(args)
    except (ProductError, detectors.DetectorError) as error:
        print("error: %s" % error, file=sys.stderr)
        return 2
    except CheckFailure as error:
        print("error: %s" % error, file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
