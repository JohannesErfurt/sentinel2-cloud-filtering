"""The threshold sweep SPEC.md B2.2 asks for -- reference only, never the target.

    python scripts/threshold_sweep.py --safe-dir <SAFE> --out output/comparison/threshold_sweep.csv

Covers brightness 0.10-0.40, NDSI (off and -0.40 to -0.10) and B10 (off and
0.002-0.012), scored against the `esa` mask at 60 m with the block-mean
kernel. This is a reference point for the choice recorded in
config/thresholds.json (B2.2), not how that choice was made -- ESA's mask is
a baseline, not the target (SPEC.md 3.2).

The five bands this needs are read once, at 60 m, and every grid point after
that is pure boolean array arithmetic on the cached arrays -- reading DN 11.6k
times would dominate the runtime for no reason.
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.detectors.esa import read_classi_mask  # noqa: E402
from pipeline.io import read_reflectance  # noqa: E402
from pipeline.masks import brightness as brightness_fn  # noqa: E402
from pipeline.masks import ndsi as ndsi_fn  # noqa: E402
from pipeline.metadata import ProductError, read_product  # noqa: E402
from pipeline.metrics import agreement  # noqa: E402

SHAPE_60M = (1830, 1830)

BRIGHT_GRID = np.round(np.arange(0.10, 0.40 + 1e-9, 0.01), 4)
NDSI_GRID = np.round(np.arange(-0.40, -0.10 + 1e-9, 0.02), 4)  # "on" values; "off" is separate
CIRRUS_GRID = np.round(np.arange(0.002, 0.012 + 1e-9, 0.0005), 5)  # "on" values; "off" is separate

# SPEC.md 3.2's reference figures, reproduced within +/-0.02 (auto criterion).
REFERENCE_F1 = {
    "brightness_only": 0.71,
    "plus_ndsi": 0.72,
    "plus_cirrus": 0.72,
    "all_three": 0.72,
}


def build_layers(meta):
    """Every boolean layer the sweep needs, precomputed once."""
    b02 = read_reflectance(meta, "B02", out_shape=SHAPE_60M)
    b03 = read_reflectance(meta, "B03", out_shape=SHAPE_60M)
    b04 = read_reflectance(meta, "B04", out_shape=SHAPE_60M)
    b11 = read_reflectance(meta, "B11", out_shape=SHAPE_60M)
    b10 = read_reflectance(meta, "B10", out_shape=SHAPE_60M)
    bright = brightness_fn(b02, b03, b04)
    n = ndsi_fn(b03, b11)

    bright_masks = {float(t): bright > np.float32(t) for t in BRIGHT_GRID}
    ndsi_masks = {float(t): n > np.float32(t) for t in NDSI_GRID}
    ndsi_masks["off"] = np.ones(SHAPE_60M, dtype=bool)  # the veto disabled: always passes
    cirrus_masks = {float(t): b10 > np.float32(t) for t in CIRRUS_GRID}
    cirrus_masks["off"] = np.zeros(SHAPE_60M, dtype=bool)  # the branch disabled: never fires

    return bright_masks, ndsi_masks, cirrus_masks


def run_sweep(meta):
    opaque, cirrus, _snow = read_classi_mask(meta)
    reference = opaque | cirrus

    bright_masks, ndsi_masks, cirrus_masks = build_layers(meta)

    rows = []
    for t_bright, bmask in bright_masks.items():
        for t_ndsi, nmask in ndsi_masks.items():
            for t_cirrus, cmask in cirrus_masks.items():
                cloud = (bmask & nmask) | cmask
                result = agreement(cloud, reference)
                rows.append(
                    {
                        "t_bright": t_bright,
                        "t_ndsi": t_ndsi,
                        "t_cirrus": t_cirrus,
                        "scene_cloud_percent": 100.0 * float(cloud.mean()),
                        "precision": result.precision,
                        "recall": result.recall,
                        "f1": result.f1,
                        "iou": result.iou,
                    }
                )
    return rows


def _best(rows, predicate):
    candidates = [r for r in rows if predicate(r)]
    return max(candidates, key=lambda r: r["f1"])


def check_ablation(rows):
    """The four rows SPEC.md 3.2's ablation table reports, and the boundary
    check: none of their optima may sit on the edge of the grid they were
    swept over, or the sweep has not found the optimum (SPEC.md 3.2's own
    warning, applied to itself)."""
    brightness_only = _best(rows, lambda r: r["t_ndsi"] == "off" and r["t_cirrus"] == "off")
    plus_ndsi = _best(rows, lambda r: r["t_ndsi"] != "off" and r["t_cirrus"] == "off")
    plus_cirrus = _best(rows, lambda r: r["t_ndsi"] == "off" and r["t_cirrus"] != "off")
    all_three = _best(rows, lambda r: r["t_ndsi"] != "off" and r["t_cirrus"] != "off")

    ablation = {
        "brightness_only": brightness_only,
        "plus_ndsi": plus_ndsi,
        "plus_cirrus": plus_cirrus,
        "all_three": all_three,
    }

    edges = {
        "t_bright": (float(BRIGHT_GRID.min()), float(BRIGHT_GRID.max())),
        "t_ndsi": (float(NDSI_GRID.min()), float(NDSI_GRID.max())),
        "t_cirrus": (float(CIRRUS_GRID.min()), float(CIRRUS_GRID.max())),
    }
    problems = []
    for name, row in ablation.items():
        for param, (lo, hi) in edges.items():
            value = row[param]
            if value in ("off",):
                continue
            if abs(value - lo) < 1e-9 or abs(value - hi) < 1e-9:
                problems.append(
                    "%s: %s = %s sits on the edge of its swept range [%.4g, %.4g] -- "
                    "widen the grid, this has not found the optimum" % (name, param, value, lo, hi)
                )
        expected = REFERENCE_F1[name]
        if abs(row["f1"] - expected) > 0.02:
            problems.append(
                "%s: F1 %.3f is more than 0.02 from SPEC.md 3.2's reference %.2f"
                % (name, row["f1"], expected)
            )
    return ablation, problems


def write_csv(path, rows):
    import csv

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fields = ["t_bright", "t_ndsi", "t_cirrus", "scene_cloud_percent", "precision", "recall", "f1", "iou"]
    with open(path, "w", encoding="utf8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            formatted = dict(row)
            for key in ("scene_cloud_percent", "precision", "recall", "f1", "iou"):
                formatted[key] = "%.6f" % formatted[key]
            writer.writerow(formatted)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--safe-dir", required=True)
    parser.add_argument("--out", default="output/comparison/threshold_sweep.csv")
    args = parser.parse_args(argv)

    try:
        meta = read_product(args.safe_dir)
    except ProductError as error:
        print("error: %s" % error, file=sys.stderr)
        return 2

    started = time.time()
    print(
        "sweeping %d brightness x %d NDSI x %d B10 = %d combinations..."
        % (len(BRIGHT_GRID), len(NDSI_GRID) + 1, len(CIRRUS_GRID) + 1,
           len(BRIGHT_GRID) * (len(NDSI_GRID) + 1) * (len(CIRRUS_GRID) + 1))
    )
    rows = run_sweep(meta)
    write_csv(args.out, rows)
    print("%s (%d rows, %.1f s)" % (args.out, len(rows), time.time() - started))

    ablation, problems = check_ablation(rows)
    print()
    print("%-16s %8s %10s %10s %10s %8s" % ("", "t_bright", "t_ndsi", "t_cirrus", "cloud%", "F1"))
    for name, row in ablation.items():
        print(
            "%-16s %8.2f %10s %10s %8.2f%% %8.3f"
            % (name, row["t_bright"], row["t_ndsi"], row["t_cirrus"], row["scene_cloud_percent"], row["f1"])
        )

    if problems:
        print()
        for problem in problems:
            print("PROBLEM: %s" % problem)
        return 1
    print()
    print("optimum lies inside the grid for all four rows; F1 within 0.02 of SPEC.md 3.2.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
