"""Render the E2 audit sample as one 4x4 contact sheet, for visual inspection.

    python scripts/audit_sheet.py --safe-dir <SAFE> --runs output/esa_new output/threshold output/s2cloudless

Reads the tile list from audit/verdicts.csv (already selected and written by
scripts/build_audit_sample.py, before this ever runs) rather than taking its
own tile arguments -- the sheet exists to look at a sample that is already
fixed, not to define one.
"""
import argparse
import csv
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.io import read_tci  # noqa: E402
from pipeline.metadata import ProductError, read_product  # noqa: E402
from pipeline.tiling import tile_window  # noqa: E402

OUTLINE_COLOURS = {"esa": "#ff2d55", "threshold": "#00d4ff", "s2cloudless": "#ffd60a"}


def _load_tile_mask(run_dir, row, col):
    import cv2

    path = os.path.join(run_dir, "tile_masks", "tile_r%02d_c%02d.png" % (row, col))
    if not os.path.isfile(path):
        return None
    image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    return None if image is None else image > 127


def _detector_name(run_dir):
    import json

    with open(os.path.join(run_dir, "run_summary.json"), encoding="utf8") as handle:
        return json.load(handle)["detector"]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--safe-dir", required=True)
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--sample", default="audit/verdicts.csv")
    parser.add_argument("--out", default="audit/audit_sheet.png")
    args = parser.parse_args(argv)

    try:
        meta = read_product(args.safe_dir)
    except ProductError as error:
        print("error: %s" % error, file=sys.stderr)
        return 2

    with open(args.sample, encoding="utf8", newline="") as handle:
        tiles = list(csv.DictReader(handle))
    if len(tiles) != 16:
        print("error: %s has %d rows, expected 16" % (args.sample, len(tiles)), file=sys.stderr)
        return 2

    run_names = {run_dir: _detector_name(run_dir) for run_dir in args.runs}

    cols, rows = 4, 4
    figure, axes = plt.subplots(rows, cols, figsize=(cols * 3.4, rows * 3.7), dpi=150)
    for index, entry in enumerate(tiles):
        row, col = int(entry["row"]), int(entry["col"])
        axis = axes[index // cols, index % cols]
        axis.imshow(read_tci(meta, tile_window(row, col)))
        for run_dir in args.runs:
            mask = _load_tile_mask(run_dir, row, col)
            if mask is not None and mask.any():
                colour = OUTLINE_COLOURS.get(run_names[run_dir], "#ffffff")
                axis.contour(mask.astype(float), levels=[0.5], colors=[colour], linewidths=0.8)
        label = entry["character"] or entry["reason"]
        axis.set_title("tile %d,%d\n%s" % (row, col, label), fontsize=7)
        axis.set_xticks([])
        axis.set_yticks([])

    from matplotlib.lines import Line2D

    handles = [Line2D([], [], color=c, linewidth=1.5, label=n) for n, c in OUTLINE_COLOURS.items() if n in run_names.values()]
    figure.legend(handles=handles, loc="lower center", ncol=len(handles), fontsize=9, bbox_to_anchor=(0.5, -0.01))
    figure.tight_layout(rect=[0, 0.02, 1, 1])

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    figure.savefig(args.out, bbox_inches="tight")
    plt.close(figure)
    size = os.path.getsize(args.out)
    print("%s  %.1f MB" % (args.out, size / 1e6))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
