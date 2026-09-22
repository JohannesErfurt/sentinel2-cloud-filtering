"""Contact sheets: the way this project looks at a cloud mask (F9).

Three PNGs replace an interactive viewer. They are cheap to produce, they drop
straight into the README, and they can be diffed between runs -- none of which a
browser widget manages.

    python scripts/contact_sheet.py --safe-dir <SAFE> --tiles      --runs output/esa output/threshold
    python scripts/contact_sheet.py --safe-dir <SAFE> --scene      --runs output/threshold
    python scripts/contact_sheet.py --safe-dir <SAFE> --thresholds

``--tiles`` and ``--scene`` read the per-tile masks a run wrote with
``--save-tile-masks``, so they work with any detector. ``--thresholds`` renders
the same tiles at four threshold settings and needs the threshold backend (B2).

ASCII output only.
"""
import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.constants import GRID, TILE_PX  # noqa: E402
from pipeline.io import block_mean, read_tci  # noqa: E402
from pipeline.metadata import ProductError, read_product  # noqa: E402
from pipeline.tiling import tile_window  # noqa: E402

#: The four tiles SPEC.md 5.1 names, and why each is worth looking at.
NAMED_TILES = [
    (11, 2, "clear"),
    (6, 5, "moderate"),
    (5, 16, "heavy"),
    (12, 19, "thin veil"),
]

OUTLINE_COLOURS = ["#ff2d55", "#00d4ff", "#ffd60a", "#32d74b"]


def _load_tile_mask(run_dir, row, col):
    import cv2

    path = os.path.join(run_dir, "tile_masks", "tile_r%02d_c%02d.png" % (row, col))
    if not os.path.isfile(path):
        return None
    image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    return None if image is None else image > 127


def _outline(axis, mask, colour):
    """Draw a mask as contour lines so the imagery underneath stays visible.

    Silently does nothing on an empty mask (a genuinely clear tile) -- which is
    exactly why the legend must not depend on any one axis having drawn
    something: the first named tile (11,2) is the clear one.
    """
    if mask is None or not mask.any():
        return
    axis.contour(mask.astype(float), levels=[0.5], colors=[colour], linewidths=0.7)


def _legend_handles(names):
    """Proxy artists for the legend, independent of which axis drew a mask."""
    from matplotlib.lines import Line2D

    return [
        Line2D([], [], color=OUTLINE_COLOURS[i % len(OUTLINE_COLOURS)], linewidth=1.5, label=name)
        for i, name in enumerate(names)
    ]


def sheet_tiles(meta, runs, tiles, out_path):
    """True colour per tile, with each run's mask outlined in its own colour."""
    names = [_detector_name(r) for r in runs]
    columns = len(tiles)
    figure, axes = plt.subplots(1, columns, figsize=(3.2 * columns, 3.7), dpi=160)
    axes = np.atleast_1d(axes)
    for axis, (row, col, note) in zip(axes, tiles):
        rgb = read_tci(meta, tile_window(row, col))
        axis.imshow(rgb)
        for index, run_dir in enumerate(runs):
            _outline(axis, _load_tile_mask(run_dir, row, col), OUTLINE_COLOURS[index % len(OUTLINE_COLOURS)])
        axis.set_title("tile %d,%d\n%s" % (row, col, note), fontsize=9)
        axis.set_xticks([])
        axis.set_yticks([])
    if runs:
        axes[0].legend(handles=_legend_handles(names), loc="lower left", fontsize=7, framealpha=0.8)
    figure.tight_layout()
    figure.savefig(out_path, bbox_inches="tight")
    plt.close(figure)
    return out_path


def sheet_scene(meta, run_dir, out_path):
    """The whole scene at 60 m, with the mask, the 20x20 grid and invalid tiles."""
    from pipeline.report import read_report_csv

    rgb = read_tci(meta)
    small = np.dstack([block_mean(rgb[:, :, c], 6) for c in range(3)]).astype(np.uint8)

    # Assemble at 10 m, then reduce by 6: a tile is 91.5 px at 60 m, so placing
    # tiles directly on the 60 m grid would drift up to 10 px by the last tile.
    full = np.zeros((GRID * TILE_PX, GRID * TILE_PX), dtype=bool)
    for row in range(GRID):
        for col in range(GRID):
            mask = _load_tile_mask(run_dir, row, col)
            if mask is not None:
                full[row * TILE_PX : (row + 1) * TILE_PX, col * TILE_PX : (col + 1) * TILE_PX] = mask
    n60 = full.shape[0] // 6
    scene_mask = full.reshape(n60, 6, n60, 6).sum(axis=(1, 3), dtype=np.uint8) > 18
    del full

    figure, axis = plt.subplots(figsize=(10, 10), dpi=160)
    axis.imshow(small)
    overlay = np.zeros(small.shape[:2] + (4,), dtype=float)
    overlay[scene_mask] = (1.0, 0.18, 0.33, 0.45)
    axis.imshow(overlay)

    report = os.path.join(run_dir, "report.csv")
    invalid = set()
    if os.path.isfile(report):
        _, rows = read_report_csv(report)
        invalid = {(i // GRID, i % GRID) for i, r in enumerate(rows) if r["valid"] == "False"}

    height = small.shape[0]
    for index in range(GRID + 1):
        position = index * height / GRID
        axis.axhline(position, color="white", linewidth=0.4, alpha=0.5)
        axis.axvline(position, color="white", linewidth=0.4, alpha=0.5)
    for row, col in invalid:
        axis.add_patch(
            plt.Rectangle(
                (col * height / GRID, row * height / GRID),
                height / GRID, height / GRID,
                facecolor="black", alpha=0.35, edgecolor="none",
            )
        )
    title = "%s -- mask in red, %d of 400 tiles discarded (shaded)" % (_detector_name(run_dir), len(invalid))
    parameters = _run_parameters(run_dir)
    if parameters:
        title += "\n" + ", ".join("%s = %s" % (k, v) for k, v in sorted(parameters.items()))
    axis.set_title(title, fontsize=11)
    axis.set_xlim(-0.5, small.shape[1] - 0.5)
    axis.set_ylim(small.shape[0] - 0.5, -0.5)
    axis.set_xticks([])
    axis.set_yticks([])
    figure.tight_layout()
    figure.savefig(out_path, bbox_inches="tight")
    plt.close(figure)
    return out_path


def _run_summary(run_dir):
    path = os.path.join(run_dir, "run_summary.json")
    if not os.path.isfile(path):
        return {}
    import json

    with open(path, encoding="utf8") as handle:
        return json.load(handle)


def _detector_name(run_dir):
    return _run_summary(run_dir).get("detector") or os.path.basename(os.path.normpath(run_dir))


def _run_parameters(run_dir):
    return {k: v for k, v in _run_summary(run_dir).get("parameters", {}).items() if v is not None}


def sheet_thresholds(meta, tiles, out_path):
    """The named tiles at five threshold settings, including the shipped one."""
    try:
        from pipeline.detectors import build
        from pipeline.detectors.threshold import load_config
    except ImportError:  # pragma: no cover
        raise SystemExit("pipeline.detectors is unavailable")

    config = load_config()
    shipped = {k: config[k] for k in ("t_bright", "t_ndsi", "t_cirrus")}
    settings = [
        {"t_bright": 0.33, "t_ndsi": -1.0, "t_cirrus": 1.0},
        {"t_bright": 0.20, "t_ndsi": -0.20, "t_cirrus": 0.005},
        shipped,
        {"t_bright": 0.16, "t_ndsi": -0.20, "t_cirrus": 0.005},
        {"t_bright": 0.14, "t_ndsi": -0.20, "t_cirrus": 0.004},
    ]
    try:
        instances = [build("threshold", **s) for s in settings]
    except Exception as error:
        raise SystemExit(
            "--thresholds needs the threshold backend (task B2 in SPEC.md): %s" % error
        )

    figure, axes = plt.subplots(
        len(settings), len(tiles), figsize=(3.0 * len(tiles), 3.2 * len(settings)), dpi=135
    )
    axes = np.atleast_2d(axes)
    for r, (detector, setting) in enumerate(zip(instances, settings)):
        for c, (row, col, note) in enumerate(tiles):
            axis = axes[r, c]
            axis.imshow(read_tci(meta, tile_window(row, col)))
            mask = detector.tile_mask(meta, row, col)
            _outline(axis, mask, OUTLINE_COLOURS[0])
            axis.set_xticks([])
            axis.set_yticks([])
            if r == 0:
                axis.set_title("tile %d,%d (%s)" % (row, col, note), fontsize=9)
            if c == 0:
                label = "b>%.2f ndsi>%.2f\nB10>%.4f" % (setting["t_bright"], setting["t_ndsi"], setting["t_cirrus"])
                if setting is shipped:
                    label = "SHIPPED (config/thresholds.json)\n" + label
                axis.set_ylabel(label, fontsize=8, fontweight="bold" if setting is shipped else "normal")
            axis.text(
                4, 24, "%.1f %%" % (100.0 * mask.mean()),
                color="white", fontsize=8,
                bbox=dict(facecolor="black", alpha=0.5, pad=1.5, edgecolor="none"),
            )
    figure.tight_layout()
    figure.savefig(out_path, bbox_inches="tight")
    plt.close(figure)
    return out_path


def main(argv=None):
    parser = argparse.ArgumentParser(description="Render contact sheets for a cloud mask.")
    parser.add_argument("--safe-dir", required=True)
    parser.add_argument("--runs", nargs="*", default=[], help="pipeline output folders")
    parser.add_argument("--out-dir", default="output/comparison")
    parser.add_argument("--tiles", action="store_true", help="named tiles with each run's outline")
    parser.add_argument("--scene", action="store_true", help="whole scene overview from one run")
    parser.add_argument("--thresholds", action="store_true", help="named tiles at five threshold sets")
    parser.add_argument(
        "--tile", action="append", default=[], metavar="ROW,COL",
        help="add a tile to the sheet; repeatable",
    )
    args = parser.parse_args(argv)

    if not (args.tiles or args.scene or args.thresholds):
        parser.error("choose at least one of --tiles, --scene, --thresholds")

    try:
        meta = read_product(args.safe_dir)
    except ProductError as error:
        print("error: %s" % error, file=sys.stderr)
        return 2

    tiles = list(NAMED_TILES)
    for entry in args.tile:
        row, col = (int(v) for v in entry.split(","))
        tiles.append((row, col, "requested"))

    os.makedirs(args.out_dir, exist_ok=True)
    written = []
    if args.tiles:
        written.append(sheet_tiles(meta, args.runs, tiles, os.path.join(args.out_dir, "tiles.png")))
    if args.scene:
        if not args.runs:
            parser.error("--scene needs one --runs folder")
        written.append(sheet_scene(meta, args.runs[0], os.path.join(args.out_dir, "scene.png")))
    if args.thresholds:
        written.append(sheet_thresholds(meta, tiles, os.path.join(args.out_dir, "thresholds.png")))

    for path in written:
        size = os.path.getsize(path)
        print("%s  %.1f MB%s" % (path, size / 1e6, "  TOO LARGE (>5 MB)" if size > 5e6 else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
