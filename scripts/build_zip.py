"""Assemble the single deliverable ZIP (S2, R10).

    python scripts/build_zip.py --detector-run output/threshold --out dist/sentinel2-cloud-filtering.zip

Packs, at the archive root: the chosen detector's ``tiles/`` (the JPEGs only),
``report.csv``, ``cloud_mask.geojson`` and ``run_summary.json``; a ``comparison/``
folder with three contact sheets; and the source tree (``README.md``,
``pipeline/``, ``scripts/``, ``tests/``, ``config/``, ``requirements.txt``). No
``.SAFE`` product, no audit verdicts, no scratch/working files.

The contact sheets are stored as JPEG (quality 85) rather than their original
PNG (a few MB each), which is what keeps the archive under 25 MB without
touching the delivered tiles themselves (quality 90, SPEC.md 1.6).
"""
import argparse
import os
import sys
import zipfile

import cv2

MAX_ZIP_BYTES = 25_000_000
CONTACT_SHEET_JPEG_QUALITY = 85

#: Source-tree directories copied wholesale, minus __pycache__ and other junk.
SOURCE_DIRS = ["pipeline", "scripts", "tests", "config"]

#: Contact-sheet PNGs, written into comparison/ as JPEGs.
CONTACT_SHEETS = ["tiles.png", "scene.png", "thresholds.png"]

#: Junk that must never end up inside the ZIP even if it's sitting in a source dir.
SKIP_SUFFIXES = (".pyc",)
SKIP_DIR_NAMES = {"__pycache__"}


def _iter_source_files(root):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIR_NAMES]
        for name in filenames:
            if name.endswith(SKIP_SUFFIXES):
                continue
            yield os.path.join(dirpath, name)


def build(detector_run, comparison_dir, out_path):
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    if os.path.exists(out_path):
        os.remove(out_path)

    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as zf:
        # The chosen detector's own deliverables, at the archive root.
        for name in ("report.csv", "cloud_mask.geojson", "run_summary.json"):
            src = os.path.join(detector_run, name)
            if not os.path.isfile(src):
                raise SystemExit("error: missing %s in %s" % (name, detector_run))
            zf.write(src, name)
        tiles_dir = os.path.join(detector_run, "tiles")
        for fname in sorted(os.listdir(tiles_dir)):
            if fname.endswith(".jpg"):
                zf.write(os.path.join(tiles_dir, fname), "tiles/%s" % fname)

        for name in CONTACT_SHEETS:
            src = os.path.join(comparison_dir, name)
            if not os.path.isfile(src):
                raise SystemExit("error: missing %s (run scripts/contact_sheet.py first)" % src)
            ok, buf = cv2.imencode(
                ".jpg", cv2.imread(src), [cv2.IMWRITE_JPEG_QUALITY, CONTACT_SHEET_JPEG_QUALITY]
            )
            if not ok:
                raise SystemExit("error: could not re-encode %s" % src)
            zf.writestr("comparison/%s.jpg" % os.path.splitext(name)[0], buf.tobytes(), zipfile.ZIP_STORED)

        zf.write("README.md", "README.md")
        zf.write("requirements.txt", "requirements.txt")
        for source_dir in SOURCE_DIRS:
            for path in _iter_source_files(source_dir):
                zf.write(path, path.replace(os.sep, "/"))

    size = os.path.getsize(out_path)
    if size > MAX_ZIP_BYTES:
        raise SystemExit(
            "error: %s is %.1f MB, over the %.0f MB limit" % (out_path, size / 1e6, MAX_ZIP_BYTES / 1e6)
        )
    print("%s (%.1f MB)" % (out_path, size / 1e6))
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detector-run", required=True, help="the shipped detector's output folder")
    parser.add_argument("--comparison-dir", default="output/comparison")
    parser.add_argument("--out", default="dist/sentinel2-cloud-filtering.zip")
    args = parser.parse_args(argv)
    return build(args.detector_run, args.comparison_dir, args.out)


if __name__ == "__main__":
    sys.exit(main())
