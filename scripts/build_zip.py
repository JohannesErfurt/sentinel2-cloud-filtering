"""Assemble the single deliverable ZIP (S2, R10).

    python scripts/build_zip.py --detector-run output/threshold --esa-run output/esa_new \
        --s2cloudless-run output/s2cloudless --comparison-dir output/comparison \
        --out dist/sentinel2-cloud-filtering.zip

Packs, at the archive root: the chosen detector's ``tiles/`` (JPEGs + world files),
``report.csv``, ``tile_stats.csv``, ``cloud_mask.geojson``, ``run_summary.json``; a
``comparison/`` folder with the other two detectors' ``report.csv`` files plus
``comparison.md``, ``decision.md``, ``reference_notes.md``, the two sweep CSVs, the
audit verdicts and the contact-sheet PNGs; and the source tree needed to reproduce
everything (``README.md``, ``pipeline/``, ``scripts/``, ``tests/``, ``config/``,
``requirements.txt``). No ``.SAFE`` product, no scratch/working files.
"""
import argparse
import os
import sys
import zipfile

MAX_ZIP_BYTES = 50 * 1024 * 1024

#: Source-tree directories copied wholesale, minus __pycache__ and other junk.
SOURCE_DIRS = ["pipeline", "scripts", "tests", "config"]

#: Files from output/comparison/ that belong in the deliverable's comparison/ folder.
COMPARISON_FILES = [
    "comparison.md",
    "decision.md",
    "reference_notes.md",
    "threshold_sweep.csv",
    "s2cloudless_sweep.csv",
    "tiles.png",
    "scene.png",
    "thresholds.png",
    "s2cloudless_candidates.png",
]

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


def build(detector_run, esa_run, s2cloudless_run, comparison_dir, audit_verdicts, out_path):
    runs_by_detector = {}
    for run_dir in (esa_run, s2cloudless_run):
        summary_path = os.path.join(run_dir, "run_summary.json")
        if not os.path.isfile(summary_path):
            raise SystemExit("error: no run_summary.json in %s" % run_dir)
        import json

        with open(summary_path, encoding="utf8") as handle:
            detector = json.load(handle)["detector"]
        runs_by_detector[detector] = run_dir

    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    if os.path.exists(out_path):
        os.remove(out_path)

    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as zf:
        # The chosen detector's own deliverables, at the archive root.
        for name in ("report.csv", "tile_stats.csv", "cloud_mask.geojson", "run_summary.json"):
            src = os.path.join(detector_run, name)
            if not os.path.isfile(src):
                raise SystemExit("error: missing %s in %s" % (name, detector_run))
            zf.write(src, name)
        tiles_dir = os.path.join(detector_run, "tiles")
        for fname in sorted(os.listdir(tiles_dir)):
            zf.write(os.path.join(tiles_dir, fname), "tiles/%s" % fname)

        # The other two detectors' report.csv, for comparison.
        for detector, run_dir in runs_by_detector.items():
            zf.write(os.path.join(run_dir, "report.csv"), "comparison/%s_report.csv" % detector)

        for name in COMPARISON_FILES:
            src = os.path.join(comparison_dir, name)
            if os.path.isfile(src):
                zf.write(src, "comparison/%s" % name)

        zf.write(audit_verdicts, "comparison/audit_verdicts.csv")

        # Source tree needed to reproduce everything.
        zf.write("README.md", "README.md")
        zf.write("requirements.txt", "requirements.txt")
        for source_dir in SOURCE_DIRS:
            for path in _iter_source_files(source_dir):
                zf.write(path, path.replace(os.sep, "/"))

    size = os.path.getsize(out_path)
    if size > MAX_ZIP_BYTES:
        raise SystemExit(
            "error: %s is %.1f MB, over the 50 MB limit" % (out_path, size / 1024 / 1024)
        )
    print("%s (%.1f MB)" % (out_path, size / 1024 / 1024))
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detector-run", required=True, help="the shipped detector's output folder")
    parser.add_argument("--esa-run", required=True)
    parser.add_argument("--s2cloudless-run", required=True)
    parser.add_argument("--comparison-dir", default="output/comparison")
    parser.add_argument("--audit-verdicts", default="audit/verdicts.csv")
    parser.add_argument("--out", default="dist/sentinel2-cloud-filtering.zip")
    args = parser.parse_args(argv)

    return build(
        args.detector_run,
        args.esa_run,
        args.s2cloudless_run,
        args.comparison_dir,
        args.audit_verdicts,
        args.out,
    )


if __name__ == "__main__":
    sys.exit(main())
