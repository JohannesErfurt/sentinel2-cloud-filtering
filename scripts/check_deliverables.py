"""Validate a pipeline output folder against checks CK1-CK8 (F9).

    python scripts/check_deliverables.py --out output/esa
    python scripts/check_deliverables.py --out output/esa --safe-dir <SAFE>

Without ``--safe-dir`` only the structural checks run -- everything that can be
decided from the folder itself. That is the mode that matters for a deliverable
someone has unzipped on another machine, where the 801 MB .SAFE folder is not
present.

With ``--safe-dir`` the true-colour comparison in CK5 also runs, and ``--rerun``
additionally re-runs the pipeline into a temporary folder to check determinism
(CK8).

ASCII output only, so it prints cleanly on any Windows console code page.
"""
import argparse
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.checks import ALL_CHECKS, check_ck8, run_checks  # noqa: E402
from pipeline.metadata import ProductError, read_product  # noqa: E402

STRUCTURAL = ["CK1", "CK2", "CK3", "CK4", "CK5", "CK6", "CK7"]


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Run deliverable checks CK1-CK8 over a pipeline output folder."
    )
    parser.add_argument("--out", required=True, help="the output folder to validate")
    parser.add_argument(
        "--safe-dir",
        default=None,
        help="the .SAFE product; enables CK5's comparison against the true-colour source",
    )
    parser.add_argument(
        "--rerun",
        action="store_true",
        help="also run CK8 by repeating the run into a temporary folder (needs --safe-dir)",
    )
    parser.add_argument(
        "--checks",
        default=None,
        help="comma-separated subset, e.g. CK2,CK3 (default: all applicable)",
    )
    args = parser.parse_args(argv)

    out_dir = os.path.abspath(args.out)
    if not os.path.isdir(out_dir):
        print("error: not a directory: %s" % out_dir, file=sys.stderr)
        return 2

    meta = None
    if args.safe_dir:
        try:
            meta = read_product(args.safe_dir)
        except ProductError as error:
            print("error: %s" % error, file=sys.stderr)
            return 2

    ids = [c.strip().upper() for c in args.checks.split(",")] if args.checks else list(STRUCTURAL)
    unknown = [c for c in ids if c not in ALL_CHECKS]
    if unknown:
        print("error: unknown checks %s" % ", ".join(unknown), file=sys.stderr)
        return 2

    mode = "full" if meta else "structural"
    print("checking %s (%s)" % (out_dir, mode))
    results = run_checks(out_dir, meta, [c for c in ids if c != "CK8"])

    if args.rerun or "CK8" in ids:
        results.append(_determinism(out_dir, args, meta))

    print()
    for result in results:
        print("  %s" % result)

    failed = [r for r in results if not r.ok]
    passed = [r for r in results if r.ok and not r.skipped]
    skipped = [r for r in results if r.skipped]
    print()
    print(
        "%d passed, %d failed, %d skipped" % (len(passed), len(failed), len(skipped))
    )
    return 1 if failed else 0


def _determinism(out_dir, args, meta):
    from pipeline.checks import CheckResult

    if not args.safe_dir:
        return CheckResult("CK8", True, "determinism needs --safe-dir to re-run", skipped=True)

    summary_path = os.path.join(out_dir, "run_summary.json")
    if not os.path.isfile(summary_path):
        return CheckResult("CK8", False, "run_summary.json missing; cannot repeat the run")
    import json

    with open(summary_path, encoding="utf8") as handle:
        summary = json.load(handle)

    from pipeline.run import build_parser, run

    temporary = tempfile.mkdtemp(prefix="ck8_")
    try:
        argv = ["--safe-dir", args.safe_dir, "--detector", summary["detector"],
                "--out", temporary, "--skip-checks", "--quiet"]
        for flag, key in (("--t-bright", "t_bright"), ("--t-ndsi", "t_ndsi"),
                          ("--t-cirrus", "t_cirrus"), ("--prob-threshold", "prob_threshold"),
                          ("--average-over", "average_over"), ("--dilation-size", "dilation_size"),
                          ("--detection-resolution", "detection_resolution")):
            value = summary.get("parameters", {}).get(key)
            if value is not None:
                argv += [flag, str(value)]
        run(build_parser().parse_args(argv))
        return check_ck8(out_dir, temporary)
    finally:
        shutil.rmtree(temporary, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
