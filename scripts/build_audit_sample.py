"""Select the E2 visual-audit sample and write it to audit/verdicts.csv,
with the verdict/notes columns empty.

    python scripts/build_audit_sample.py --threshold-run output/threshold --s2cloudless-run output/s2cloudless

This is a deliberately separate step from actually filling the verdicts in:
the sample (SPEC.md E2's four named tiles plus the 12 tiles where `threshold`
and `s2cloudless` disagree most in cloud_cover_percent) is fixed by this
script, from numbers alone, before anyone looks at a single image. Recording
it first and looking second is the whole point -- it is what stops the
sample being quietly steered toward whichever tiles make a preferred
detector look good.
"""
import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.compare import load_runs, select_audit_sample  # noqa: E402

FIELDS = ["row", "col", "character", "reason", "verdict", "notes"]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threshold-run", default="output/threshold")
    parser.add_argument("--s2cloudless-run", default="output/s2cloudless")
    parser.add_argument("--out", default="audit/verdicts.csv")
    args = parser.parse_args(argv)

    if os.path.isfile(args.out):
        print("error: %s already exists -- refusing to overwrite an existing audit" % args.out, file=sys.stderr)
        print("(delete it first if you genuinely want to reselect the sample)", file=sys.stderr)
        return 2

    threshold_run, s2c_run = load_runs([args.threshold_run, args.s2cloudless_run])
    sample = select_audit_sample(threshold_run, s2c_run)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        for entry in sample:
            writer.writerow({
                "row": entry["row"], "col": entry["col"], "character": entry["character"],
                "reason": entry["reason"], "verdict": "", "notes": "",
            })

    print("%s (%d tiles selected; verdict column left empty)" % (args.out, len(sample)))
    for entry in sample:
        print("  %2d,%-2d  %s" % (entry["row"], entry["col"], entry["reason"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
