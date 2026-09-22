"""Apply SPEC.md 5.3's rule and write output/comparison/decision.md (E3).

    python scripts/decide.py --runs output/esa_new output/threshold output/s2cloudless --audit-verdicts audit/verdicts.csv

The rule (fixed in SPEC.md 0.4, before this script or the audit ever ran):

  1. `esa` is excluded -- it is a baseline, not detection.
  2. The winner is whichever of `threshold` and `s2cloudless` agrees better
     with the E2 audit.
  3. If they are within one tile of sixteen, `threshold` wins on simplicity.

This script does not invent a fourth rule, and does not look at any number
not already produced by pipeline.compare -- it just applies the three steps
above mechanically and writes down which one fired.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.compare import (  # noqa: E402
    ComparisonError,
    audit_agreement,
    find_reference,
    load_audit_verdicts,
    load_runs,
    tile_level,
)

#: SPEC.md 5.3's own tie margin: "within one tile of sixteen". The audit is
#: 16 tiles but scores 15 (one thin_cloud tile excluded, decision D7) -- the
#: margin is applied as a tile count, not a fraction, so it means the same
#: thing either way: a one-tile difference in how many of the scored tiles
#: each candidate got right.
TIE_MARGIN_TILES = 1


def decide(runs, verdicts):
    reference = find_reference(runs)
    if reference is None:
        raise ComparisonError("no esa run given -- step 1 of the rule needs it as the excluded baseline")

    candidates = {r.detector: r for r in runs if r.detector != "esa"}
    missing = {"threshold", "s2cloudless"} - set(candidates)
    if missing:
        raise ComparisonError("missing candidate run(s) for: %s" % ", ".join(sorted(missing)))

    threshold_run = candidates["threshold"]
    s2c_run = candidates["s2cloudless"]

    esa_agreement = {}
    audit_result = {}
    for name, run in (("threshold", threshold_run), ("s2cloudless", s2c_run)):
        stats = tile_level(reference, run)
        esa_agreement[name] = 1 - stats["disagreements"] / len(reference.rows)
        audit_result[name] = audit_agreement(run, verdicts)

    scored = audit_result["threshold"]["scored"]
    threshold_matches = audit_result["threshold"]["matches"]
    s2c_matches = audit_result["s2cloudless"]["matches"]

    gap = abs(threshold_matches - s2c_matches)
    if gap <= TIE_MARGIN_TILES:
        winner = "threshold"
        rule = "step 3 (within %d tile(s) of %d scored -- simplicity tie-break)" % (TIE_MARGIN_TILES, scored)
    elif threshold_matches > s2c_matches:
        winner = "threshold"
        rule = "step 2 (higher audit agreement)"
    else:
        winner = "s2cloudless"
        rule = "step 2 (higher audit agreement)"

    return {
        "winner": winner,
        "rule": rule,
        "scored": scored,
        "esa_agreement": esa_agreement,
        "audit_result": audit_result,
        "reference_invalid": reference.invalid_count(),
    }


def build_report(decision: dict) -> str:
    lines = ["# Which detector ships (E3, decision D5)", ""]
    lines.append(
        "**The rule was fixed in SPEC.md 0.4, before this evaluation ran:**"
    )
    lines.append("")
    lines.append("1. `esa` is excluded -- it is a baseline, not detection.")
    lines.append("2. The winner is whichever of `threshold` and `s2cloudless` agrees better with the E2 audit.")
    lines.append("3. If they are within %d tile of the %d scored, `threshold` wins on simplicity."
                 % (TIE_MARGIN_TILES, decision["scored"]))
    lines.append("")
    lines.append("No fourth rule was added after seeing the numbers below, and no rule was reworded to fit them.")
    lines.append("")

    lines.append("## The numbers both rules needed")
    lines.append("")
    lines.append("| Candidate | Agreement with ESA (400 tiles) | Agreement with audit (%d scored) | Audit matches |"
                 % decision["scored"])
    lines.append("|---|---|---|---|")
    for name in ("threshold", "s2cloudless"):
        esa_pct = 100 * decision["esa_agreement"][name]
        audit = decision["audit_result"][name]
        audit_pct = 100 * audit["agreement"] if audit["agreement"] is not None else float("nan")
        lines.append(
            "| %s | %.1f %% | %.1f %% | %d / %d |"
            % (name, esa_pct, audit_pct, audit["matches"], decision["scored"])
        )
    lines.append("")

    gap = abs(decision["audit_result"]["threshold"]["matches"] - decision["audit_result"]["s2cloudless"]["matches"])
    lines.append(
        "The audit gap is %d tile(s) out of %d scored -- %s the %d-tile tie margin."
        % (gap, decision["scored"], "within" if gap <= TIE_MARGIN_TILES else "outside", TIE_MARGIN_TILES)
    )
    lines.append("")

    lines.append("## Verdict")
    lines.append("")
    lines.append("**`%s` ships**, decided by %s." % (decision["winner"], decision["rule"]))
    lines.append("")
    if decision["winner"] != "s2cloudless":
        lines.append(
            "`s2cloudless` does not ship, but the ML bonus (R7) is satisfied regardless: it is fully "
            "implemented as a backend (`--detector s2cloudless`), it was tuned deliberately (B3.3), and "
            "it is evaluated here on equal footing with the detector that does ship."
        )
        lines.append("")

    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--audit-verdicts", default="audit/verdicts.csv")
    parser.add_argument("--out", default="output/comparison/decision.md")
    args = parser.parse_args(argv)

    try:
        runs = load_runs(args.runs)
        verdicts = load_audit_verdicts(args.audit_verdicts)
        decision = decide(runs, verdicts)
    except ComparisonError as error:
        print("error: %s" % error, file=sys.stderr)
        return 2

    report = build_report(decision)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf8") as handle:
        handle.write(report)
    print("%s -- %s ships (%s)" % (args.out, decision["winner"], decision["rule"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
