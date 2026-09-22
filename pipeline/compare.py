"""Cross-detector comparison (E1): SPEC.md 5.1's protocol, run over any subset
of pipeline output folders.

    python -m pipeline.compare --runs output/esa output/threshold output/s2cloudless
    python -m pipeline.compare --runs output/esa output/threshold --out output/comparison/comparison.md

ESA is identified by its ``run_summary.json``'s own ``detector`` field, not by
folder name -- a run folder can be called anything. When an `esa` run is among
those given, it is the reference every other run is compared against
(SPEC.md 5.1); it is a baseline, not ground truth, and every report this
module writes says so.

Pixel-level precision/recall/F1/IoU need per-pixel masks, which report.csv
does not carry -- they come from ``tile_masks/*.png``, written when a run used
``--save-tile-masks``. Tile-level and named-tile figures only need
``report.csv`` and work for any run, with or without that flag.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import cv2
import numpy as np

from .constants import GRID, SCENE_PX, TILE_PX
from .metrics import Agreement, agreement
from .report import read_report_csv

#: SPEC.md 0.1/3.5/5.1's four named tiles, reused everywhere in this project.
NAMED_TILES = [
    (11, 2, "genuinely clear"),
    (6, 5, "moderate cloud"),
    (5, 16, "heavy cloud"),
    (12, 19, "thin cirrus veil, the tile that flips (SPEC.md 3.5)"),
]

#: The naive `brightness > 0.33` rule's figures, measured against the real
#: product elsewhere in this project (SPEC.md 0.3, 3.5) and fixed constants
#: of this one scene -- not recomputed here, since doing so would need
#: --safe-dir, and this module's job is to compare deliverables that already
#: exist, not to re-run detection.
NAIVE_RULE_REFERENCE = {
    "label": "brightness > 0.33 (naive, reference only)",
    "invalid_tiles": 35,
    "scene_cloud_percent": 13.6,
}


class ComparisonError(RuntimeError):
    """A run folder is missing something this module needs."""


class RunData:
    """One pipeline output folder: its report, summary, and (if present) the
    per-tile masks needed for pixel-level comparison."""

    def __init__(self, path: str):
        self.path = os.path.abspath(path)
        self.name = os.path.basename(os.path.normpath(path))

        summary_path = os.path.join(self.path, "run_summary.json")
        if not os.path.isfile(summary_path):
            raise ComparisonError("%s has no run_summary.json -- is this a pipeline output folder?" % self.path)
        with open(summary_path, encoding="utf8") as handle:
            self.summary = json.load(handle)
        self.detector = self.summary.get("detector", self.name)

        report_path = os.path.join(self.path, "report.csv")
        if not os.path.isfile(report_path):
            raise ComparisonError("%s has no report.csv" % self.path)
        _, self.rows = read_report_csv(report_path)
        if len(self.rows) != GRID * GRID:
            raise ComparisonError("%s/report.csv has %d rows, expected %d" % (self.path, len(self.rows), GRID * GRID))

        self._mask: np.ndarray | None = None

    def valid(self, index: int) -> bool:
        return self.rows[index]["valid"] == "True"

    def cloud_percent(self, index: int) -> float:
        return float(self.rows[index]["cloud_cover_percent"])

    def invalid_count(self) -> int:
        return sum(1 for row in self.rows if row["valid"] == "False")

    def scene_cloud_percent(self) -> float | None:
        return self.summary.get("results", {}).get("scene_cloud_percent")

    def has_tile_masks(self) -> bool:
        return os.path.isdir(os.path.join(self.path, "tile_masks"))

    def full_mask(self) -> np.ndarray:
        """Reassemble the 10 m scene mask from ``tile_masks/*.png``. Cached."""
        if self._mask is not None:
            return self._mask
        masks_dir = os.path.join(self.path, "tile_masks")
        if not os.path.isdir(masks_dir):
            raise ComparisonError(
                "%s has no tile_masks/ -- rerun with --save-tile-masks for pixel-level comparison" % self.path
            )
        mask = np.zeros((SCENE_PX, SCENE_PX), dtype=bool)
        for row in range(GRID):
            for col in range(GRID):
                tile_path = os.path.join(masks_dir, "tile_r%02d_c%02d.png" % (row, col))
                tile = cv2.imread(tile_path, cv2.IMREAD_UNCHANGED)
                if tile is None:
                    raise ComparisonError("missing or unreadable %s" % tile_path)
                mask[row * TILE_PX : (row + 1) * TILE_PX, col * TILE_PX : (col + 1) * TILE_PX] = tile > 127
        self._mask = mask
        return mask


def load_runs(paths: list[str]) -> list[RunData]:
    return [RunData(p) for p in paths]


def find_reference(runs: list[RunData]) -> RunData | None:
    """The `esa` run among ``runs``, if one was given."""
    for run in runs:
        if run.detector == "esa":
            return run
    return None


def pixel_level(reference: RunData, other: RunData) -> Agreement:
    return agreement(other.full_mask(), reference.full_mask())


def tile_level(reference: RunData, other: RunData) -> dict:
    """Invalid-tile count, disagreement count, and its direction.

    SPEC.md 5.1: "A method that drops tiles ESA keeps behaves very differently
    from one that keeps tiles ESA drops" -- both directions are reported
    separately, never folded into one "disagreement" number.
    """
    disagreements = 0
    discards_where_esa_keeps = 0
    keeps_where_esa_discards = 0
    for index in range(len(reference.rows)):
        ref_valid, other_valid = reference.valid(index), other.valid(index)
        if ref_valid == other_valid:
            continue
        disagreements += 1
        if ref_valid and not other_valid:
            discards_where_esa_keeps += 1
        else:
            keeps_where_esa_discards += 1
    return {
        "invalid_tiles": other.invalid_count(),
        "disagreements": disagreements,
        "discards_where_esa_keeps": discards_where_esa_keeps,
        "keeps_where_esa_discards": keeps_where_esa_discards,
    }


def select_audit_sample(candidate_a: RunData, candidate_b: RunData) -> list[dict]:
    """The 16-tile sample E2 asks for: the four named tiles, plus the 12
    tiles (of the remaining 396) where ``candidate_a`` and ``candidate_b``
    disagree most in ``cloud_cover_percent``.

    A stated, reproducible rule with no human judgement in the selection
    itself -- the point is to fix *which* tiles get looked at before anyone
    looks at them, so the sample cannot be quietly steered toward whichever
    tiles make a preferred detector look good.
    """
    named = {(row, col) for row, col, _ in NAMED_TILES}
    diffs = []
    for row in range(GRID):
        for col in range(GRID):
            if (row, col) in named:
                continue
            index = row * GRID + col
            diff = abs(candidate_a.cloud_percent(index) - candidate_b.cloud_percent(index))
            diffs.append((diff, row, col))
    diffs.sort(key=lambda item: item[0], reverse=True)

    sample = [
        {"row": row, "col": col, "character": character, "reason": "named tile"}
        for row, col, character in NAMED_TILES
    ]
    for diff, row, col in diffs[:12]:
        sample.append(
            {
                "row": row,
                "col": col,
                "character": "",
                "reason": "top-12 disagreement (%.4f pp between %s and %s)" % (diff, candidate_a.detector, candidate_b.detector),
            }
        )
    return sample


def named_tile_table(runs: list[RunData]) -> list[dict]:
    rows = []
    for row, col, character in NAMED_TILES:
        index = row * GRID + col
        entry = {"row": row, "col": col, "character": character}
        for run in runs:
            entry[run.name] = run.cloud_percent(index)
        rows.append(entry)
    return rows


#: A human "cloud" verdict is scored as expecting the tile to be discarded
#: (invalid), "clear" as expecting it kept (valid). "thin_cloud" is
#: deliberately excluded: SPEC.md decision D7 ("how to treat thin cloud") is
#: open, so there is no defined right answer to score a detector against for
#: those tiles -- folding them in would silently pre-decide D7 inside a
#: scoring function instead of leaving it as the open decision it is.
_VERDICT_EXPECTS_VALID = {"cloud": False, "clear": True}


def load_audit_verdicts(path: str) -> list[dict]:
    import csv

    with open(path, encoding="utf8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 16:
        raise ComparisonError("%s has %d rows, expected 16 (SPEC.md E2)" % (path, len(rows)))
    for row in rows:
        if row["verdict"] not in ("cloud", "thin_cloud", "clear"):
            raise ComparisonError("%s: tile (%s,%s) has verdict %r, expected cloud/thin_cloud/clear"
                                   % (path, row["row"], row["col"], row["verdict"]))
    return rows


def audit_agreement(run: RunData, verdicts: list[dict]) -> dict:
    """How often ``run``'s valid/invalid call matches the human verdicts,
    scored only on the tiles with a defined expected answer (cloud or clear,
    not thin_cloud -- see `_VERDICT_EXPECTS_VALID`)."""
    scored = 0
    matches = 0
    excluded = 0
    mismatches = []
    for entry in verdicts:
        verdict = entry["verdict"]
        if verdict not in _VERDICT_EXPECTS_VALID:
            excluded += 1
            continue
        row, col = int(entry["row"]), int(entry["col"])
        index = row * GRID + col
        expected_valid = _VERDICT_EXPECTS_VALID[verdict]
        actual_valid = run.valid(index)
        scored += 1
        if actual_valid == expected_valid:
            matches += 1
        else:
            mismatches.append((row, col, verdict))
    return {
        "scored": scored,
        "matches": matches,
        "excluded": excluded,
        "agreement": matches / scored if scored else None,
        "mismatches": mismatches,
    }


def build_report(runs: list[RunData], audit_verdicts_path: str | None = None) -> str:
    reference = find_reference(runs)
    lines = ["# Cross-detector comparison", ""]
    lines.append(
        "**ESA's `MSK_CLASSI` mask is a baseline, not ground truth** (SPEC.md 2.1, 3.2). Every "
        "\"agreement with ESA\" figure below measures similarity to that baseline, not accuracy "
        "against a measurement of the truth; SPEC.md 3.2 explains why maximising it is not a valid "
        "way to choose a detector's parameters, and the same caution applies to reading these numbers."
    )
    lines.append("")

    if reference is not None:
        others = [r for r in runs if r is not reference]
        with_masks = [r for r in others if r.has_tile_masks() and reference.has_tile_masks()]
        if with_masks:
            lines.append("## Pixel level (precision / recall / F1 / IoU against ESA)")
            lines.append("")
            lines.append("| Detector | Precision | Recall | F1 | IoU |")
            lines.append("|---|---|---|---|---|")
            for run in with_masks:
                result = pixel_level(reference, run)
                lines.append(
                    "| %s | %.2f | %.2f | %.2f | %.2f |"
                    % (run.detector, result.precision, result.recall, result.f1, result.iou)
                )
            lines.append("")
        skipped = [r.name for r in others if r not in with_masks]
        if skipped:
            lines.append(
                "*(Pixel-level metrics skipped for: %s -- no `tile_masks/`. Rerun with "
                "`--save-tile-masks` to include them.)*"
                % ", ".join(skipped)
            )
            lines.append("")

        lines.append("## Tile level")
        lines.append("")
        lines.append(
            "| Detector | Invalid tiles | Disagreements with ESA | ESA keeps, detector discards | "
            "ESA discards, detector keeps |"
        )
        lines.append("|---|---|---|---|---|")
        lines.append("| esa (reference) | %d | -- | -- | -- |" % reference.invalid_count())
        for run in others:
            stats = tile_level(reference, run)
            lines.append(
                "| %s | %d | %d | %d | %d |"
                % (
                    run.detector,
                    stats["invalid_tiles"],
                    stats["disagreements"],
                    stats["discards_where_esa_keeps"],
                    stats["keeps_where_esa_discards"],
                )
            )
        lines.append("")

        if audit_verdicts_path:
            verdicts = load_audit_verdicts(audit_verdicts_path)
            thin_cloud_count = sum(1 for v in verdicts if v["verdict"] == "thin_cloud")
            lines.append("## Visual audit (E2)")
            lines.append("")
            lines.append(
                "A small, subjective sample of %d tiles (the four named tiles plus the 12 where "
                "`threshold` and `s2cloudless` disagree most in `cloud_cover_percent`), verdicted by "
                "eye against the true-colour imagery, **not** against any detector's output "
                "(`%s`). %d of the %d are `thin_cloud` and excluded from scoring below: SPEC.md "
                "decision D7 (how to treat thin cloud) is open, so there is no defined right answer "
                "to score against for those. This is one person's judgement on one tile in "
                "twenty-five of the scene -- read the agreement gap as suggestive, not decisive."
                % (len(verdicts), audit_verdicts_path, thin_cloud_count, len(verdicts))
            )
            lines.append("")
            lines.append("| Detector | Agreement with ESA (all 400 tiles) | Agreement with audit (%d scored) |"
                          % (len(verdicts) - thin_cloud_count))
            lines.append("|---|---|---|")
            esa_self_agreement = 1.0
            esa_audit = audit_agreement(reference, verdicts)
            lines.append(
                "| esa (reference) | %.1f %% | %.1f %% |"
                % (100 * esa_self_agreement, 100 * esa_audit["agreement"] if esa_audit["agreement"] is not None else float("nan"))
            )
            for run in others:
                esa_agree_pct = 100 * (1 - tile_level(reference, run)["disagreements"] / len(reference.rows))
                run_audit = audit_agreement(run, verdicts)
                lines.append(
                    "| %s | %.1f %% | %.1f %% |"
                    % (run.detector, esa_agree_pct, 100 * run_audit["agreement"] if run_audit["agreement"] is not None else float("nan"))
                )
            lines.append("")
    else:
        lines.append(
            "*(No `esa` run was given, so no ESA-relative pixel- or tile-level comparison is "
            "reported -- only per-run figures below.)*"
        )
        lines.append("")
        lines.append("## Per-run invalid-tile counts")
        lines.append("")
        lines.append("| Detector | Invalid tiles | Scene cloud % |")
        lines.append("|---|---|---|")
        for run in runs:
            lines.append("| %s | %d | %s |" % (run.detector, run.invalid_count(), _fmt(run.scene_cloud_percent())))
        lines.append("")

    lines.append("## Named tiles")
    lines.append("")
    header = "| Tile | Character | " + " | ".join(run.detector for run in runs) + " |"
    lines.append(header)
    lines.append("|---|---|" + "---|" * len(runs))
    for entry in named_tile_table(runs):
        cells = " | ".join("%.4f %%" % entry[run.name] for run in runs)
        lines.append("| %d,%d | %s | %s |" % (entry["row"], entry["col"], entry["character"], cells))
    lines.append("")

    lines.append("## The spread")
    lines.append("")
    lines.append(
        "The headline result: how many of the 400 tiles each method discards, and how many of the "
        "25-35 % band (61 under ESA) sit right at the cut."
    )
    lines.append("")
    lines.append("| Method | Invalid tiles | Scene cloud % |")
    lines.append("|---|---|---|")
    for run in runs:
        lines.append("| %s | %d | %s |" % (run.detector, run.invalid_count(), _fmt(run.scene_cloud_percent())))
    lines.append(
        "| %s | %d | %.1f |"
        % (NAIVE_RULE_REFERENCE["label"], NAIVE_RULE_REFERENCE["invalid_tiles"], NAIVE_RULE_REFERENCE["scene_cloud_percent"])
    )
    lines.append("")

    return "\n".join(lines) + "\n"


def _fmt(value) -> str:
    return "%.4f" % value if isinstance(value, (int, float)) else "n/a"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare pipeline output folders (SPEC.md 5.1, E1).")
    parser.add_argument("--runs", nargs="+", required=True, help="two or more pipeline output folders")
    parser.add_argument("--out", default="output/comparison/comparison.md")
    parser.add_argument(
        "--audit-verdicts", default=None,
        help="audit/verdicts.csv (E2); adds the audit-agreement section if given and an esa run is present",
    )
    args = parser.parse_args(argv)

    if len(args.runs) < 2:
        parser.error("give at least two --runs folders to compare")

    try:
        runs = load_runs(args.runs)
    except ComparisonError as error:
        print("error: %s" % error, file=sys.stderr)
        return 2

    try:
        report = build_report(runs, args.audit_verdicts)
    except ComparisonError as error:
        print("error: %s" % error, file=sys.stderr)
        return 2
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf8") as handle:
        handle.write(report)
    print("%s (%d runs: %s)" % (args.out, len(runs), ", ".join(r.detector for r in runs)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
