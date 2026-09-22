"""E3 -- applying SPEC.md 5.3's rule mechanically to a comparison's numbers."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from pipeline.compare import ComparisonError, load_runs  # noqa: E402
from tests.conftest import make_stats  # noqa: E402
from tests.test_compare import _write_run  # noqa: E402

import decide as decide_module  # noqa: E402


def _run_with_valid_pattern(tmp_path, name, meta, detector, invalid_rows_cols):
    """A run where exactly the given (row, col) tiles are invalid."""
    from pipeline.constants import TILE_PIXELS

    invalid = set(invalid_rows_cols)
    return _write_run(
        tmp_path, name, meta, detector,
        lambda r, c: TILE_PIXELS if (r, c) in invalid else 0,
        with_masks=False,
    )


def _verdicts_for(tiles_and_verdicts):
    return [{"row": str(r), "col": str(c), "verdict": v} for r, c, v in tiles_and_verdicts]


def test_threshold_wins_outright_when_clearly_better(tmp_path, meta):
    # 8 verdicts: threshold matches all 8, s2cloudless matches only 6 -- a
    # 2-tile gap, outside the 1-tile tie margin, so step 2 must decide it.
    cloud_tiles = [(0, 0), (0, 1), (0, 2), (0, 3)]
    clear_tiles = [(1, 0), (1, 1), (1, 2), (1, 3)]
    esa = _run_with_valid_pattern(tmp_path, "esa", meta, "esa", [])
    threshold = _run_with_valid_pattern(tmp_path, "threshold", meta, "threshold", cloud_tiles)
    s2c = _run_with_valid_pattern(tmp_path, "s2cloudless", meta, "s2cloudless", cloud_tiles[:2])
    runs = load_runs([esa, threshold, s2c])
    verdicts = _verdicts_for([(r, c, "cloud") for r, c in cloud_tiles] + [(r, c, "clear") for r, c in clear_tiles])

    decision = decide_module.decide(runs, verdicts)
    assert decision["winner"] == "threshold"
    assert "step 2" in decision["rule"]


def test_s2cloudless_wins_outright_when_clearly_better(tmp_path, meta):
    cloud_tiles = [(0, 0), (0, 1), (0, 2), (0, 3)]
    clear_tiles = [(1, 0), (1, 1), (1, 2), (1, 3)]
    esa = _run_with_valid_pattern(tmp_path, "esa", meta, "esa", [])
    threshold = _run_with_valid_pattern(tmp_path, "threshold", meta, "threshold", cloud_tiles[:2])
    s2c = _run_with_valid_pattern(tmp_path, "s2cloudless", meta, "s2cloudless", cloud_tiles)
    runs = load_runs([esa, threshold, s2c])
    verdicts = _verdicts_for([(r, c, "cloud") for r, c in cloud_tiles] + [(r, c, "clear") for r, c in clear_tiles])

    decision = decide_module.decide(runs, verdicts)
    assert decision["winner"] == "s2cloudless"
    assert "step 2" in decision["rule"]


def test_threshold_wins_the_tie_break_within_margin(tmp_path, meta):
    """Both candidates score identically -> step 3 fires, not step 2."""
    esa = _run_with_valid_pattern(tmp_path, "esa", meta, "esa", [])
    threshold = _run_with_valid_pattern(tmp_path, "threshold", meta, "threshold", [(0, 0)])
    s2c = _run_with_valid_pattern(tmp_path, "s2cloudless", meta, "s2cloudless", [(0, 0)])
    runs = load_runs([esa, threshold, s2c])
    verdicts = _verdicts_for([(0, 0, "cloud"), (1, 1, "clear")])

    decision = decide_module.decide(runs, verdicts)
    assert decision["winner"] == "threshold"
    assert "step 3" in decision["rule"]


def test_missing_esa_run_raises(tmp_path, meta):
    threshold = _run_with_valid_pattern(tmp_path, "threshold", meta, "threshold", [])
    s2c = _run_with_valid_pattern(tmp_path, "s2cloudless", meta, "s2cloudless", [])
    runs = load_runs([threshold, s2c])
    with pytest.raises(ComparisonError, match="esa"):
        decide_module.decide(runs, [])


def test_missing_candidate_raises(tmp_path, meta):
    esa = _run_with_valid_pattern(tmp_path, "esa", meta, "esa", [])
    threshold = _run_with_valid_pattern(tmp_path, "threshold", meta, "threshold", [])
    runs = load_runs([esa, threshold])
    with pytest.raises(ComparisonError, match="s2cloudless"):
        decide_module.decide(runs, [])


def test_report_names_the_winner_and_the_rule(tmp_path, meta):
    esa = _run_with_valid_pattern(tmp_path, "esa", meta, "esa", [])
    threshold = _run_with_valid_pattern(tmp_path, "threshold", meta, "threshold", [(0, 0)])
    s2c = _run_with_valid_pattern(tmp_path, "s2cloudless", meta, "s2cloudless", [])
    runs = load_runs([esa, threshold, s2c])
    verdicts = _verdicts_for([(0, 0, "cloud"), (1, 1, "clear"), (2, 2, "clear"), (3, 3, "clear")])
    decision = decide_module.decide(runs, verdicts)
    report = decide_module.build_report(decision)
    assert "`threshold` ships" in report
    assert "fixed in SPEC.md 0.4, before this evaluation ran" in report


def test_report_states_ml_bonus_is_satisfied_when_s2cloudless_does_not_ship(tmp_path, meta):
    esa = _run_with_valid_pattern(tmp_path, "esa", meta, "esa", [])
    threshold = _run_with_valid_pattern(tmp_path, "threshold", meta, "threshold", [(0, 0)])
    s2c = _run_with_valid_pattern(tmp_path, "s2cloudless", meta, "s2cloudless", [])
    runs = load_runs([esa, threshold, s2c])
    verdicts = _verdicts_for([(0, 0, "cloud"), (1, 1, "clear"), (2, 2, "clear"), (3, 3, "clear")])
    decision = decide_module.decide(runs, verdicts)
    report = decide_module.build_report(decision)
    assert "ML bonus" in report and "satisfied" in report


# ------------------------------------------------------------ against the real product
@pytest.mark.needs_product
def test_the_real_decision_matches_the_recorded_verdict():
    """Cross-checks the actual decision this project recorded
    (output/comparison/decision.md) reproduces from the real runs and the
    real audit/verdicts.csv."""
    import os

    from pipeline.compare import load_audit_verdicts

    paths = ["output/esa_new", "output/threshold", "output/s2cloudless"]
    if not all(os.path.isdir(p) for p in paths) or not os.path.isfile("audit/verdicts.csv"):
        pytest.skip("needs this project's own real output folders and audit/verdicts.csv")

    runs = load_runs(paths)
    verdicts = load_audit_verdicts("audit/verdicts.csv")
    decision = decide_module.decide(runs, verdicts)

    assert decision["winner"] == "threshold"
    assert decision["audit_result"]["threshold"]["matches"] == 13
    assert decision["audit_result"]["s2cloudless"]["matches"] == 12
    assert decision["scored"] == 15
