"""E1 -- the cross-detector comparison module.

precision/recall/F1/IoU on tiny hand-made masks are already covered
thoroughly by tests/test_metrics.py, which pipeline.compare reuses rather
than reimplementing; these tests cover this module's own wrapping logic:
loading run folders, tile-level direction counting, and working for any
subset of runs in any order.
"""
from __future__ import annotations

import json

import cv2
import numpy as np
import pytest

from pipeline.compare import (
    ComparisonError,
    NAMED_TILES,
    audit_agreement,
    build_report,
    find_reference,
    load_audit_verdicts,
    load_runs,
    named_tile_table,
    pixel_level,
    select_audit_sample,
    tile_level,
)
from pipeline.constants import GRID, TILE_PIXELS, TILE_PX
from tests.conftest import make_stats


def _write_run(tmp_path, name, meta, detector, cloud_for, with_masks=True):
    from pipeline.report import write_report_csv, write_tile_stats_csv
    from pipeline.tiling import scene_cloud_percent

    stats = make_stats(meta, cloud_for)
    out_dir = tmp_path / name
    out_dir.mkdir()
    write_report_csv(str(out_dir / "report.csv"), stats)
    write_tile_stats_csv(str(out_dir / "tile_stats.csv"), stats)

    valid = sum(1 for s in stats if s.valid)
    summary = {
        "detector": detector,
        "parameters": {},
        "results": {"scene_cloud_percent": scene_cloud_percent(stats), "valid_tiles": valid,
                    "invalid_tiles": GRID * GRID - valid},
    }
    (out_dir / "run_summary.json").write_text(json.dumps(summary), encoding="utf8")

    if with_masks:
        masks_dir = out_dir / "tile_masks"
        masks_dir.mkdir()
        for item in stats:
            tile = np.zeros((TILE_PX, TILE_PX), dtype=np.uint8)
            tile[: item.cloud_pixels // TILE_PX] = 255  # arbitrary shape with the right pixel count
            cv2.imwrite(str(masks_dir / (item.name + ".png")), tile)
    return str(out_dir)


def test_load_runs_reads_detector_name_from_run_summary(tmp_path, meta):
    path = _write_run(tmp_path, "my_esa_run", meta, "esa", lambda r, c: 0, with_masks=False)
    runs = load_runs([path])
    assert runs[0].detector == "esa"
    assert runs[0].name == "my_esa_run"  # folder name need not match detector name


def test_missing_run_summary_raises_clearly(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ComparisonError, match="run_summary.json"):
        load_runs([str(empty)])


def test_find_reference_locates_esa_regardless_of_folder_name(tmp_path, meta):
    a = _write_run(tmp_path, "a", meta, "threshold", lambda r, c: 0, with_masks=False)
    b = _write_run(tmp_path, "esa_new", meta, "esa", lambda r, c: 0, with_masks=False)
    runs = load_runs([a, b])
    reference = find_reference(runs)
    assert reference is not None and reference.detector == "esa"


def test_find_reference_is_none_without_an_esa_run(tmp_path, meta):
    a = _write_run(tmp_path, "a", meta, "threshold", lambda r, c: 0, with_masks=False)
    b = _write_run(tmp_path, "b", meta, "s2cloudless", lambda r, c: 0, with_masks=False)
    runs = load_runs([a, b])
    assert find_reference(runs) is None


def test_tile_level_counts_direction_separately(tmp_path, meta):
    # esa: tile 0 invalid, tile 1 valid, everything else valid.
    esa_path = _write_run(tmp_path, "esa", meta, "esa",
                           lambda r, c: TILE_PIXELS if (r, c) == (0, 0) else 0, with_masks=False)
    # other: tile 0 valid (disagrees: esa keeps it invalid -> "esa discards, other keeps"
    #        reversed here: esa invalid, other valid), tile 1 invalid (esa valid, other invalid).
    other_path = _write_run(tmp_path, "other", meta, "threshold",
                             lambda r, c: TILE_PIXELS if (r, c) == (0, 1) else 0, with_masks=False)
    esa_run, other_run = load_runs([esa_path, other_path])
    stats = tile_level(esa_run, other_run)
    assert stats["disagreements"] == 2
    assert stats["discards_where_esa_keeps"] == 1  # (0,1): esa keeps (valid), other discards
    assert stats["keeps_where_esa_discards"] == 1  # (0,0): esa discards, other keeps


def test_named_tile_table_has_all_four_tiles_and_every_run(tmp_path, meta):
    a = _write_run(tmp_path, "a", meta, "esa", lambda r, c: 0, with_masks=False)
    b = _write_run(tmp_path, "b", meta, "threshold", lambda r, c: 0, with_masks=False)
    runs = load_runs([a, b])
    table = named_tile_table(runs)
    assert len(table) == len(NAMED_TILES)
    assert {(e["row"], e["col"]) for e in table} == {(r, c) for r, c, _ in NAMED_TILES}
    assert all(run.name in entry for entry in table for run in runs)


def test_pixel_level_matches_metrics_agreement_directly(tmp_path, meta):
    """The wrapper (reading tile_masks/, reassembling, calling agreement)
    must give the same numbers as calling pipeline.metrics.agreement on the
    reassembled arrays by hand."""
    from pipeline.metrics import agreement as raw_agreement

    esa_path = _write_run(tmp_path, "esa", meta, "esa", lambda r, c: (r * c) % 5 * 1000)
    other_path = _write_run(tmp_path, "other", meta, "threshold", lambda r, c: (r + c) % 4 * 1500)
    esa_run, other_run = load_runs([esa_path, other_path])

    result = pixel_level(esa_run, other_run)
    expected = raw_agreement(other_run.full_mask(), esa_run.full_mask())
    assert result.as_dict() == expected.as_dict()


def test_missing_tile_masks_raises_a_clear_error(tmp_path, meta):
    path = _write_run(tmp_path, "no_masks", meta, "esa", lambda r, c: 0, with_masks=False)
    (run,) = load_runs([path])
    with pytest.raises(ComparisonError, match="tile_masks"):
        run.full_mask()


def test_build_report_works_for_two_runs_without_esa(tmp_path, meta):
    """SPEC.md E1: 'works for any subset of runs, in any order' -- including
    a subset that does not contain esa at all."""
    a = _write_run(tmp_path, "a", meta, "threshold", lambda r, c: 0, with_masks=False)
    b = _write_run(tmp_path, "b", meta, "s2cloudless", lambda r, c: 0, with_masks=False)
    report = build_report(load_runs([a, b]))
    assert "No `esa` run was given" in report
    assert "threshold" in report and "s2cloudless" in report


def test_build_report_order_does_not_matter(tmp_path, meta):
    a = _write_run(tmp_path, "a", meta, "esa", lambda r, c: 0, with_masks=False)
    b = _write_run(tmp_path, "b", meta, "threshold", lambda r, c: 0, with_masks=False)
    forward = build_report(load_runs([a, b]))
    backward = build_report(load_runs([b, a]))
    assert "esa" in forward and "threshold" in forward
    assert "esa" in backward and "threshold" in backward


def test_report_states_esa_is_a_baseline_not_ground_truth(tmp_path, meta):
    a = _write_run(tmp_path, "a", meta, "esa", lambda r, c: 0, with_masks=False)
    b = _write_run(tmp_path, "b", meta, "threshold", lambda r, c: 0, with_masks=False)
    report = build_report(load_runs([a, b]))
    assert "baseline" in report.lower() and "not ground truth" in report.lower()


def test_report_skips_pixel_metrics_when_tile_masks_are_missing(tmp_path, meta):
    a = _write_run(tmp_path, "a", meta, "esa", lambda r, c: 0, with_masks=False)
    b = _write_run(tmp_path, "b", meta, "threshold", lambda r, c: 0, with_masks=False)
    report = build_report(load_runs([a, b]))
    assert "Pixel level" not in report or "skipped" in report


# ------------------------------------------------------------ against the real product
@pytest.mark.needs_product
def test_the_actual_three_runs_reproduce_the_recorded_findings():
    """Cross-checks the three real output folders this project already has,
    against the figures recorded in SPEC.md while implementing E1."""
    import os

    paths = [os.path.join("output", name) for name in ("esa_new", "threshold", "s2cloudless")]
    if not all(os.path.isdir(p) for p in paths):
        pytest.skip("needs the three real output folders from this project's own session")
    runs = load_runs(paths)
    reference = find_reference(runs)
    assert reference is not None and reference.invalid_count() == 107

    by_detector = {r.detector: r for r in runs}
    threshold_stats = tile_level(reference, by_detector["threshold"])
    assert threshold_stats["invalid_tiles"] == 83
    assert threshold_stats["disagreements"] == 30

    s2c_stats = tile_level(reference, by_detector["s2cloudless"])
    assert s2c_stats["invalid_tiles"] == 161
    assert s2c_stats["disagreements"] == 56


# --------------------------------------------------------------------- E2
def test_select_audit_sample_has_16_tiles_no_duplicates(tmp_path, meta):
    a = _write_run(tmp_path, "a", meta, "threshold", lambda r, c: (r * 7 + c * 3) % 100 * 3000, with_masks=False)
    b = _write_run(tmp_path, "b", meta, "s2cloudless", lambda r, c: (r * 3 + c * 11) % 100 * 3000, with_masks=False)
    a_run, b_run = load_runs([a, b])
    sample = select_audit_sample(a_run, b_run)
    assert len(sample) == 16
    coords = [(e["row"], e["col"]) for e in sample]
    assert len(set(coords)) == 16  # no duplicates, even if a named tile also had a large diff


def test_select_audit_sample_always_includes_the_four_named_tiles(tmp_path, meta):
    a = _write_run(tmp_path, "a", meta, "threshold", lambda r, c: 0, with_masks=False)
    b = _write_run(tmp_path, "b", meta, "s2cloudless", lambda r, c: 0, with_masks=False)
    a_run, b_run = load_runs([a, b])
    sample = select_audit_sample(a_run, b_run)
    coords = {(e["row"], e["col"]) for e in sample}
    for row, col, _ in NAMED_TILES:
        assert (row, col) in coords


def test_select_audit_sample_picks_the_biggest_disagreements(tmp_path, meta):
    """The 12 non-named tiles must be exactly the 12 largest |diff| values,
    not merely large ones -- this is the actual selection rule, not a
    reasonable-looking approximation of it."""
    def cloud_for(row, col):
        # A near-injective mapping over the 400 tiles (row*20+col in
        # [0, 400)), scaled so every tile gets a distinct pixel count and
        # ties at the top-12 boundary are not a matter of dict/sort order.
        return min((row * GRID + col) * 700, TILE_PIXELS)

    def other_cloud_for(row, col):
        return min((399 - (row * GRID + col)) * 500, TILE_PIXELS)

    a = _write_run(tmp_path, "a", meta, "threshold", cloud_for, with_masks=False)
    b = _write_run(tmp_path, "b", meta, "s2cloudless", other_cloud_for, with_masks=False)
    a_run, b_run = load_runs([a, b])
    sample = select_audit_sample(a_run, b_run)

    named = {(r, c) for r, c, _ in NAMED_TILES}
    from pipeline.constants import GRID as _GRID

    all_diffs = sorted(
        (
            abs(a_run.cloud_percent(r * _GRID + c) - b_run.cloud_percent(r * _GRID + c)),
            r, c,
        )
        for r in range(_GRID) for c in range(_GRID) if (r, c) not in named
    )
    expected_top12 = {(r, c) for _, r, c in all_diffs[-12:]}
    got = {(e["row"], e["col"]) for e in sample if (e["row"], e["col"]) not in named}
    assert got == expected_top12


def test_load_audit_verdicts_requires_16_rows(tmp_path):
    path = tmp_path / "short.csv"
    path.write_text("row,col,character,reason,verdict,notes\n0,0,,,clear,\n", encoding="utf8")
    with pytest.raises(ComparisonError, match="16"):
        load_audit_verdicts(str(path))


def test_load_audit_verdicts_rejects_an_unknown_verdict(tmp_path):
    rows = ["row,col,character,reason,verdict,notes"]
    for i in range(16):
        rows.append("%d,%d,,,%s," % (i, i, "cloud" if i else "maybe"))
    path = tmp_path / "bad.csv"
    path.write_text("\n".join(rows) + "\n", encoding="utf8")
    with pytest.raises(ComparisonError, match="verdict"):
        load_audit_verdicts(str(path))


def test_audit_agreement_scores_only_cloud_and_clear_verdicts(tmp_path, meta):
    # A run where tile (0,0) is invalid (cloud-like) and everything else valid.
    run_path = _write_run(tmp_path, "run", meta, "threshold",
                           lambda r, c: TILE_PIXELS if (r, c) == (0, 0) else 0, with_masks=False)
    (run,) = load_runs([run_path])

    verdicts = [{"row": "0", "col": "0", "verdict": "cloud"}]  # correct: run says invalid
    verdicts += [{"row": "0", "col": str(i), "verdict": "clear"} for i in range(1, 15)]  # correct: valid
    verdicts += [{"row": "5", "col": "5", "verdict": "thin_cloud"}]  # excluded

    result = audit_agreement(run, verdicts)
    assert result["scored"] == 15
    assert result["excluded"] == 1
    assert result["matches"] == 15
    assert result["agreement"] == pytest.approx(1.0)


def test_audit_agreement_reports_mismatches(tmp_path, meta):
    run_path = _write_run(tmp_path, "run", meta, "threshold", lambda r, c: 0, with_masks=False)  # everything valid
    (run,) = load_runs([run_path])
    verdicts = [{"row": "3", "col": "3", "verdict": "cloud"}]  # run says valid, verdict expects invalid
    result = audit_agreement(run, verdicts)
    assert result["matches"] == 0
    assert result["mismatches"] == [(3, 3, "cloud")]


@pytest.mark.needs_product
def test_the_real_audit_sample_and_verdicts_are_internally_consistent():
    """audit/verdicts.csv, as this project actually recorded it: exactly 16
    rows, valid verdicts, and its selection matches what select_audit_sample
    would compute fresh from the same two runs (i.e. nobody hand-edited the
    tile list after the fact)."""
    import os

    if not (os.path.isdir("output/threshold") and os.path.isdir("output/s2cloudless")):
        pytest.skip("needs this project's own output/threshold and output/s2cloudless")
    if not os.path.isfile("audit/verdicts.csv"):
        pytest.skip("needs this project's own audit/verdicts.csv")

    threshold_run, s2c_run = load_runs(["output/threshold", "output/s2cloudless"])
    expected = {(e["row"], e["col"]) for e in select_audit_sample(threshold_run, s2c_run)}

    verdicts = load_audit_verdicts("audit/verdicts.csv")
    actual = {(int(v["row"]), int(v["col"])) for v in verdicts}
    assert actual == expected
