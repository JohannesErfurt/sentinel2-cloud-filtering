"""F8 and F9 -- the CLI surface, and the checks catching deliberately broken output."""
from __future__ import annotations

import json
import os

import pytest

from pipeline import detectors
from pipeline.checks import run_checks
from pipeline.report import read_report_csv


def _ids(results):
    return {r.id: r for r in results}


# ------------------------------------------------------------------------- F9
def test_a_correct_folder_passes_the_structural_checks(synthetic_run):
    results = run_checks(str(synthetic_run), None, ["CK1", "CK2", "CK3", "CK4", "CK5", "CK6"])
    failures = [r for r in results if not r.ok]
    assert not failures, "\n".join(str(r) for r in failures)


def test_ck2_catches_a_399_row_csv(synthetic_run):
    path = synthetic_run / "report.csv"
    lines = path.read_text(encoding="utf8").splitlines()
    path.write_text("\n".join(lines[:-1]) + "\n", encoding="utf8")
    assert _ids(run_checks(str(synthetic_run), None, ["CK2"]))["CK2"].ok is False


def test_ck2_catches_a_renamed_column(synthetic_run):
    path = synthetic_run / "report.csv"
    text = path.read_text(encoding="utf8")
    path.write_text(text.replace("cloud_cover_percent", "cloud_pct", 1), encoding="utf8")
    assert _ids(run_checks(str(synthetic_run), None, ["CK2"]))["CK2"].ok is False


def test_ck3_catches_a_cloudy_tile_marked_valid(synthetic_run):
    path = synthetic_run / "report.csv"
    lines = path.read_text(encoding="utf8").splitlines()
    assert lines[2].endswith(",False"), "tile 0,1 is cloudy in the synthetic run"
    lines[2] = lines[2][: -len("False")] + "True"
    path.write_text("\n".join(lines) + "\n", encoding="utf8")
    assert _ids(run_checks(str(synthetic_run), None, ["CK3"]))["CK3"].ok is False


def test_ck5_catches_a_clear_tile_marked_invalid_but_still_written(synthetic_run):
    """CK3 must allow a clear tile to be False (no-data); CK5 then sees its stray JPEG."""
    path = synthetic_run / "report.csv"
    lines = path.read_text(encoding="utf8").splitlines()
    assert lines[1].endswith(",True"), "tile 0,0 is clear in the synthetic run"
    lines[1] = lines[1][: -len("True")] + "False"
    path.write_text("\n".join(lines) + "\n", encoding="utf8")
    results = _ids(run_checks(str(synthetic_run), None, ["CK3", "CK5"]))
    assert results["CK3"].ok is True
    assert results["CK5"].ok is False


def test_ck3_catches_a_truncated_percentage(synthetic_run):
    """'at least 4 decimals' exists so a rounded value cannot be re-derived."""
    path = synthetic_run / "report.csv"
    header, rows = read_report_csv(str(path))
    lines = [",".join(header)]
    for row in rows:
        row["cloud_cover_percent"] = "%.1f" % float(row["cloud_cover_percent"])
        lines.append(",".join(row[name] for name in header))
    path.write_text("\n".join(lines) + "\n", encoding="utf8")
    assert _ids(run_checks(str(synthetic_run), None, ["CK3"]))["CK3"].ok is False


def test_ck5_catches_a_deleted_jpeg(synthetic_run):
    jpegs = sorted((synthetic_run / "tiles").glob("*.jpg"))
    os.remove(jpegs[0])
    assert _ids(run_checks(str(synthetic_run), None, ["CK5"]))["CK5"].ok is False


def test_ck5_rejects_anything_but_jpegs_in_tiles(synthetic_run):
    (synthetic_run / "tiles" / "tile_r00_c00.jgw").write_text("10\n0\n0\n-10\n0\n0\n", encoding="utf8")
    assert _ids(run_checks(str(synthetic_run), None, ["CK5"]))["CK5"].ok is False


def test_ck6_catches_a_summary_that_disagrees_with_the_csv(synthetic_run):
    path = synthetic_run / "run_summary.json"
    summary = json.loads(path.read_text(encoding="utf8"))
    summary["results"]["scene_cloud_percent"] += 5.0
    path.write_text(json.dumps(summary), encoding="utf8")
    assert _ids(run_checks(str(synthetic_run), None, ["CK6"]))["CK6"].ok is False


def test_ck4_catches_a_shifted_bounding_box(synthetic_run):
    path = synthetic_run / "report.csv"
    lines = path.read_text(encoding="utf8").splitlines()
    fields = lines[1].split(",")
    fields[0] = "%.8f" % (float(fields[0]) + 0.01)
    lines[1] = ",".join(fields)
    path.write_text("\n".join(lines) + "\n", encoding="utf8")
    assert _ids(run_checks(str(synthetic_run), None, ["CK4"]))["CK4"].ok is False


def test_ck7_is_skipped_when_there_is_no_geojson(synthetic_run):
    result = _ids(run_checks(str(synthetic_run), None, ["CK7"]))["CK7"]
    assert result.ok and result.skipped


def test_ck8_compares_two_folders(synthetic_run, tmp_path):
    import shutil

    from pipeline.checks import check_ck8

    copy = tmp_path / "copy"
    shutil.copytree(synthetic_run, copy)
    assert check_ck8(str(synthetic_run), str(copy)).ok

    (copy / "report.csv").write_text("changed\n", encoding="utf8")
    assert check_ck8(str(synthetic_run), str(copy)).ok is False


# ------------------------------------------------------------------------- F8
def test_help_lists_every_option():
    from pipeline.run import build_parser

    text = build_parser().format_help()
    for flag in [
        "--safe-dir", "--detector", "--out", "--detection-resolution", "--jpeg-quality",
        "--save-tile-masks", "--no-geojson", "--geojson-simplify",
        "--t-bright", "--t-ndsi", "--t-cirrus",
        "--prob-threshold", "--average-over", "--dilation-size",
    ]:
        assert flag in text, "%s is missing from --help" % flag


def test_a_bad_safe_dir_fails_with_a_clear_message(tmp_path, capsys):
    from pipeline.run import main

    code = main(["--safe-dir", str(tmp_path), "--detector", "esa", "--out", str(tmp_path / "o")])
    assert code == 2
    assert "MTD_MSIL1C.xml is missing" in capsys.readouterr().err


def test_an_unknown_detector_names_the_backend_tasks():
    with pytest.raises(detectors.DetectorError, match="B1"):
        detectors.build("no-such-detector")


def test_the_registry_reports_what_is_available():
    """The three backends are tasks B1-B3; the registry must not pretend."""
    names = detectors.available()
    assert set(names) <= {"esa", "threshold", "s2cloudless"}


def test_detector_base_class_requires_a_tile_mask():
    class Incomplete(detectors.Detector):
        name = "incomplete"

    with pytest.raises(TypeError):
        Incomplete()
