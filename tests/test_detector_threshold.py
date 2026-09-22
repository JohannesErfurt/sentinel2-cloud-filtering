"""B2 -- the threshold backend: config loading, the resolution decision (D2),
window-independence, and the reference figures against the real product."""
from __future__ import annotations

import json

import numpy as np
import pytest

from pipeline import detectors
from pipeline.constants import TILE_PX
from pipeline.detectors.threshold import (
    FALLBACK_THRESHOLDS,
    ThresholdConfigError,
    ThresholdDetector,
    load_config,
)
from pipeline.tiling import compute_tile_stats, iter_tiles, tile_window


def test_registered_under_threshold():
    assert "threshold" in detectors.available()
    assert isinstance(detectors.build("threshold"), ThresholdDetector)


def test_rejects_an_unknown_detection_resolution():
    with pytest.raises(ValueError, match="10 or 60"):
        ThresholdDetector(detection_resolution=45)


# ------------------------------------------------------------------ config
def test_missing_config_falls_back_to_the_spec_starting_values(tmp_path):
    values = load_config(str(tmp_path / "absent.json"))
    assert values == FALLBACK_THRESHOLDS


def test_a_valid_config_is_read(tmp_path):
    path = tmp_path / "thresholds.json"
    path.write_text(json.dumps({"t_bright": 0.19, "t_ndsi": -0.21, "t_cirrus": 0.004}), encoding="utf8")
    values = load_config(str(path))
    assert values == {"t_bright": 0.19, "t_ndsi": -0.21, "t_cirrus": 0.004}


def test_malformed_json_raises_rather_than_silently_falling_back(tmp_path):
    path = tmp_path / "thresholds.json"
    path.write_text("{not valid json", encoding="utf8")
    with pytest.raises(ThresholdConfigError, match="not valid JSON"):
        load_config(str(path))


def test_a_config_missing_a_key_raises(tmp_path):
    path = tmp_path / "thresholds.json"
    path.write_text(json.dumps({"t_bright": 0.18, "t_ndsi": -0.2}), encoding="utf8")
    with pytest.raises(ThresholdConfigError, match="t_cirrus"):
        load_config(str(path))


def test_explicit_kwargs_override_the_config(tmp_path):
    path = tmp_path / "thresholds.json"
    path.write_text(json.dumps({"t_bright": 0.19, "t_ndsi": -0.21, "t_cirrus": 0.004}), encoding="utf8")
    detector = ThresholdDetector(t_bright=0.25, config_path=str(path))
    assert detector.t_bright == 0.25  # overridden
    assert detector.t_ndsi == -0.21  # from the config
    assert detector.t_cirrus == 0.004  # from the config


def test_parameters_reports_everything_that_changes_the_output():
    detector = ThresholdDetector(t_bright=0.2, t_ndsi=-0.15, t_cirrus=0.006, detection_resolution=60)
    assert detector.parameters() == {
        "t_bright": 0.2, "t_ndsi": -0.15, "t_cirrus": 0.006, "detection_resolution": 60,
    }


# --------------------------------------------------------------- close/cut
def test_close_releases_the_cached_60m_mask():
    detector = ThresholdDetector(detection_resolution=60)
    assert detector._cloud10 is None
    detector.close()
    assert detector._cloud10 is None


# ------------------------------------------------------------ against the real product
@pytest.mark.needs_product
@pytest.mark.parametrize("resolution", [10, 60])
def test_tile_mask_is_bool_and_549_square(product, resolution):
    detector = ThresholdDetector(detection_resolution=resolution)
    mask = detector.tile_mask(product, 6, 5)
    assert mask.dtype == bool
    assert mask.shape == (TILE_PX, TILE_PX)


@pytest.mark.needs_product
def test_10m_and_60m_resolution_agree_closely_on_tile_validity(product):
    """SPEC.md 3.3: 0 of 400 tiles were measured to change validity between
    the two resolutions for brightness alone; the full combined rule is
    checked here on the four named tiles as a lighter version of that claim.
    """
    fine = ThresholdDetector(detection_resolution=10)
    coarse = ThresholdDetector(detection_resolution=60)
    for row, col in [(11, 2), (6, 5), (5, 16), (12, 19)]:
        fine_stats = compute_tile_stats(product, row, col, fine.tile_mask(product, row, col))
        coarse_stats = compute_tile_stats(product, row, col, coarse.tile_mask(product, row, col))
        assert abs(fine_stats.cloud_cover_percent - coarse_stats.cloud_cover_percent) < 5.0
    fine.close()
    coarse.close()


@pytest.mark.needs_product
def test_streaming_tiles_equal_cutting_from_an_independently_computed_scene_mask(product):
    """SPEC.md B2.3: five random tiles, window-by-window, must equal cutting
    the same tiles from a mask computed directly over a larger window -- a
    genuinely different code path through read_on_10m_grid, not the same
    per-tile call repeated.
    """
    from rasterio.windows import Window

    detector = ThresholdDetector(detection_resolution=10)
    # Two tiles' extent, so the cross-check window is not itself tile-sized.
    big_window = Window(5 * TILE_PX, 5 * TILE_PX, 2 * TILE_PX, 2 * TILE_PX)
    reference = detector.scene_mask(product, big_window)

    rng = np.random.default_rng(0)
    for _ in range(5):
        row = int(rng.integers(5, 7))
        col = int(rng.integers(5, 7))
        tile = detector.tile_mask(product, row, col)
        top = (row - 5) * TILE_PX
        left = (col - 5) * TILE_PX
        expected = reference[top : top + TILE_PX, left : left + TILE_PX]
        assert np.array_equal(tile, expected)


@pytest.mark.needs_product
def test_reassembled_tiles_equal_the_full_10m_mask(product):
    detector = ThresholdDetector(detection_resolution=10)
    full = detector.scene_mask(product)  # whole scene, ~2.4 GB peak -- deliberate, see docstring
    for row, col in [(0, 0), (6, 5), (19, 19)]:
        tile = detector.tile_mask(product, row, col)
        top, left = row * TILE_PX, col * TILE_PX
        assert np.array_equal(tile, full[top : top + TILE_PX, left : left + TILE_PX])


@pytest.mark.needs_product
def test_scene_layers_are_at_60m(product):
    detector = ThresholdDetector()
    layers, resolution = detector.scene_layers(product)
    assert resolution == 60
    assert set(layers) == {"cloud"}
    assert layers["cloud"].shape == (1830, 1830)
    assert layers["cloud"].dtype == bool


@pytest.mark.needs_product
def test_named_tile_ordering_matches_visible_cloud_cover(product):
    """Not exact figures (those depend on the chosen thresholds) -- just that
    a clear tile scores near zero and a heavy tile scores far higher, so the
    rule is not e.g. inverted or reading the wrong bands."""
    detector = ThresholdDetector()
    clear = compute_tile_stats(product, 11, 2, detector.tile_mask(product, 11, 2)).cloud_cover_percent
    heavy = compute_tile_stats(product, 5, 16, detector.tile_mask(product, 5, 16)).cloud_cover_percent
    assert clear < 5.0
    assert heavy > 50.0
    assert heavy > clear


def test_tile_12_19_reproduces_the_spec_3_5_table(product):
    """SPEC.md 3.5: three T_bright settings 0.02 apart span 29.26-30.59% on
    this one tile, and the lowest flips its verdict -- the sharpest single
    illustration in the project of why this task has no unique answer."""
    settings = {
        0.20: (29.2613, True),
        0.18: (29.7613, True),   # config/thresholds.json's actual choice
        0.16: (30.5898, False),
    }
    for t_bright, (expected_percent, expected_valid) in settings.items():
        detector = ThresholdDetector(t_bright=t_bright, t_ndsi=-0.20, t_cirrus=0.005, detection_resolution=10)
        stats = compute_tile_stats(product, 12, 19, detector.tile_mask(product, 12, 19))
        assert stats.cloud_cover_percent == pytest.approx(expected_percent, abs=0.01)
        assert stats.valid is expected_valid


@pytest.mark.slow
def test_two_runs_with_different_thresholds_disagree_and_both_pass(product, tmp_path, safe_dir):
    """SPEC.md B2.3: two --out folders, different thresholds, different
    report.csv, and CK1-CK8 pass on each. Runs the real CLI end to end twice,
    so it is slow; not part of the default fast loop."""
    from pipeline.checks import IN_PIPELINE_CHECKS, assert_checks
    from pipeline.run import build_parser, run

    out_a = tmp_path / "a"
    out_b = tmp_path / "b"
    run(build_parser().parse_args([
        "--safe-dir", safe_dir, "--detector", "threshold", "--out", str(out_a),
        "--t-bright", "0.20", "--t-ndsi", "-0.20", "--t-cirrus", "0.005", "--quiet",
    ]))
    run(build_parser().parse_args([
        "--safe-dir", safe_dir, "--detector", "threshold", "--out", str(out_b),
        "--t-bright", "0.16", "--t-ndsi", "-0.20", "--t-cirrus", "0.005", "--quiet",
    ]))

    report_a = (out_a / "report.csv").read_bytes()
    report_b = (out_b / "report.csv").read_bytes()
    assert report_a != report_b

    assert_checks(str(out_a), None, IN_PIPELINE_CHECKS)
    assert_checks(str(out_b), None, IN_PIPELINE_CHECKS)
