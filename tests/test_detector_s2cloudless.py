"""B3 -- the s2cloudless backend: config loading, band-order safety, and the
reference figures against the real product."""
from __future__ import annotations

import json

import numpy as np
import pytest

from pipeline import detectors
from pipeline.constants import BANDS, TILE_PX
from pipeline.detectors.s2cloudless import (
    LIBRARY_DEFAULTS,
    S2CloudlessConfigError,
    S2CloudlessDetector,
    _none_if_zero,
    build_reflectance_stack,
    load_config,
    mask_from_probability,
)
from pipeline.tiling import compute_tile_stats, iter_tiles


def test_registered_under_s2cloudless():
    assert "s2cloudless" in detectors.available()
    assert isinstance(detectors.build("s2cloudless"), S2CloudlessDetector)


def test_the_library_s2_bands_order_matches_ours():
    """SPEC.md B3.1: the assertion the detector makes at runtime, checked here
    too as a static fact so a library upgrade that changes it is caught by
    the fast suite, not only by a real run."""
    from s2cloudless.utils import S2_BANDS

    assert list(S2_BANDS) == list(BANDS)


def test_passing_the_wrong_number_of_bands_raises():
    """The library's own guard rail, exercised directly -- this is what
    protects the detector from silently misfeeding a mis-shaped stack. Cheap:
    the shape check runs before the model is ever loaded."""
    from s2cloudless import S2PixelCloudDetector

    detector = S2PixelCloudDetector(all_bands=True)
    wrong = np.zeros((1, 10, 10, 10), dtype=np.float32)  # 10 bands, but all_bands=True wants 13
    with pytest.raises(ValueError, match="all_bands"):
        detector.get_cloud_probability_maps(wrong)


# --------------------------------------------------------- mask_from_probability
def test_matches_the_library_wherever_the_library_does_not_crash():
    """The whole point of reimplementing this locally: identical output to
    S2PixelCloudDetector.get_mask_from_prob for every combination it can
    actually run (see mask_from_probability's own docstring for the one it
    cannot: average_over off with a real dilation_size)."""
    from s2cloudless import S2PixelCloudDetector

    rng = np.random.default_rng(0)
    prob = rng.random((40, 40)).astype(np.float32)
    for average_over, dilation_size in [(None, None), (2, None), (None, None), (1, 1), (2, 1)]:
        library = S2PixelCloudDetector(all_bands=True, average_over=average_over, dilation_size=dilation_size)
        library_mask = library.get_mask_from_prob(prob[None, ...], threshold=0.4)[0].astype(bool)
        local_mask = mask_from_probability(prob, 0.4, average_over, dilation_size)
        assert np.array_equal(library_mask, local_mask), (average_over, dilation_size)


def test_the_combination_that_crashes_the_library_works_here():
    """average_over=None with a real dilation_size: this is exactly the
    'Unsupported data type (=1)' OpenCV bug this function routes around, and
    it is required by B3.3's own sweep grid."""
    rng = np.random.default_rng(1)
    # A sparse field, not uniform noise: dilating a 40x40 mask where ~60% of
    # pixels already pass threshold saturates to all-True regardless of
    # dilation_size, which would not distinguish "ran without crashing" from
    # "ran and produced something trivial".
    prob = rng.random((40, 40)).astype(np.float32) * 0.1
    prob[10, 10] = 0.9
    prob[30, 30] = 0.9
    mask = mask_from_probability(prob, 0.4, None, 2)
    assert mask.dtype == bool
    assert 0.0 < mask.mean() < 1.0  # a sane, non-degenerate result: the two seeded spots, dilated


def test_mask_from_probability_reduces_to_a_plain_threshold_with_no_morphology():
    prob = np.array([[0.1, 0.5, 0.9]], dtype=np.float32)
    mask = mask_from_probability(prob, 0.4, None, None)
    np.testing.assert_array_equal(mask, [[False, True, True]])


# ------------------------------------------------------------------ config
def test_missing_config_falls_back_to_library_defaults(tmp_path):
    assert load_config(str(tmp_path / "absent.json")) == LIBRARY_DEFAULTS


def test_a_valid_config_is_read(tmp_path):
    path = tmp_path / "s2cloudless.json"
    path.write_text(json.dumps({"prob_threshold": 0.6, "average_over": None, "dilation_size": None}), encoding="utf8")
    assert load_config(str(path)) == {"prob_threshold": 0.6, "average_over": None, "dilation_size": None}


def test_malformed_json_raises(tmp_path):
    path = tmp_path / "s2cloudless.json"
    path.write_text("{not json", encoding="utf8")
    with pytest.raises(S2CloudlessConfigError, match="not valid JSON"):
        load_config(str(path))


def test_a_config_missing_a_key_raises(tmp_path):
    path = tmp_path / "s2cloudless.json"
    path.write_text(json.dumps({"prob_threshold": 0.6}), encoding="utf8")
    with pytest.raises(S2CloudlessConfigError, match="average_over"):
        load_config(str(path))


def test_none_if_zero():
    assert _none_if_zero(0) is None
    assert _none_if_zero(2) == 2
    assert _none_if_zero(None) is None


def test_explicit_kwargs_override_the_config(tmp_path):
    path = tmp_path / "s2cloudless.json"
    path.write_text(json.dumps({"prob_threshold": 0.6, "average_over": 2, "dilation_size": 1}), encoding="utf8")
    detector = S2CloudlessDetector(prob_threshold=0.7, config_path=str(path))
    assert detector.prob_threshold == 0.7  # overridden
    assert detector.average_over == 2  # from the config
    assert detector.dilation_size == 1  # from the config


def test_cli_style_zero_means_off(tmp_path):
    """--average-over 0 / --dilation-size 0 is this project's CLI spelling of
    "disable this post-processing step" (argparse has no clean way to pass
    None through a plain int flag)."""
    detector = S2CloudlessDetector(average_over=0, dilation_size=0)
    assert detector.average_over is None
    assert detector.dilation_size is None


def test_parameters_reports_everything_that_changes_the_output():
    detector = S2CloudlessDetector(prob_threshold=0.6, average_over=0, dilation_size=0)
    assert detector.parameters() == {"prob_threshold": 0.6, "average_over": None, "dilation_size": None}


def test_close_releases_the_cache():
    detector = S2CloudlessDetector()
    assert detector._cloud10 is None
    detector.close()
    assert detector._cloud10 is None


# ------------------------------------------------------------ against the real product
@pytest.mark.needs_product
def test_build_reflectance_stack_is_13_bands_clipped_at_zero(product):
    # 183 divides every band's native size evenly (1830, 5490 and 10980 are
    # all multiples of it) -- an arbitrary smaller shape like 100 does not,
    # and read_reflectance rejects an out_shape that would not block-average
    # exactly.
    stack = build_reflectance_stack(product, shape=(183, 183))
    assert stack.shape == (183, 183, 13)
    assert stack.min() >= 0.0
    assert stack.dtype == np.float32


@pytest.mark.needs_product
def test_tile_mask_is_bool_and_549_square(product):
    detector = S2CloudlessDetector()
    mask = detector.tile_mask(product, 6, 5)
    assert mask.dtype == bool
    assert mask.shape == (TILE_PX, TILE_PX)
    detector.close()


@pytest.mark.needs_product
def test_reassembled_tiles_equal_the_full_10m_mask(product):
    detector = S2CloudlessDetector()
    detector._ensure_loaded(product)
    full = detector._cloud10
    for row, col in [(0, 0), (6, 5), (19, 19)]:
        tile = detector.tile_mask(product, row, col)
        top, left = row * TILE_PX, col * TILE_PX
        assert np.array_equal(tile, full[top : top + TILE_PX, left : left + TILE_PX])
    detector.close()


@pytest.mark.needs_product
def test_a_genuinely_clear_tile_stays_mostly_clear(product):
    """SPEC.md 4.2: on tile 11,2 the model flags 0.45% with library defaults."""
    detector = S2CloudlessDetector(prob_threshold=0.4, average_over=1, dilation_size=1)
    mask = detector.tile_mask(product, 11, 2)
    assert mask.mean() < 0.05
    detector.close()


@pytest.mark.needs_product
def test_scene_layers_are_at_60m_and_boolean(product):
    detector = S2CloudlessDetector()
    layers, resolution = detector.scene_layers(product)
    assert resolution == 60
    assert set(layers) == {"cloud"}
    assert layers["cloud"].shape == (1830, 1830)
    assert layers["cloud"].dtype == bool
    detector.close()


@pytest.mark.needs_product
def test_write_scene_artifacts_saves_the_probability_map(product, tmp_path):
    detector = S2CloudlessDetector()
    detector.write_scene_artifacts(product, str(tmp_path))
    path = tmp_path / "cloud_probability_60m.npy"
    assert path.is_file()
    prob = np.load(path)
    assert prob.shape == (1830, 1830)
    assert prob.dtype == np.float32
    assert 0.0 <= prob.min() and prob.max() <= 1.0
    detector.close()


@pytest.mark.needs_product
def test_library_defaults_reproduce_spec_4_3(product):
    """SPEC.md B3.2: threshold=0.4, average_over=1, dilation_size=1 (the
    library's own constructor defaults, explicitly passed here regardless of
    what config/s2cloudless.json ships) gives 39.3% +/- 0.5 scene cloud and
    259 +/- 3 invalid tiles."""
    detector = S2CloudlessDetector(prob_threshold=0.4, average_over=1, dilation_size=1)
    stats = [
        compute_tile_stats(product, row, col, detector.tile_mask(product, row, col))
        for row, col in iter_tiles()
    ]
    detector.close()

    scene_percent = float(np.mean([s.cloud_cover_percent for s in stats]))
    invalid = sum(1 for s in stats if not s.valid)
    assert scene_percent == pytest.approx(39.3, abs=0.5)
    assert invalid == pytest.approx(259, abs=3)
