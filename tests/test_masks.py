"""B2.1 -- the three band-test formulas, on synthetic arrays."""
from __future__ import annotations

import numpy as np
import pytest

from pipeline.masks import brightness, cirrus_flag, combined_cloud_mask, ndsi


def test_brightness_is_the_mean_of_three_bands():
    b02 = np.array([[0.3]], dtype=np.float32)
    b03 = np.array([[0.6]], dtype=np.float32)
    b04 = np.array([[0.9]], dtype=np.float32)
    assert brightness(b02, b03, b04)[0, 0] == pytest.approx(0.6)


def test_brightness_is_elementwise_on_arrays():
    b02 = np.array([0.0, 0.3, 1.0], dtype=np.float32)
    b03 = np.array([0.0, 0.3, 1.0], dtype=np.float32)
    b04 = np.array([0.0, 0.3, 1.0], dtype=np.float32)
    np.testing.assert_allclose(brightness(b02, b03, b04), [0.0, 0.3, 1.0], atol=1e-6)


def test_ndsi_of_equal_bands_is_zero():
    b03 = np.array([0.1, 0.5, 0.9], dtype=np.float32)
    b11 = np.array([0.1, 0.5, 0.9], dtype=np.float32)
    np.testing.assert_allclose(ndsi(b03, b11), 0.0, atol=1e-6)


def test_ndsi_matches_the_spec_reference_table():
    """SPEC.md 3.1: cloud, bare soil, vegetation, shadow, water class means.

    The table's B03/B11 values are themselves rounded to 3 decimals for
    display, so recomputing NDSI from them does not reproduce its NDSI column
    to more than about a percent (the shadow row moves the most, since it is
    the ratio of two small, nearly-cancelling numbers). This checks the
    formula and its sign convention, not the table's own rounding.
    """
    b03 = np.array([0.553, 0.202, 0.088, 0.069, 0.062], dtype=np.float32)
    b11 = np.array([0.605, 0.263, 0.143, 0.045, 0.015], dtype=np.float32)
    expected = np.array([-0.044, -0.130, -0.238, 0.217, 0.610], dtype=np.float32)
    np.testing.assert_allclose(ndsi(b03, b11), expected, atol=0.01)


def test_ndsi_zero_denominator_does_not_raise_or_emit_inf():
    b03 = np.array([0.0, -0.001], dtype=np.float32)
    b11 = np.array([0.0, 0.001], dtype=np.float32)
    with np.errstate(all="raise"):
        result = ndsi(b03, b11)
    assert np.all(np.isfinite(result))
    assert result[0] == 0.0  # the exact-zero case


def test_cirrus_flag_is_a_simple_threshold():
    b10 = np.array([0.001, 0.006, 0.005], dtype=np.float32)
    flag = cirrus_flag(b10, t_cirrus=0.005)
    np.testing.assert_array_equal(flag, [False, True, False])
    assert flag.dtype == np.bool_


def test_combined_rule_matches_the_formula():
    # bright, low ndsi (vetoed), no cirrus -> not cloud
    b02 = np.array([[0.3, 0.3, 0.0]], dtype=np.float32)
    b03 = np.array([[0.3, 0.6, 0.0]], dtype=np.float32)
    b04 = np.array([[0.3, 0.6, 0.0]], dtype=np.float32)
    b11 = np.array([[0.9, 0.1, 0.0]], dtype=np.float32)  # high B11 -> low/negative ndsi for col0
    b10 = np.array([[0.0, 0.0, 0.01]], dtype=np.float32)  # col2: cirrus only
    mask = combined_cloud_mask(b02, b03, b04, b11, b10, t_bright=0.2, t_ndsi=-0.2, t_cirrus=0.005)
    # col0: bright=0.3>0.2 True, ndsi=(0.3-0.9)/1.2=-0.5 <= -0.2 -> vetoed -> False
    # col1: bright=0.5>0.2 True, ndsi=(0.6-0.1)/0.7=0.714 > -0.2 -> True (cloud)
    # col2: dark, but cirrus flag true -> cloud regardless
    np.testing.assert_array_equal(mask, [[False, True, True]])


def test_combined_rule_is_boolean_dtype():
    shape = (5, 5)
    zeros = np.zeros(shape, dtype=np.float32)
    mask = combined_cloud_mask(zeros, zeros, zeros, zeros, zeros, 0.2, -0.2, 0.005)
    assert mask.dtype == np.bool_


# ------------------------------------------------------------ against the real product
@pytest.mark.needs_product
def test_naive_rule_flags_the_expected_scene_fraction(product):
    """SPEC.md B2.1: brightness > 0.33 at 60m, block-mean kernel, flags 13.6% +/- 0.2."""
    from pipeline.io import read_reflectance

    shape = (1830, 1830)
    b02 = read_reflectance(product, "B02", out_shape=shape)
    b03 = read_reflectance(product, "B03", out_shape=shape)
    b04 = read_reflectance(product, "B04", out_shape=shape)
    flagged = 100.0 * float((brightness(b02, b03, b04) > 0.33).mean())
    assert flagged == pytest.approx(13.6, abs=0.2)


@pytest.mark.needs_product
def test_tests_receive_offset_corrected_reflectance(product):
    """A regression test for the failure mode of SPEC.md 1.1: without the
    offset every reflectance is 0.1 too high, so brightness > 0.33 would flag
    far more of the scene than with it."""
    from pipeline.io import read_reflectance
    from pipeline.tiling import tile_window

    window = tile_window(12, 19)  # the thin-veil tile: sensitive to the shift
    with_offset = [read_reflectance(product, b, window=window, apply_offset=True) for b in ("B02", "B03", "B04")]
    without_offset = [read_reflectance(product, b, window=window, apply_offset=False) for b in ("B02", "B03", "B04")]
    flagged_with = float((brightness(*with_offset) > 0.33).mean())
    flagged_without = float((brightness(*without_offset) > 0.33).mean())
    assert flagged_without > flagged_with
