"""The live threshold viewer: quantisation round-trips and the self-checks
that guard them (B2's viewer task).

The interactive half -- sliders, hover, zoom, pan, the ESA reference toggle --
only exists in the browser; it was driven by hand against the real product
after a real bug (a resize race that left the canvas permanently scaled to
zero -- see the ResizeObserver fix in view_mask.py and view_thresholds.py).
"""
from __future__ import annotations

import numpy as np
import pytest

from pipeline.view_thresholds import (
    QUANT_RANGES,
    mask_from_quantized,
    quantize,
    quantized_threshold,
)


def test_quantize_clips_and_scales_to_a_full_byte_range():
    array = np.array([-5.0, 0.0, 0.5, 1.0, 5.0], dtype=np.float32)
    q = quantize(array, 0.0, 1.0)
    assert q.dtype == np.uint8
    np.testing.assert_array_equal(q, [0, 0, 128, 255, 255])


def test_quantized_threshold_matches_the_quantize_scale():
    """A value and its own threshold must land in the same bin, or a pixel
    sitting exactly on a real threshold would be classified inconsistently
    between the two quantisation call sites."""
    lo, hi = QUANT_RANGES["ndsi"]
    for t in (-0.4, -0.2, 0.0, 0.3):
        array = np.array([t], dtype=np.float32)
        assert int(quantize(array, lo, hi)[0]) == quantized_threshold(t, lo, hi)


def test_mask_from_quantized_matches_full_precision_closely():
    """Not exact -- that is the whole point of quantising -- but close, and
    the generator itself refuses to write a page where it is not (see
    build_viewer_html's own assertion, exercised against the real product
    below)."""
    from pipeline.masks import combined_cloud_mask

    rng = np.random.default_rng(0)
    shape = (50, 50)
    b02 = rng.uniform(0, 0.5, shape).astype(np.float32)
    b03 = rng.uniform(0, 0.5, shape).astype(np.float32)
    b04 = rng.uniform(0, 0.5, shape).astype(np.float32)
    b11 = rng.uniform(0, 0.5, shape).astype(np.float32)
    b10 = rng.uniform(0, 0.01, shape).astype(np.float32)

    full = combined_cloud_mask(b02, b03, b04, b11, b10, 0.2, -0.2, 0.005)
    q_bright = quantize(brightness_of(b02, b03, b04), *QUANT_RANGES["brightness"])
    q_ndsi = quantize(ndsi_of(b03, b11), *QUANT_RANGES["ndsi"])
    q_b10 = quantize(b10, *QUANT_RANGES["b10"])
    quant = mask_from_quantized(q_bright, q_ndsi, q_b10, 0.2, -0.2, 0.005)

    disagreement = float((full != quant).mean())
    assert disagreement < 0.05  # a handful of edge-case pixels, not a broken rule


def brightness_of(b02, b03, b04):
    from pipeline.masks import brightness

    return brightness(b02, b03, b04)


def ndsi_of(b03, b11):
    from pipeline.masks import ndsi

    return ndsi(b03, b11)


def test_extreme_thresholds_behave_sensibly():
    """T_bright=1 with T_cirrus at max flags almost nothing; T_bright=0 with
    the veto off flags everything -- SPEC.md's own sanity check for the
    threshold viewer, run directly against the quantisation arithmetic."""
    rng = np.random.default_rng(1)
    shape = (30, 30)
    bright = quantize(rng.uniform(0, 1, shape).astype(np.float32), *QUANT_RANGES["brightness"])
    ndsi_arr = quantize(rng.uniform(-1, 1, shape).astype(np.float32), *QUANT_RANGES["ndsi"])
    b10 = quantize(rng.uniform(0, 0.02, shape).astype(np.float32), *QUANT_RANGES["b10"])

    almost_nothing = mask_from_quantized(bright, ndsi_arr, b10, 1.0, -0.2, 0.02)
    assert almost_nothing.mean() < 0.05

    everything = mask_from_quantized(bright, ndsi_arr, b10, 0.0, -1.0, 0.02)
    assert everything.mean() > 0.95


# ------------------------------------------------------------ against the real product
@pytest.mark.needs_product
def test_build_viewer_html_against_the_real_product(safe_dir):
    from pipeline.view_thresholds import MAX_BYTES, build_viewer_html

    html = build_viewer_html(safe_dir)
    assert len(html.encode("utf8")) < MAX_BYTES
    assert "http://" not in html and "https://" not in html

    marker = "const DATA = "
    start = html.index(marker) + len(marker)
    end = html.index(";\n", start)
    import json

    payload = json.loads(html[start:end])

    assert payload["maskSize"] == 1830
    assert payload["initial"]["t_bright"] == pytest.approx(0.18)
    assert len(payload["selfCheckPresets"]) == 3
    for preset in payload["selfCheckPresets"]:
        assert abs(preset["full_percent"] - preset["quantized_percent"]) < 0.4


@pytest.mark.needs_product
def test_build_viewer_html_uses_the_given_config(safe_dir, tmp_path):
    import json

    from pipeline.view_thresholds import build_viewer_html

    config_path = tmp_path / "thresholds.json"
    config_path.write_text(json.dumps({"t_bright": 0.22, "t_ndsi": -0.15, "t_cirrus": 0.006}), encoding="utf8")

    html = build_viewer_html(safe_dir, str(config_path))
    marker = "const DATA = "
    start = html.index(marker) + len(marker)
    end = html.index(";\n", start)
    payload = json.loads(html[start:end])
    assert payload["initial"] == {"t_bright": 0.22, "t_ndsi": -0.15, "t_cirrus": 0.006}
