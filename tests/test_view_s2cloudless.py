"""The s2cloudless live-threshold viewer: exact tile arithmetic and encoding."""
from __future__ import annotations

import json
import os

import numpy as np
import pytest

from pipeline.constants import GRID, TILE_PIXELS
from pipeline.detectors import Detector
from pipeline.io import upsample_nearest
from pipeline.view_s2cloudless import (
    PROB_SCALE,
    encode_probability,
    overlap_weights,
    scene_stats,
    threshold_code,
    tile_counts,
)


def test_overlap_weights_cover_every_10m_pixel_exactly_once():
    w = overlap_weights()
    assert (w.sum(axis=1) == 6).all()
    assert (w.sum(axis=0) == 549).all()


def test_tile_counts_equal_upsampling_then_cutting():
    rng = np.random.default_rng(0)
    mask60 = rng.random((1830, 1830)) > 0.7
    mask10 = upsample_nearest(mask60, 6)
    expected = np.array(
        [[Detector.cut(mask10, r, c).sum() for c in range(GRID)] for r in range(GRID)]
    )
    assert np.array_equal(tile_counts(mask60, overlap_weights()), expected)


def test_encoding_is_exact_for_float32_probabilities_above_one_half():
    rng = np.random.default_rng(1)
    prob = rng.uniform(0.5, 1.0, 100_000).astype(np.float32)
    q = encode_probability(prob)
    assert q.max() < PROB_SCALE
    for t in (0.5, 0.6, 0.75, 0.8, 0.95):
        assert np.array_equal(q > threshold_code(t), prob > t)


def test_encoding_clips_and_keeps_one_below_the_24_bit_ceiling():
    q = encode_probability(np.array([-0.1, 0.0, 1.0, 1.2], dtype=np.float32))
    assert q.tolist() == [0, 0, PROB_SCALE - 1, PROB_SCALE - 1]


def test_scene_stats_uses_the_integer_30_percent_rule():
    q = np.zeros((1830, 1830), dtype=np.uint32)
    q[:92, :92] = PROB_SCALE - 1  # tile 0,0 fully cloudy, plus a half-pixel spill
    stats = scene_stats(q, overlap_weights(), 0.5)
    assert stats["invalid"] == 1
    assert stats["tile_percent"][0, 0] == pytest.approx(100.0)
    assert 0 < stats["tile_percent"][0, 1] < 30


# ------------------------------------------------------------ against the real product
@pytest.mark.needs_product
def test_build_viewer_html_matches_the_real_run(safe_dir):
    from pipeline.view_s2cloudless import MAX_BYTES, build_viewer_html

    run_dir = "output/s2cloudless"
    if not os.path.isfile(os.path.join(run_dir, "cloud_probability_60m.npy")):
        pytest.skip("needs this project's own output/s2cloudless run")

    html = build_viewer_html(safe_dir, run_dir)
    assert len(html.encode("utf8")) < MAX_BYTES
    assert "http://" not in html and "https://" not in html

    marker = "const DATA = "
    start = html.index(marker) + len(marker)
    payload = json.loads(html[start : html.index(";\n", start)])

    assert payload["runCheck"]["invalid"] == 161
    assert payload["esaReference"]["invalid"] == 107
    invalid = [p["invalid"] for p in payload["curve"]]
    assert len(invalid) == 101
    assert all(a >= b for a, b in zip(invalid, invalid[1:]))
    shipped = next(p for p in payload["presets"] if "config" in p["label"])
    assert shipped["invalid"] == 161
    assert shipped["scene"] == pytest.approx(27.7646, abs=1e-4)
    assert payload["tilePixels"] == TILE_PIXELS
