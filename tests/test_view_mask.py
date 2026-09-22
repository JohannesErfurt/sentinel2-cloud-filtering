"""The esa mask viewer: PNG round-trips and the self-checks that guard them.

The interactive half (zoom, pan, hover) only exists in the browser and isn't
covered here; it was verified by hand against the real product (SPEC.md notes
this under the viewer task).
"""
from __future__ import annotations

import base64
import json

import numpy as np
import pytest

from pipeline.view_mask import (
    MASK_SIZE,
    _data_uri_jpeg,
    _data_uri_layer_png,
)


def _decode_data_uri(uri: str) -> bytes:
    header, encoded = uri.split(",", 1)
    assert header.startswith("data:image/")
    return base64.b64decode(encoded)


def test_jpeg_data_uri_decodes_back_to_the_same_image():
    import cv2

    rgb = np.zeros((20, 30, 3), dtype=np.uint8)
    rgb[:, :10] = (200, 50, 50)
    uri = _data_uri_jpeg(rgb, quality=95)
    raw = _decode_data_uri(uri)
    decoded = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)[:, :, ::-1]
    assert decoded.shape == rgb.shape
    # JPEG is lossy; a flat block of colour survives well within a few levels.
    assert np.abs(decoded.astype(int) - rgb.astype(int)).mean() < 3.0


def test_layer_png_is_transparent_where_false_and_coloured_where_true():
    import cv2

    mask = np.zeros((10, 10), dtype=bool)
    mask[2:5, 2:5] = True
    uri = _data_uri_layer_png(mask, (220, 38, 38))
    raw = _decode_data_uri(uri)
    decoded = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_UNCHANGED)
    assert decoded.shape == (10, 10, 4)
    alpha = decoded[:, :, 3]
    assert np.array_equal(alpha > 0, mask)
    # BGRA in the file; a true pixel should read back close to (220, 38, 38) RGB.
    true_pixel = decoded[3, 3]
    assert tuple(int(v) for v in true_pixel[:3][::-1]) == (220, 38, 38)


def test_layer_png_handles_an_entirely_empty_mask():
    """The real product's snow channel is empty; this must not be a special case."""
    mask = np.zeros((12, 12), dtype=bool)
    uri = _data_uri_layer_png(mask, (34, 211, 238))
    assert uri.startswith("data:image/png;base64,")


def test_layer_png_rejects_a_corrupted_round_trip(monkeypatch):
    """The self-check exists to catch exactly this: an encode that lies."""
    import cv2

    import pipeline.view_mask as vm

    real_imdecode = cv2.imdecode

    def lying_imdecode(buf, flags):
        result = real_imdecode(buf, flags)
        result[0, 0, 3] = 255 - result[0, 0, 3]  # flip one alpha value
        return result

    monkeypatch.setattr(vm.cv2, "imdecode", lying_imdecode)
    mask = np.zeros((10, 10), dtype=bool)
    mask[0, 0] = False  # so the flipped corner disagrees with the source
    with pytest.raises(AssertionError, match="round-trip"):
        vm._data_uri_layer_png(mask, (1, 2, 3))


# ------------------------------------------------------------ against the real product
@pytest.mark.needs_product
def test_build_viewer_html_against_the_real_product(safe_dir, tmp_path):
    from pipeline.view_mask import MAX_BYTES, build_viewer_html

    html = build_viewer_html(safe_dir)
    assert len(html.encode("utf8")) < MAX_BYTES
    assert "http://" not in html and "https://" not in html

    # Extract the embedded JSON payload back out of the template.
    marker = "const DATA = "
    start = html.index(marker) + len(marker)
    end = html.index(";\n", start)
    payload = json.loads(html[start:end])

    assert payload["maskSize"] == MASK_SIZE
    assert len(payload["tiles"]) == 400
    assert payload["counts"]["total"] == pytest.approx(20.9537, abs=5e-4)
    assert payload["counts"]["opaque"] == pytest.approx(19.559, abs=1e-3)
    assert payload["counts"]["cirrus"] == pytest.approx(1.394, abs=1e-3)
    assert payload["counts"]["snow"] == 0.0

    by_rc = {(t["row"], t["col"]): t for t in payload["tiles"]}
    assert by_rc[(11, 2)]["cloud_cover_percent"] == pytest.approx(0.0, abs=1e-3)
    assert by_rc[(6, 5)]["cloud_cover_percent"] == pytest.approx(40.9985, abs=1e-3)
    assert by_rc[(12, 19)]["valid"] is True


@pytest.mark.needs_product
def test_build_viewer_html_cross_checks_against_a_real_run(safe_dir, tmp_path):
    """--run cross-checks the viewer's own numbers against report.csv."""
    import os

    from pipeline.detectors import build as build_detector
    from pipeline.report import write_report_csv
    from pipeline.tiling import compute_tile_stats, iter_tiles
    from pipeline.metadata import read_product
    from pipeline.view_mask import build_viewer_html

    meta = read_product(safe_dir)
    detector = build_detector("esa")
    stats = [
        compute_tile_stats(meta, row, col, detector.tile_mask(meta, row, col))
        for row, col in iter_tiles()
    ]
    detector.close()

    run_dir = tmp_path / "esa_run"
    os.makedirs(run_dir)
    write_report_csv(str(run_dir / "report.csv"), stats)

    build_viewer_html(safe_dir, str(run_dir))  # must not raise


@pytest.mark.needs_product
def test_build_viewer_html_catches_a_tampered_run(safe_dir, tmp_path):
    import os

    from pipeline.detectors import build as build_detector
    from pipeline.report import write_report_csv
    from pipeline.tiling import compute_tile_stats, iter_tiles
    from pipeline.metadata import read_product
    from pipeline.view_mask import build_viewer_html

    meta = read_product(safe_dir)
    detector = build_detector("esa")
    stats = [
        compute_tile_stats(meta, row, col, detector.tile_mask(meta, row, col))
        for row, col in iter_tiles()
    ]
    detector.close()

    run_dir = tmp_path / "esa_run"
    os.makedirs(run_dir)
    write_report_csv(str(run_dir / "report.csv"), stats)

    text = (run_dir / "report.csv").read_text(encoding="utf8")
    lines = text.splitlines()
    lines[1] = lines[1].replace(",True", ",False") if ",True" in lines[1] else lines[1].replace(",False", ",True")
    (run_dir / "report.csv").write_text("\n".join(lines) + "\n", encoding="utf8")

    with pytest.raises(AssertionError, match="disagrees"):
        build_viewer_html(safe_dir, str(run_dir))
