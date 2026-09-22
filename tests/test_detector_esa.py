"""B1 -- the esa backend: MSK_CLASSI loading, swap detection, and the tile mask."""
from __future__ import annotations

import numpy as np
import pytest
import rasterio
from rasterio.transform import Affine

from pipeline import detectors
from pipeline.constants import GRID, TILE_PX
from pipeline.detectors.esa import (
    ClassiMaskError,
    EsaDetector,
    read_classi_mask,
    verify_channel_order_by_brightness,
)
from pipeline.io import upsample_nearest
from pipeline.tiling import compute_tile_stats, iter_tiles


def _write_classi(path, opaque, cirrus, snow, shape=(60, 60)):
    profile = dict(
        driver="GTiff", dtype="uint8", count=3, height=shape[0], width=shape[1],
        crs="EPSG:32632", transform=Affine(60.0, 0.0, 600000.0, 0.0, -60.0, 5500020.0),
    )
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(opaque.astype("uint8"), 1)
        dst.write(cirrus.astype("uint8"), 2)
        dst.write(snow.astype("uint8"), 3)


def _classi_meta(meta, tmp_path, opaque, cirrus, snow):
    """A stand-in product whose mask path and percentages match a synthetic mask."""
    from dataclasses import replace

    path = str(tmp_path / "MSK_CLASSI_B00.tif")
    shape = opaque.shape
    _write_classi(path, opaque, cirrus, snow, shape)
    return replace(
        meta,
        sizes={**meta.sizes, 60: shape},
        cloud_coverage_assessment=100.0 * float((opaque | cirrus).mean()),
        snow_coverage_assessment=100.0 * float(snow.mean()),
        _mask_classi_path=path,
    )


def test_registered_under_esa():
    assert "esa" in detectors.available()
    assert isinstance(detectors.build("esa"), EsaDetector)


def test_esa_takes_no_parameters():
    with pytest.raises(TypeError):
        detectors.build("esa", threshold=0.5)


# ------------------------------------------------------------- read_classi_mask
def test_reads_the_three_channels_in_file_order(meta, tmp_path):
    rng = np.random.default_rng(0)
    opaque = rng.random((60, 60)) > 0.7
    cirrus = (~opaque) & (rng.random((60, 60)) > 0.9)
    snow = np.zeros((60, 60), bool)
    fake = _classi_meta(meta, tmp_path, opaque, cirrus, snow)

    got_opaque, got_cirrus, got_snow = read_classi_mask(fake)
    assert got_opaque.dtype == bool and got_opaque.shape == (60, 60)
    assert np.array_equal(got_opaque, opaque)
    assert np.array_equal(got_cirrus, cirrus)
    assert np.array_equal(got_snow, snow)


def test_rejects_overlapping_opaque_and_cirrus(meta, tmp_path):
    opaque = np.zeros((60, 60), bool)
    opaque[0:10, 0:10] = True
    cirrus = np.zeros((60, 60), bool)
    cirrus[5:15, 5:15] = True  # overlaps opaque
    snow = np.zeros((60, 60), bool)
    fake = _classi_meta(meta, tmp_path, opaque, cirrus, snow)
    with pytest.raises(ClassiMaskError, match="overlap"):
        read_classi_mask(fake)


def test_rejects_a_mismatched_wrong_band_count(meta, tmp_path):
    from dataclasses import replace

    path = str(tmp_path / "single_band.tif")
    with rasterio.open(
        path, "w", driver="GTiff", dtype="uint8", count=1, height=60, width=60,
        crs="EPSG:32632", transform=Affine(60.0, 0.0, 600000.0, 0.0, -60.0, 5500020.0),
    ) as dst:
        dst.write(np.zeros((60, 60), "uint8"), 1)
    fake = replace(meta, sizes={**meta.sizes, 60: (60, 60)}, _mask_classi_path=path)
    with pytest.raises(ClassiMaskError, match="3"):
        read_classi_mask(fake)


def test_a_swap_with_snow_is_caught_by_the_percentage_checks(meta, tmp_path):
    """Swapping opaque and snow is NOT symmetric: it changes both aggregates."""
    opaque = np.zeros((60, 60), bool)
    opaque[0:20, 0:20] = True  # 400 / 3600 = 11.1 %
    cirrus = np.zeros((60, 60), bool)
    cirrus[40:45, 40:45] = True
    snow = np.zeros((60, 60), bool)  # truly empty, like the real product

    fake = _classi_meta(meta, tmp_path, opaque, cirrus, snow)
    # Now write the mask with opaque and snow physically swapped in the file,
    # while `fake`'s metadata still says what the *correct* percentages are.
    swapped_path = str(tmp_path / "swapped.tif")
    _write_classi(swapped_path, snow, cirrus, opaque, shape=(60, 60))
    from dataclasses import replace

    broken = replace(fake, _mask_classi_path=swapped_path)
    with pytest.raises(ClassiMaskError, match="Cloud_Coverage_Assessment"):
        read_classi_mask(broken)


def test_snow_coverage_mismatch_is_reported(meta, tmp_path):
    opaque = np.zeros((60, 60), bool)
    cirrus = np.zeros((60, 60), bool)
    snow = np.zeros((60, 60), bool)
    snow[0:10, 0:10] = True
    fake = _classi_meta(meta, tmp_path, opaque, cirrus, snow)
    from dataclasses import replace

    # Metadata claims no snow, but the file has some.
    broken = replace(fake, snow_coverage_assessment=0.0)
    with pytest.raises(ClassiMaskError, match="Snow_Coverage_Assessment"):
        read_classi_mask(broken)


# --------------------------------------------------- verify_channel_order_by_brightness
def test_opaque_brighter_than_cirrus_passes():
    opaque = np.zeros((10, 10), bool)
    opaque[0:3] = True
    cirrus = np.zeros((10, 10), bool)
    cirrus[7:10] = True
    brightness = np.full((10, 10), 0.1, dtype=np.float32)
    brightness[opaque] = 0.415
    brightness[cirrus] = 0.189
    verify_channel_order_by_brightness(opaque, cirrus, brightness)  # must not raise


def test_a_swapped_opaque_cirrus_mask_is_caught_by_brightness():
    """The one swap the percentage checks cannot see -- SPEC.md 2.1's second check."""
    opaque = np.zeros((10, 10), bool)
    opaque[0:3] = True
    cirrus = np.zeros((10, 10), bool)
    cirrus[7:10] = True
    brightness = np.full((10, 10), 0.1, dtype=np.float32)
    # Swapped: the dimmer class is now labelled "opaque".
    brightness[opaque] = 0.189
    brightness[cirrus] = 0.415
    with pytest.raises(ClassiMaskError, match="swapped"):
        verify_channel_order_by_brightness(opaque, cirrus, brightness)


def test_brightness_check_is_a_noop_when_one_class_is_empty():
    empty = np.zeros((10, 10), bool)
    some = np.zeros((10, 10), bool)
    some[0:3] = True
    brightness = np.zeros((10, 10), dtype=np.float32)
    verify_channel_order_by_brightness(empty, some, brightness)
    verify_channel_order_by_brightness(some, empty, brightness)


# ------------------------------------------------------------------- EsaDetector
def test_tile_mask_is_bool_and_the_right_shape(meta, tmp_path):
    # A mask sized so 6x upsampling covers at least one full tile (549 px).
    coarse = TILE_PX // 6 + 1  # 92 px at 60 m -> 552 px at 10 m
    rng = np.random.default_rng(1)
    opaque = rng.random((coarse, coarse)) > 0.5
    cirrus = np.zeros((coarse, coarse), bool)
    snow = np.zeros((coarse, coarse), bool)
    fake = _classi_meta(meta, tmp_path, opaque, cirrus, snow)

    mask = EsaDetector(verify_channels=False).tile_mask(fake, 0, 0)
    assert mask.dtype == bool
    assert mask.shape == (TILE_PX, TILE_PX)


def test_tile_mask_equals_upsample_then_cut(meta, tmp_path):
    """SPEC.md B1.3: upsample the whole mask first, then cut -- never the reverse."""
    size = TILE_PX  # a mask exactly one tile wide at 10 m needs 549/6 = 91.5 px at 60 m,
    # so use a mask sized so 6x upsampling lands exactly on a whole number of tiles.
    coarse = 2 * TILE_PX // 6  # two tiles' worth, 183 px at 60 m -> 1098 px at 10 m
    rng = np.random.default_rng(2)
    opaque = rng.random((coarse, coarse)) > 0.6
    cirrus = (~opaque) & (rng.random((coarse, coarse)) > 0.9)
    snow = np.zeros((coarse, coarse), bool)
    fake = _classi_meta(meta, tmp_path, opaque, cirrus, snow)

    detector = EsaDetector(verify_channels=False)
    scene10 = upsample_nearest(opaque | cirrus, 6)
    for row, col in [(0, 0), (0, 1), (1, 0), (1, 1)]:
        tile = detector.tile_mask(fake, row, col)
        top, left = row * TILE_PX, col * TILE_PX
        assert np.array_equal(tile, scene10[top : top + TILE_PX, left : left + TILE_PX])


def test_reassembled_tiles_hold_no_interpolated_value(meta, tmp_path):
    coarse = 2 * TILE_PX // 6
    rng = np.random.default_rng(3)
    opaque = rng.random((coarse, coarse)) > 0.6
    cirrus = np.zeros((coarse, coarse), bool)
    snow = np.zeros((coarse, coarse), bool)
    fake = _classi_meta(meta, tmp_path, opaque, cirrus, snow)

    detector = EsaDetector(verify_channels=False)
    for row, col in [(0, 0), (0, 1), (1, 0), (1, 1)]:
        tile = detector.tile_mask(fake, row, col)
        assert set(np.unique(tile)) <= {False, True}


def test_scene_layers_returns_opaque_and_cirrus_at_60m(meta, tmp_path):
    opaque = np.zeros((60, 60), bool)
    opaque[:5] = True
    cirrus = np.zeros((60, 60), bool)
    cirrus[50:] = True
    snow = np.zeros((60, 60), bool)
    fake = _classi_meta(meta, tmp_path, opaque, cirrus, snow)

    detector = EsaDetector(verify_channels=False)
    layers, resolution = detector.scene_layers(fake)
    assert resolution == 60
    assert set(layers) == {"opaque", "cirrus"}
    assert np.array_equal(layers["opaque"], opaque)
    assert np.array_equal(layers["cirrus"], cirrus)


def test_parameters_are_empty():
    assert EsaDetector().parameters() == {}


def test_verify_channels_defaults_to_on(meta, tmp_path):
    """The default constructor genuinely reads bands for the brightness check."""
    coarse = TILE_PX // 6 + 1
    opaque = np.zeros((coarse, coarse), bool)
    opaque[:5] = True
    cirrus = np.zeros((coarse, coarse), bool)
    snow = np.zeros((coarse, coarse), bool)
    fake = _classi_meta(meta, tmp_path, opaque, cirrus, snow)
    from pipeline.metadata import ProductError

    with pytest.raises(ProductError, match="B02"):
        EsaDetector().tile_mask(fake, 0, 0)  # no band files behind this stand-in product


def test_close_releases_the_cached_mask(meta, tmp_path):
    opaque = np.zeros((60, 60), bool)
    cirrus = np.zeros((60, 60), bool)
    snow = np.zeros((60, 60), bool)
    fake = _classi_meta(meta, tmp_path, opaque, cirrus, snow)
    detector = EsaDetector(verify_channels=False)
    detector.tile_mask(fake, 0, 0)
    assert detector._cloud10 is not None
    detector.close()
    assert detector._cloud10 is None


# ------------------------------------------------------------ against the real product
@pytest.mark.needs_product
def test_real_product_reference_figures(product):
    """SPEC.md B1.4 / 0.3: 107 invalid, 293 valid, named tiles, 49 clear tiles."""
    detector = EsaDetector()
    stats = []
    for row, col in iter_tiles():
        mask = detector.tile_mask(product, row, col)
        stats.append(compute_tile_stats(product, row, col, mask))
    detector.close()

    valid = sum(1 for s in stats if s.valid)
    assert valid == 293 and (400 - valid) == 107

    mean_percent = float(np.mean([s.cloud_cover_percent for s in stats]))
    assert mean_percent == pytest.approx(20.9537, abs=5e-4)

    by_tile = {(s.row, s.col): s.cloud_cover_percent for s in stats}
    assert by_tile[(11, 2)] == pytest.approx(0.0000, abs=1e-3)
    assert by_tile[(6, 5)] == pytest.approx(40.9985, abs=1e-3)
    assert by_tile[(5, 16)] == pytest.approx(82.2927, abs=1e-3)
    assert by_tile[(12, 19)] == pytest.approx(16.2471, abs=1e-3)

    assert sum(1 for p in by_tile.values() if p == 0.0) == 49


@pytest.mark.needs_product
def test_real_product_scene_layers_reproduce_the_xml_percentage(product):
    detector = EsaDetector()
    layers, resolution = detector.scene_layers(product)
    assert resolution == 60
    union = layers["opaque"] | layers["cirrus"]
    assert 100.0 * float(union.mean()) == pytest.approx(product.cloud_coverage_assessment, abs=1e-4)
    detector.close()
