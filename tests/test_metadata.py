"""F1 -- the product reader, and the radiometric offset it must not drop."""
from __future__ import annotations

import os

import numpy as np
import pytest

from pipeline.constants import BANDS
from pipeline.io import to_reflectance
from pipeline.metadata import ProductError, read_product

PRODUCT_XML = """<?xml version="1.0" encoding="UTF-8"?>
<n1:Level-1C_User_Product xmlns:n1="https://psd-15.sentinel2.eo.esa.int/PSD/User_Product_Level-1C.xsd">
  <n1:General_Info>
    <Product_Info>
      <PRODUCT_URI>S2C_MSIL1C_TEST.SAFE</PRODUCT_URI>
      <PROCESSING_LEVEL>Level-1C</PROCESSING_LEVEL>
      <PROCESSING_BASELINE>05.11</PROCESSING_BASELINE>
      <GENERATION_TIME>2025-10-02T14:31:20.000000Z</GENERATION_TIME>
      <PRODUCT_START_TIME>2025-10-02T10:18:51.025Z</PRODUCT_START_TIME>
      <Datatake><SPACECRAFT_NAME>Sentinel-2C</SPACECRAFT_NAME></Datatake>
    </Product_Info>
    <Product_Image_Characteristics>
      <Special_Values><SPECIAL_VALUE_TEXT>NODATA</SPECIAL_VALUE_TEXT><SPECIAL_VALUE_INDEX>0</SPECIAL_VALUE_INDEX></Special_Values>
      <Special_Values><SPECIAL_VALUE_TEXT>SATURATED</SPECIAL_VALUE_TEXT><SPECIAL_VALUE_INDEX>65535</SPECIAL_VALUE_INDEX></Special_Values>
      <QUANTIFICATION_VALUE unit="none">10000</QUANTIFICATION_VALUE>
      {offsets}
    </Product_Image_Characteristics>
  </n1:General_Info>
  <n1:Geometric_Info>
    <Product_Footprint><Product_Footprint><Global_Footprint><EXT_POS_LIST>
      49.6444 10.3852 49.6163 11.9046 48.6298 11.8475 48.6570 10.3579 49.6444 10.3852
    </EXT_POS_LIST></Global_Footprint></Product_Footprint></Product_Footprint>
  </n1:Geometric_Info>
</n1:Level-1C_User_Product>
"""

GRANULE_XML = """<?xml version="1.0" encoding="UTF-8"?>
<n1:Level-1C_Tile_ID xmlns:n1="https://psd-15.sentinel2.eo.esa.int/PSD/S2_PDI_Level-1C_Tile_Metadata.xsd">
  <n1:General_Info><SENSING_TIME>2025-10-02T10:27:28.089482Z</SENSING_TIME></n1:General_Info>
  <n1:Geometric_Info>
    <Tile_Geocoding>
      <HORIZONTAL_CS_CODE>EPSG:32632</HORIZONTAL_CS_CODE>
      <Size resolution="10"><NROWS>10980</NROWS><NCOLS>10980</NCOLS></Size>
      <Size resolution="20"><NROWS>5490</NROWS><NCOLS>5490</NCOLS></Size>
      <Size resolution="60"><NROWS>1830</NROWS><NCOLS>1830</NCOLS></Size>
      <Geoposition resolution="10"><ULX>600000</ULX><ULY>5500020</ULY><XDIM>10</XDIM><YDIM>-10</YDIM></Geoposition>
      <Geoposition resolution="20"><ULX>600000</ULX><ULY>5500020</ULY><XDIM>20</XDIM><YDIM>-20</YDIM></Geoposition>
      <Geoposition resolution="60"><ULX>600000</ULX><ULY>5500020</ULY><XDIM>60</XDIM><YDIM>-60</YDIM></Geoposition>
    </Tile_Geocoding>
    <Tile_Angles>
      <Mean_Sun_Angle><ZENITH_ANGLE unit="deg">53.5213</ZENITH_ANGLE><AZIMUTH_ANGLE unit="deg">168.3974</AZIMUTH_ANGLE></Mean_Sun_Angle>
    </Tile_Angles>
  </n1:Geometric_Info>
</n1:Level-1C_Tile_ID>
"""


def _write_safe(root, offset=-1000):
    """A .SAFE skeleton: real XML, empty band files (nothing here opens them)."""
    safe = root / "S2C_MSIL1C_TEST.SAFE"
    granule = safe / "GRANULE" / "L1C_TEST" / "IMG_DATA"
    granule.mkdir(parents=True)
    (safe / "GRANULE" / "L1C_TEST" / "QI_DATA").mkdir()
    if offset is None:
        offsets = ""
    else:
        rows = "\n".join(
            '<RADIO_ADD_OFFSET band_id="%d">%d</RADIO_ADD_OFFSET>' % (i, offset)
            for i in range(len(BANDS))
        )
        offsets = "<Radiometric_Offset_List>%s</Radiometric_Offset_List>" % rows
    (safe / "MTD_MSIL1C.xml").write_text(PRODUCT_XML.format(offsets=offsets), encoding="utf8")
    (safe / "GRANULE" / "L1C_TEST" / "MTD_TL.xml").write_text(GRANULE_XML, encoding="utf8")
    for band in BANDS:
        (granule / ("T32UPV_20251002T101851_%s.jp2" % band)).write_bytes(b"")
    (granule / "T32UPV_20251002T101851_TCI.jp2").write_bytes(b"")
    return safe


def test_constants_come_from_the_xml(tmp_path):
    meta = read_product(str(_write_safe(tmp_path)))
    assert meta.quantification_value == 10000.0
    assert meta.offset("B02") == -1000.0
    assert meta.has_radiometric_offset is True
    assert meta.epsg == 32632
    assert meta.origin(10) == (600000.0, 5500020.0)
    assert meta.shape(10) == (10980, 10980)
    assert meta.nodata_value == 0


def test_a_different_offset_in_the_xml_changes_the_result(tmp_path):
    """The offset is read, not assumed: change the file, change the reflectance."""
    normal = read_product(str(_write_safe(tmp_path / "a")))
    altered = read_product(str(_write_safe(tmp_path / "b", offset=-2000)))
    dn = np.array([[3000]], dtype=np.uint16)
    assert to_reflectance(dn, normal, "B02")[0, 0] == pytest.approx(0.2)
    assert to_reflectance(dn, altered, "B02")[0, 0] == pytest.approx(0.1)


def test_a_product_without_an_offset_list_yields_zero(tmp_path):
    """Pre-baseline-04.00 products have no offset at all; that is not an error."""
    meta = read_product(str(_write_safe(tmp_path, offset=None)))
    assert meta.has_radiometric_offset is False
    assert all(meta.offset(band) == 0.0 for band in BANDS)
    dn = np.array([[3000]], dtype=np.uint16)
    assert to_reflectance(dn, meta, "B02")[0, 0] == pytest.approx(0.3)


def test_skipping_the_offset_shifts_every_value_by_exactly_one_tenth(tmp_path):
    """The failure mode of SPEC.md 1.1, stated as arithmetic."""
    meta = read_product(str(_write_safe(tmp_path)))
    dn = np.arange(0, 20000, 137, dtype=np.uint16)
    corrected = to_reflectance(dn, meta, "B04", apply_offset=True)
    naive = to_reflectance(dn, meta, "B04", apply_offset=False)
    assert np.allclose(naive - corrected, 0.1, atol=1e-6)


def test_missing_product_reports_a_useful_error(tmp_path):
    with pytest.raises(ProductError, match="MTD_MSIL1C.xml is missing"):
        read_product(str(tmp_path))
    with pytest.raises(ProductError, match="not a directory"):
        read_product(str(tmp_path / "nope"))


def test_missing_band_is_named(tmp_path):
    safe = _write_safe(tmp_path)
    os.remove(safe / "GRANULE" / "L1C_TEST" / "IMG_DATA" / "T32UPV_20251002T101851_B11.jp2")
    with pytest.raises(ProductError, match="B11"):
        read_product(str(safe))


# ------------------------------------------------------------ against the real product
@pytest.mark.needs_product
def test_real_product_matches_the_spec_reference_table(product):
    assert product.processing_baseline == "05.11"
    assert product.quantification_value == 10000.0
    assert all(product.offset(band) == -1000.0 for band in BANDS)
    assert product.epsg == 32632
    assert product.origin(10) == (600000.0, 5500020.0)
    assert product.shape(10) == (10980, 10980)
    assert product.cloud_coverage_assessment == pytest.approx(20.9537, abs=5e-4)
    assert product.snow_coverage_assessment == 0.0
    assert product.mean_sun_zenith == pytest.approx(53.5213, abs=1e-3)
    min_lat, min_lon, max_lat, max_lon = product.footprint_bbox()
    assert (min_lat, min_lon) == pytest.approx((48.62982, 10.35791), abs=1e-4)
    assert (max_lat, max_lon) == pytest.approx((49.64444, 11.90457), abs=1e-4)


@pytest.mark.needs_product
def test_b04_median_over_tile_6_5(product):
    """SPEC.md F1: the median of B04 over tile 6,5 is 0.092 +/- 0.001."""
    from pipeline.io import read_reflectance
    from pipeline.tiling import tile_window

    patch = read_reflectance(product, "B04", window=tile_window(6, 5))
    assert float(np.median(patch)) == pytest.approx(0.092, abs=0.001)
