"""Read a Sentinel-2 L1C .SAFE product's metadata (F1).

Every number that varies between products is parsed here, never hard-coded. In
particular ``QUANTIFICATION_VALUE`` and ``RADIO_ADD_OFFSET`` come from
``MTD_MSIL1C.xml``: baseline 04.00 introduced the -1000 offset, and a product
processed before it simply has no ``Radiometric_Offset_List``. Omitting the
offset on a baseline >= 04.00 product shifts every reflectance by +0.1 and
silently invalidates every threshold (SPEC.md 1.1).
"""
from __future__ import annotations

import glob
import os
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

from .constants import BAND_ID, BANDS, RESOLUTIONS


def _localname(tag: str) -> str:
    """Strip the XML namespace from a tag: '{http://...}Foo' -> 'Foo'."""
    return tag.rpartition("}")[2]


def _iterfind(root: ET.Element, name: str):
    """Yield every element whose local tag name is ``name``, at any depth."""
    for element in root.iter():
        if _localname(element.tag) == name:
            yield element


def _find(root: ET.Element, name: str) -> ET.Element | None:
    return next(_iterfind(root, name), None)


def _text(root: ET.Element, name: str) -> str | None:
    element = _find(root, name)
    return None if element is None or element.text is None else element.text.strip()


def _float(root: ET.Element, name: str) -> float | None:
    raw = _text(root, name)
    return None if raw is None else float(raw)


class ProductError(RuntimeError):
    """The .SAFE folder is missing, malformed, or not a Level-1C product."""


@dataclass(frozen=True)
class ProductMetadata:
    """Everything the pipeline needs to know about one L1C granule."""

    safe_dir: str
    granule_dir: str

    product_uri: str
    processing_level: str
    processing_baseline: str
    spacecraft: str
    sensing_time: str
    generation_time: str

    epsg: int
    #: resolution (m) -> (nrows, ncols)
    sizes: dict[int, tuple[int, int]]
    #: resolution (m) -> (ulx, uly, xdim, ydim)
    geoposition: dict[int, tuple[float, float, float, float]]

    quantification_value: float
    #: band name -> radiometric offset in DN. Absent pre-baseline-04.00, in
    #: which case every band maps to 0.0.
    radio_add_offset: dict[str, float]
    has_radiometric_offset: bool
    nodata_value: int
    saturated_value: int

    cloud_coverage_assessment: float
    snow_coverage_assessment: float
    mean_sun_zenith: float
    mean_sun_azimuth: float

    #: Scene footprint as (latitude, longitude) pairs, first point repeated last.
    footprint: tuple[tuple[float, float], ...]

    _band_paths: dict[str, str] = field(repr=False, default_factory=dict)
    _tci_path: str = field(repr=False, default="")
    _mask_classi_path: str = field(repr=False, default="")

    # -------------------------------------------------------------- accessors
    def band_path(self, band: str) -> str:
        try:
            return self._band_paths[band]
        except KeyError:
            raise ProductError(
                "band %r not found in %s (have: %s)"
                % (band, self.granule_dir, ", ".join(sorted(self._band_paths)))
            ) from None

    @property
    def tci_path(self) -> str:
        if not self._tci_path:
            raise ProductError("no TCI image in %s" % self.granule_dir)
        return self._tci_path

    @property
    def mask_classi_path(self) -> str:
        if not self._mask_classi_path:
            raise ProductError("no MSK_CLASSI_B00.jp2 in %s/QI_DATA" % self.granule_dir)
        return self._mask_classi_path

    def offset(self, band: str) -> float:
        """Radiometric offset in DN for ``band``; 0.0 when the product has none."""
        return self.radio_add_offset.get(band, 0.0)

    def origin(self, resolution: int = 10) -> tuple[float, float]:
        """Upper-left corner (easting, northing) of the granule, in its own CRS."""
        ulx, uly, _, _ = self.geoposition[resolution]
        return ulx, uly

    def shape(self, resolution: int = 10) -> tuple[int, int]:
        return self.sizes[resolution]

    def footprint_bbox(self) -> tuple[float, float, float, float]:
        """(min_lat, min_lon, max_lat, max_lon) of the scene footprint."""
        lats = [lat for lat, _ in self.footprint]
        lons = [lon for _, lon in self.footprint]
        return min(lats), min(lons), max(lats), max(lons)

    def to_dict(self) -> dict:
        """A JSON-serialisable summary, written as ``scene_metadata.json``."""
        min_lat, min_lon, max_lat, max_lon = self.footprint_bbox()
        return {
            "product_uri": self.product_uri,
            "processing_level": self.processing_level,
            "processing_baseline": self.processing_baseline,
            "spacecraft": self.spacecraft,
            "sensing_time": self.sensing_time,
            "generation_time": self.generation_time,
            "epsg": self.epsg,
            "sizes": {str(k): list(v) for k, v in sorted(self.sizes.items())},
            "geoposition": {str(k): list(v) for k, v in sorted(self.geoposition.items())},
            "quantification_value": self.quantification_value,
            "radio_add_offset": dict(sorted(self.radio_add_offset.items())),
            "has_radiometric_offset": self.has_radiometric_offset,
            "nodata_value": self.nodata_value,
            "saturated_value": self.saturated_value,
            "cloud_coverage_assessment": self.cloud_coverage_assessment,
            "snow_coverage_assessment": self.snow_coverage_assessment,
            "mean_sun_zenith": self.mean_sun_zenith,
            "mean_sun_azimuth": self.mean_sun_azimuth,
            "footprint": [list(p) for p in self.footprint],
            "footprint_bbox": [min_lat, min_lon, max_lat, max_lon],
        }


def read_product(safe_dir: str) -> ProductMetadata:
    """Parse a .SAFE folder. Raises :class:`ProductError` with a clear message."""
    safe_dir = os.path.abspath(safe_dir)
    if not os.path.isdir(safe_dir):
        raise ProductError("not a directory: %s" % safe_dir)

    product_xml = os.path.join(safe_dir, "MTD_MSIL1C.xml")
    if not os.path.isfile(product_xml):
        raise ProductError(
            "%s is not a Sentinel-2 L1C .SAFE folder: MTD_MSIL1C.xml is missing" % safe_dir
        )

    granules = sorted(glob.glob(os.path.join(safe_dir, "GRANULE", "*")))
    granules = [g for g in granules if os.path.isdir(g)]
    if len(granules) != 1:
        raise ProductError(
            "expected exactly one granule in %s/GRANULE, found %d" % (safe_dir, len(granules))
        )
    granule_dir = granules[0]

    granule_xml = os.path.join(granule_dir, "MTD_TL.xml")
    if not os.path.isfile(granule_xml):
        raise ProductError("granule metadata missing: %s" % granule_xml)

    product = ET.parse(product_xml).getroot()
    granule = ET.parse(granule_xml).getroot()

    level = _text(product, "PROCESSING_LEVEL") or ""
    if "Level-1C" not in level:
        raise ProductError("expected a Level-1C product, got %r" % level)

    quant = _float(product, "QUANTIFICATION_VALUE")
    if quant is None:
        raise ProductError("QUANTIFICATION_VALUE missing from %s" % product_xml)

    offsets: dict[str, float] = {}
    for element in _iterfind(product, "RADIO_ADD_OFFSET"):
        band_id = int(element.attrib["band_id"])
        offsets[BANDS[band_id]] = float((element.text or "0").strip())
    has_offset = bool(offsets)
    if not has_offset:
        # Pre-baseline-04.00: no Radiometric_Offset_List at all. Reflectance is
        # then simply DN / QUANTIFICATION_VALUE.
        offsets = {band: 0.0 for band in BANDS}

    special = {}
    for element in _iterfind(product, "Special_Values"):
        name = _text(element, "SPECIAL_VALUE_TEXT")
        index = _text(element, "SPECIAL_VALUE_INDEX")
        if name is not None and index is not None:
            special[name] = int(float(index))

    sizes: dict[int, tuple[int, int]] = {}
    for element in _iterfind(granule, "Size"):
        resolution = int(element.attrib["resolution"])
        nrows, ncols = _text(element, "NROWS"), _text(element, "NCOLS")
        if nrows is not None and ncols is not None:
            sizes[resolution] = (int(nrows), int(ncols))

    geoposition: dict[int, tuple[float, float, float, float]] = {}
    for element in _iterfind(granule, "Geoposition"):
        resolution = int(element.attrib["resolution"])
        geoposition[resolution] = (
            float(_text(element, "ULX")),
            float(_text(element, "ULY")),
            float(_text(element, "XDIM")),
            float(_text(element, "YDIM")),
        )

    missing = [r for r in RESOLUTIONS if r not in sizes or r not in geoposition]
    if missing:
        raise ProductError("granule geocoding missing for resolutions %s" % missing)

    cs_code = _text(granule, "HORIZONTAL_CS_CODE") or ""
    match = re.search(r"(\d+)$", cs_code)
    if not match:
        raise ProductError("cannot parse EPSG code from %r" % cs_code)

    sun = _find(granule, "Mean_Sun_Angle")
    if sun is None:
        raise ProductError("Mean_Sun_Angle missing from %s" % granule_xml)

    raw_footprint = _text(product, "EXT_POS_LIST")
    if not raw_footprint:
        raise ProductError("EXT_POS_LIST missing from %s" % product_xml)
    numbers = [float(v) for v in raw_footprint.split()]
    footprint = tuple(zip(numbers[0::2], numbers[1::2]))  # (lat, lon) pairs

    image_dir = os.path.join(granule_dir, "IMG_DATA")
    band_paths: dict[str, str] = {}
    for band in BANDS:
        matches = glob.glob(os.path.join(image_dir, "*_%s.jp2" % band))
        if matches:
            band_paths[band] = matches[0]
    absent = [b for b in BANDS if b not in band_paths]
    if absent:
        raise ProductError("bands missing from %s: %s" % (image_dir, ", ".join(absent)))

    tci = glob.glob(os.path.join(image_dir, "*_TCI.jp2"))
    classi = glob.glob(os.path.join(granule_dir, "QI_DATA", "MSK_CLASSI_B00.jp2"))

    return ProductMetadata(
        safe_dir=safe_dir,
        granule_dir=granule_dir,
        product_uri=_text(product, "PRODUCT_URI") or os.path.basename(safe_dir),
        processing_level=level,
        processing_baseline=_text(product, "PROCESSING_BASELINE") or "",
        spacecraft=_text(product, "SPACECRAFT_NAME") or "",
        sensing_time=_text(granule, "SENSING_TIME") or _text(product, "PRODUCT_START_TIME") or "",
        generation_time=_text(product, "GENERATION_TIME") or "",
        epsg=int(match.group(1)),
        sizes=sizes,
        geoposition=geoposition,
        quantification_value=quant,
        radio_add_offset=offsets,
        has_radiometric_offset=has_offset,
        nodata_value=special.get("NODATA", 0),
        saturated_value=special.get("SATURATED", 65535),
        cloud_coverage_assessment=_float(product, "Cloud_Coverage_Assessment") or 0.0,
        snow_coverage_assessment=_float(product, "Snow_Coverage_Assessment") or 0.0,
        mean_sun_zenith=float(_text(sun, "ZENITH_ANGLE")),
        mean_sun_azimuth=float(_text(sun, "AZIMUTH_ANGLE")),
        footprint=footprint,
        _band_paths=band_paths,
        _tci_path=tci[0] if tci else "",
        _mask_classi_path=classi[0] if classi else "",
    )


__all__ = ["ProductMetadata", "ProductError", "read_product", "BAND_ID"]
