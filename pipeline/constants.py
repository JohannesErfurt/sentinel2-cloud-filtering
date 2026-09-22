"""Fixed properties of Sentinel-2 L1C products and of this task's tile grid.

Nothing here is read from a product. Values that vary between products -- the
radiometric offset, the quantification value, the tile origin -- live in
``pipeline.metadata`` and are parsed from the XML (F1).
"""

# Band names in the order Sentinel-2 numbers them. The index into this list is
# the ``band_id`` attribute used by RADIO_ADD_OFFSET and SOLAR_IRRADIANCE in
# MTD_MSIL1C.xml, and it is also the band order s2cloudless expects.
BANDS = [
    "B01", "B02", "B03", "B04", "B05", "B06", "B07",
    "B08", "B8A", "B09", "B10", "B11", "B12",
]

BAND_ID = {band: index for index, band in enumerate(BANDS)}

#: Native ground sample distance of each band, in metres.
BAND_RESOLUTION = {
    "B01": 60, "B02": 10, "B03": 10, "B04": 10, "B05": 20, "B06": 20, "B07": 20,
    "B08": 10, "B8A": 20, "B09": 60, "B10": 60, "B11": 20, "B12": 20,
}

#: The resolutions a granule is delivered on, and the raster size of each.
RESOLUTIONS = (10, 20, 60)

#: Tile grid (SPEC.md 0.2). 10980 = 549 * 20 exactly, so the scene divides into
#: a 20x20 grid with no remainder strip and no padding.
TILE_PX = 549
GRID = 20
SCENE_PX = TILE_PX * GRID  # 10980
TILE_PIXELS = TILE_PX * TILE_PX  # 301401
TILE_METRES = TILE_PX * 10  # 5490

#: A tile is discarded when more than this percentage of its pixels are cloud.
CLOUD_THRESHOLD_PERCENT = 30

#: A tile is discarded when more than this fraction of it is no-data, whatever
#: its cloud percentage (SPEC.md 1.4). This product has essentially no no-data,
#: so the rule changes no verdict here; it exists for other granules.
MAX_NODATA_FRACTION = 0.5

#: The exact column order the task brief specifies for report.csv (SPEC.md 1.5).
#: No extra columns are permitted.
REPORT_COLUMNS = [
    "min_latitude",
    "min_longitude",
    "max_latitude",
    "max_longitude",
    "cloud_cover_percent",
    "valid",
]

#: Coordinate precision for GeoJSON output (SPEC.md 1.7). Six decimals is about
#: 0.1 m on the ground -- far finer than a 10 m pixel -- and roughly halves the
#: file against full float repr.
GEOJSON_DECIMALS = 6

WGS84_EPSG = 4326
