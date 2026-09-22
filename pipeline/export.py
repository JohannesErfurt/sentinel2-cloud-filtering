"""Deliverable output: JPEG tiles and GeoJSON masks (F6, F7)."""
from __future__ import annotations

import json
import os

import cv2
import numpy as np
from pyproj import Transformer
from rasterio import features
from shapely.geometry import mapping, shape

from .constants import GEOJSON_DECIMALS, WGS84_EPSG
from .tiling import _transformer

DEFAULT_JPEG_QUALITY = 90


# ---------------------------------------------------------------------- JPEG
def write_tile_jpeg(path: str, rgb: np.ndarray, quality: int = DEFAULT_JPEG_QUALITY) -> str:
    """Write an RGB uint8 array as a JPEG.

    The input is RGB because that is what ``io.read_tci`` returns and what
    rasterio gives for the TCI's band order; OpenCV writes BGR, so the channels
    are swapped here exactly once. A test compares per-channel means of the
    decoded file against the source to catch a regression that would otherwise
    look merely "a bit blue".
    """
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("expected (height, width, 3) RGB, got shape %s" % (rgb.shape,))
    if rgb.dtype != np.uint8:
        raise ValueError("expected uint8 RGB, got %s" % rgb.dtype)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    ok = cv2.imwrite(path, rgb[:, :, ::-1], [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise IOError("cv2 failed to write %s" % path)
    return path


# ------------------------------------------------------------------- GeoJSON
def _round_coords(geometry, decimals: int):
    """Round every coordinate of a GeoJSON geometry mapping, in place-safe form."""

    def walk(node):
        if isinstance(node, (list, tuple)):
            if node and isinstance(node[0], (int, float)):
                return [round(float(value), decimals) for value in node]
            return [walk(child) for child in node]
        return node

    return {"type": geometry["type"], "coordinates": walk(geometry["coordinates"])}


def mask_to_features(
    mask: np.ndarray,
    transform,
    class_name: str,
    simplify_m: float | None = None,
) -> tuple[list[dict], float]:
    """Vectorise a boolean mask into polygons in the raster's own CRS.

    Returns the features (with **UTM** coordinates) and their total area in
    square metres, so the caller can compare it against the mask's pixel area
    before anything is reprojected or rounded.
    """
    if mask.dtype != np.bool_:
        raise ValueError("expected a boolean mask, got %s" % mask.dtype)

    results: list[dict] = []
    total_area = 0.0
    payload = mask.astype(np.uint8)
    for geometry, value in features.shapes(payload, mask=mask, transform=transform):
        if not value:
            continue
        polygon = shape(geometry)
        if simplify_m:
            polygon = polygon.simplify(simplify_m, preserve_topology=True)
            if polygon.is_empty:
                continue
        if not polygon.is_valid:
            polygon = polygon.buffer(0)
            if polygon.is_empty:
                continue
        total_area += polygon.area
        results.append(
            {
                "type": "Feature",
                "properties": {"class": class_name},
                "geometry": mapping(polygon),
            }
        )
    return results, total_area


def features_to_wgs84(feature_list: list[dict], epsg: int, decimals: int = GEOJSON_DECIMALS):
    """Reproject features to EPSG:4326 and round, in (longitude, latitude) order.

    GeoJSON is longitude-first. Rounding to 6 decimals is about 0.1 m on the
    ground -- a hundredth of a pixel -- and roughly halves the file against full
    float repr, which is a far bigger lever than polygon simplification.
    """
    transformer = _transformer(epsg)
    output = []
    for feature in feature_list:
        geometry = feature["geometry"]

        def project(node):
            if isinstance(node, (list, tuple)) and node and isinstance(node[0], (int, float)):
                lon, lat = transformer.transform(node[0], node[1])
                return [lon, lat]
            return [project(child) for child in node]

        projected = {"type": geometry["type"], "coordinates": project(geometry["coordinates"])}
        output.append(
            {
                "type": "Feature",
                "properties": dict(feature["properties"]),
                "geometry": _round_coords(projected, decimals),
            }
        )
    return output


def build_geojson(
    layers: dict[str, np.ndarray],
    transform,
    epsg: int,
    simplify_m: float | None = None,
    decimals: int = GEOJSON_DECIMALS,
) -> tuple[dict, dict]:
    """Vectorise one or more named masks into a single FeatureCollection.

    ``layers`` maps a class name to a boolean mask, so the ESA backend can emit
    ``opaque`` and ``cirrus`` as separate features while a single-class detector
    passes one entry. Returns the collection and a per-class area summary in m2.
    """
    collected: list[dict] = []
    areas: dict[str, float] = {}
    for class_name, mask in layers.items():
        found, area = mask_to_features(mask, transform, class_name, simplify_m)
        collected.extend(found)
        areas[class_name] = area
    collection = {
        "type": "FeatureCollection",
        "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}},
        "features": features_to_wgs84(collected, epsg, decimals),
    }
    return collection, areas


def write_geojson(path: str, collection: dict) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf8", newline="") as handle:
        json.dump(collection, handle, separators=(",", ":"), sort_keys=False)
        handle.write("\n")
    return path


def geojson_utm_area(collection: dict, epsg: int) -> float:
    """Total polygon area in m2, reprojecting the written lon/lat back to UTM.

    Used by CK7 to compare the delivered file -- after rounding and any
    simplification -- against the mask's own pixel area.
    """
    transformer = Transformer.from_crs(WGS84_EPSG, epsg, always_xy=True)
    total = 0.0
    for feature in collection.get("features", []):
        geometry = feature["geometry"]

        def project(node):
            if isinstance(node, (list, tuple)) and node and isinstance(node[0], (int, float)):
                east, north = transformer.transform(node[0], node[1])
                return [east, north]
            return [project(child) for child in node]

        total += shape(
            {"type": geometry["type"], "coordinates": project(geometry["coordinates"])}
        ).area
    return total


__all__ = [
    "DEFAULT_JPEG_QUALITY",
    "write_tile_jpeg",
    "mask_to_features",
    "features_to_wgs84",
    "build_geojson",
    "write_geojson",
    "geojson_utm_area",
]
