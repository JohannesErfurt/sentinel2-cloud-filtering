"""Smoke test for the Sentinel-2 cloud-filtering environment.

Run from the repository root, inside the Python 3.11 virtual environment:

    python scripts/smoke_test.py --safe-dir <path to the .SAFE folder>

Each check is independent. Failures are reported at the end and set a non-zero
exit code. This is a plumbing test, not an evaluation of any cloud detector.
ASCII output only, so it prints cleanly on any Windows console code page.
"""
import argparse
import glob
import importlib.metadata as md
import os
import re
import sys
import tempfile
import time
import traceback

import numpy as np

BANDS = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B10", "B11", "B12"]
RES = {"B01": 60, "B02": 10, "B03": 10, "B04": 10, "B05": 20, "B06": 20, "B07": 20, "B08": 10,
       "B8A": 20, "B09": 60, "B10": 60, "B11": 20, "B12": 20}
TILE = 549
CTX = {}
RESULTS = []


def band_path(band):
    return glob.glob(os.path.join(CTX["img"], "*_%s.jp2" % band))[0]


def read_xml_constants():
    text = open(CTX["mtd"], encoding="utf8").read()
    quant = float(re.search(r"<QUANTIFICATION_VALUE[^>]*>([\d.]+)<", text).group(1))
    offsets = set(re.findall(r"<RADIO_ADD_OFFSET[^>]*>(-?\d+)<", text))
    assert len(offsets) == 1, "band offsets differ: %s" % offsets
    cloud = float(re.search(r"<Cloud_Coverage_Assessment>([\d.]+)<", text).group(1))
    return quant, float(offsets.pop()), cloud


# ---------------------------------------------------------------- checks
def check_environment():
    assert sys.version_info[:2] == (3, 11), "expected Python 3.11, got %d.%d" % sys.version_info[:2]
    opencv = sorted(d.metadata["Name"] for d in md.distributions() if d.metadata["Name"].lower().startswith("opencv"))
    assert opencv == ["opencv-python-headless"], "opencv distributions installed: %s" % opencv
    names = ["numpy", "scipy", "matplotlib", "opencv-python-headless", "rasterio", "pyproj", "shapely",
             "s2cloudless", "lightgbm", "pytest"]
    return "python %s | %s" % (sys.version.split()[0], " ".join("%s=%s" % (n, md.version(n)) for n in names))


def check_rasterio_jp2():
    import rasterio
    with rasterio.Env() as env:
        assert "JP2OpenJPEG" in env.drivers(), "GDAL JP2OpenJPEG driver missing from the rasterio wheel"
    with rasterio.open(band_path("B04")) as src:
        assert src.crs.to_epsg() == 32632, src.crs
        assert (src.transform.c, src.transform.f) == (600000.0, 5500020.0), src.transform
        assert src.res == (10.0, 10.0) and src.shape == (10980, 10980) and src.dtypes[0] == "uint16"
        return "B04: EPSG:32632, origin (600000, 5500020), 10 m, 10980x10980, uint16"


def check_window_read_matches_opencv():
    import cv2
    import rasterio
    from rasterio.windows import Window
    r, c = 6, 5
    win = Window(c * TILE, r * TILE, TILE, TILE)
    t = time.time()
    with rasterio.open(band_path("B04")) as src:
        a = src.read(1, window=win)
    t_rio = time.time() - t
    t = time.time()
    b = cv2.imread(band_path("B04"), cv2.IMREAD_UNCHANGED)[r * TILE:(r + 1) * TILE, c * TILE:(c + 1) * TILE]
    t_cv = time.time() - t
    assert a.shape == (TILE, TILE) and np.array_equal(a, b), "rasterio window != opencv crop"
    CTX["t_rio"], CTX["t_cv"] = t_rio, t_cv
    return "tile 6,5 identical in both readers; rasterio window %.2fs vs full-band opencv %.1fs" % (t_rio, t_cv)


def check_reflectance():
    import rasterio
    from rasterio.windows import Window
    quant, offset, _ = read_xml_constants()
    assert quant == 10000 and offset == -1000, (quant, offset)
    with rasterio.open(band_path("B04")) as src:
        dn = src.read(1, window=Window(5 * TILE, 6 * TILE, TILE, TILE))
    refl = (dn.astype("float32") + offset) / quant
    assert 0.0 < float(np.median(refl)) < 0.6 and refl.max() < 1.6
    return "offset %+d, quantification %d read from XML; tile 6,5 B04 median reflectance %.3f" % (
        offset, quant, np.median(refl))


def check_esa_mask():
    import cv2
    import rasterio
    _, _, cloud_meta = read_xml_constants()
    with rasterio.open(CTX["mask_path"]) as src:
        assert src.count == 3 and src.dtypes[0] == "uint8" and src.shape == (1830, 1830) and src.res == (60.0, 60.0)
        a = src.read()
    opaque, cirrus, snow = a[0] > 0, a[1] > 0, a[2] > 0
    union = opaque | cirrus
    assert (opaque & cirrus).sum() == 0
    assert abs(100 * union.mean() - cloud_meta) < 1e-3, (100 * union.mean(), cloud_meta)
    assert snow.sum() == 0
    bgr = cv2.imread(CTX["mask_path"], cv2.IMREAD_UNCHANGED)
    assert np.array_equal(bgr[..., 2] > 0, opaque) and np.array_equal(bgr[..., 1] > 0, cirrus), "cv2 BGR order changed"
    CTX["union"] = union
    return "opaque %.3f%% + cirrus %.3f%% = %.4f%% (metadata %.4f%%); snow 0; opencv BGR order confirmed" % (
        100 * opaque.mean(), 100 * cirrus.mean(), 100 * union.mean(), cloud_meta)


def check_geo():
    from pyproj import Transformer
    tf = Transformer.from_crs(32632, 4326, always_xy=True)
    lon, lat = tf.transform(600000, 5500020)
    assert abs(lat - 49.6444) < 1e-4 and abs(lon - 10.3852) < 1e-4, (lat, lon)
    r, c = 6, 5
    e0, n1 = 600000 + c * 5490, 5500020 - r * 5490
    corners = [tf.transform(e, n) for e in (e0, e0 + 5490) for n in (n1, n1 - 5490)]
    lons, lats = [p[0] for p in corners], [p[1] for p in corners]
    got = [min(lats), min(lons), max(lats), max(lons)]
    # Reference: bounding box of all FOUR corners, from an independent hand-written inverse transverse
    # Mercator that agrees with pyproj to 1e-9 deg on every corner.
    ref = [49.29258, 10.75286, 49.34311, 10.83015]
    assert max(abs(g - x) for g, x in zip(got, ref)) < 2e-5, (got, ref)
    # Grid north is not true north, so two opposite corners are NOT enough: the SW+NE-only box is ~130 m short.
    (_, lat_sw), (_, lat_ne) = (lons[1], lats[1]), (lons[2], lats[2])
    two_corner = [min(lat_sw, lat_ne), max(lat_sw, lat_ne)]
    assert two_corner[0] > got[0] + 5e-4 and two_corner[1] < got[2] - 5e-4, "two-corner box should be visibly too small"
    return ("UL corner -> %.4f N, %.4f E; tile 6,5 four-corner bbox matches the independent implementation; "
            "a two-corner box would be %.0f m short in latitude" % (lat, lon, (two_corner[0] - got[0]) * 111000))


def check_vectorise():
    import rasterio
    from pyproj import Transformer
    from rasterio import features
    from shapely.geometry import shape
    from shapely.ops import transform as shp_transform
    union = CTX["union"].astype("uint8")
    with rasterio.open(CTX["mask_path"]) as src:
        tr = src.transform
    t = time.time()
    polys = [shape(g) for g, v in features.shapes(union, mask=union.astype(bool), transform=tr)]
    dt = time.time() - t
    assert polys and all(p.is_valid for p in polys)
    area = sum(p.area for p in polys)
    expect = float(union.sum()) * 60 * 60
    assert abs(area - expect) / expect < 1e-9, (area, expect)
    tf = Transformer.from_crs(32632, 4326, always_xy=True)
    big = max(polys, key=lambda p: p.area)
    ll = shp_transform(tf.transform, big)
    minx, miny, maxx, maxy = ll.bounds
    assert 10.3 < minx < maxx < 12.0 and 48.6 < miny < maxy < 49.7
    return "%d valid polygons in %.1fs; total area %.1f km2 equals pixels x 3600 m2; reprojection to EPSG:4326 works" % (
        len(polys), dt, area / 1e6)


def check_resampling_assumptions():
    import cv2
    rng = np.random.default_rng(0)
    m = (rng.random((40, 40)) > 0.5).astype("uint8")
    assert np.array_equal(cv2.resize(m, (240, 240), interpolation=cv2.INTER_NEAREST),
                          np.repeat(np.repeat(m, 6, 0), 6, 1)), "INTER_NEAREST != np.repeat"
    x = rng.random((60, 60)).astype("float32")
    assert np.allclose(cv2.resize(x, (10, 10), interpolation=cv2.INTER_AREA), x.reshape(10, 6, 10, 6).mean(axis=(1, 3)), atol=1e-6)
    lin = cv2.resize(m, (240, 240), interpolation=cv2.INTER_LINEAR)
    assert set(np.unique(lin)) <= {0, 1} and not np.array_equal(lin, np.repeat(np.repeat(m, 6, 0), 6, 1)), \
        "expected uint8 INTER_LINEAR to round to 0/1 and move edge pixels"
    return "nearest == np.repeat; INTER_AREA == block mean; uint8 INTER_LINEAR rounds to 0/1 and moves edges (as SPEC 1.2 warns)"


def build_stack_60m(r0, r1, c0, c1, apply_offset=True):
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.windows import Window
    quant, offset, _ = read_xml_constants()
    layers = []
    for b in BANDS:
        f = 60 // RES[b]
        with rasterio.open(band_path(b)) as src:
            dn = src.read(1, window=Window(c0 * f, r0 * f, (c1 - c0) * f, (r1 - r0) * f),
                          out_shape=(r1 - r0, c1 - c0), resampling=Resampling.average, out_dtype="float32")
        layers.append((dn + (offset if apply_offset else 0.0)) / quant)
    return np.stack(layers, axis=-1).astype("float32")


def check_s2cloudless():
    from s2cloudless import S2PixelCloudDetector
    from s2cloudless.utils import MODEL_BAND_IDS
    assert MODEL_BAND_IDS == [0, 1, 3, 4, 7, 8, 9, 10, 11, 12]
    r0, r1, c0, c1 = 1030, 1330, 1530, 1830          # 300 x 300 at 60 m, around the hazy tile 12,19
    stack = build_stack_60m(r0, r1, c0, c1)
    assert stack.shape == (300, 300, 13) and np.isfinite(stack).all()
    det = S2PixelCloudDetector(threshold=0.4, all_bands=True)
    t = time.time()
    prob = det.get_cloud_probability_maps(stack[None])
    mask = det.get_mask_from_prob(prob)[0].astype(bool)
    dt = time.time() - t
    assert prob.shape == (1, 300, 300) and 0.0 <= float(prob.min()) and float(prob.max()) <= 1.0
    esa = CTX["union"][r0:r1, c0:c1]
    agree = float((mask == esa).mean())
    nooff = det.get_mask_from_prob(det.get_cloud_probability_maps(build_stack_60m(r0, r1, c0, c1, False)[None]))[0].astype(bool)
    assert nooff.mean() > mask.mean(), "expected the un-offset input to look cloudier"
    c0r, c1r, c0c, c1c = 1006, 1098, 183, 275                      # tile 11,2: genuinely clear (ESA 0.0%)
    clear = det.get_mask_from_prob(det.get_cloud_probability_maps(build_stack_60m(c0r, c1r, c0c, c1c)[None]))[0].astype(bool)
    assert CTX["union"][c0r:c1r, c0c:c1c].mean() == 0.0 and clear.mean() < 0.05, "model flags a clear-sky tile"
    CTX["s2c_clear"] = 100 * clear.mean()
    CTX["s2c_note"] = ("crop rows %d:%d cols %d:%d | ESA cloud %.1f%% | s2cloudless %.1f%% | pixel agreement %.1f%% | "
                       "WITHOUT the -1000 offset it flags %.1f%% (a %+.1f pt shift) | clear tile 11,2: %.1f%% | %.1fs" % (
                           r0, r1, c0, c1, 100 * esa.mean(), 100 * mask.mean(), 100 * agree,
                           100 * nooff.mean(), 100 * (nooff.mean() - mask.mean()), CTX["s2c_clear"], dt))
    return "model loaded from the package (no download), ran on real data, output in [0,1]. " + CTX["s2c_note"]


def check_tooling():
    import cv2
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pytest
    import scipy.ndimage as ndi
    img = (np.random.default_rng(1).random((64, 64, 3)) * 255).astype("uint8")
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "t.jpg")
        assert cv2.imwrite(p, img, [cv2.IMWRITE_JPEG_QUALITY, 90]) and cv2.imread(p).shape == (64, 64, 3)
        fig, ax = plt.subplots()
        ax.imshow(img)
        fig.savefig(os.path.join(d, "f.png"))
        plt.close(fig)
    assert ndi.binary_dilation(np.eye(5, dtype=bool)).sum() > 5
    return "JPEG write/read, matplotlib (Agg), scipy.ndimage, pytest %s" % pytest.__version__


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--safe-dir", default=None)
    args = ap.parse_args()
    safe = args.safe_dir or (glob.glob("S2C*.SAFE") or [None])[0]
    if not safe or not os.path.isdir(safe):
        sys.exit("no .SAFE folder found; pass --safe-dir")
    gran = glob.glob(os.path.join(safe, "GRANULE", "*"))[0]
    CTX.update(img=os.path.join(gran, "IMG_DATA"), mtd=os.path.join(safe, "MTD_MSIL1C.xml"),
               mask_path=os.path.join(gran, "QI_DATA", "MSK_CLASSI_B00.jp2"))
    checks = [check_environment, check_rasterio_jp2, check_window_read_matches_opencv, check_reflectance,
              check_esa_mask, check_geo, check_vectorise, check_resampling_assumptions, check_s2cloudless, check_tooling]
    for fn in checks:
        t = time.time()
        try:
            RESULTS.append((fn.__name__, True, fn(), time.time() - t))
        except Exception as e:                                        # noqa: BLE001 - report every failure
            traceback.print_exc(limit=3)
            RESULTS.append((fn.__name__, False, "%s: %s" % (type(e).__name__, e), time.time() - t))
        print("%s  %-36s %5.1fs" % ("PASS" if RESULTS[-1][1] else "FAIL", fn.__name__, RESULTS[-1][3]), flush=True)
    print("\n" + "=" * 100)
    for name, ok, note, _ in RESULTS:
        print("[%s] %s\n       %s" % ("PASS" if ok else "FAIL", name, note))
    failed = [n for n, ok, _, _ in RESULTS if not ok]
    print("=" * 100 + "\n%d/%d checks passed%s" % (len(RESULTS) - len(failed), len(RESULTS),
                                                 "" if not failed else "  FAILED: " + ", ".join(failed)))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
