# SPEC — Cloud filtering in Sentinel-2 imagery

Implementation plan for the Flypix technical task, written as a checklist.

## How to read this document

- `- [ ]` is a task. Tick it (`- [x]`) only when **every** line under its **Done when** is true.
- IDs: `F*` foundations · `B1.*`, `B2.*`, `B3.*` detector backends · `E*` evaluation and selection ·
  `S*` submission · `D*` open decisions.
- `(auto)` means a script, an in-pipeline assertion or `pytest` checks it. `(manual)` means you check
  it by eye.
- A box is ticked `[x]` **only** if it was actually done and verified in this repository. Everything
  else is open.
- Numbers quoted in **Done when** lines are measurements taken against the product in this
  repository, not assumptions. Where a measurement depends on an implementation choice, the choice is
  named in the criterion and the tolerance reflects the spread.

---

## 0. Context

### 0.1 The brief

From `tech-task-flypix-ai.pdf` ("What You Need to Do", "Bonus", "Deliverables"):

| # | Requirement | Where |
|---|---|---|
| R1 | Read and crop the image into **549 × 549 pixel tiles** | F3 |
| R2 | **Detect clouds** in each tile | B1 / B2 / B3 |
| R3 | **Estimate cloud coverage** per tile | F4 |
| R4 | **Discard** tiles with more than 30 % cloud pixels (invalid) | F4 |
| R5 | Save the **valid tiles as JPEGs** | F6 |
| R6 | **CSV report**, one row per tile, columns `min_latitude, min_longitude, max_latitude, max_longitude, cloud_cover_percent, valid (True/False)` | F5 |
| R7 | *Bonus:* a Machine Learning or Deep Learning approach | B3 |
| R8 | *Bonus:* cloud coverage mask as **GeoJSON** | F7 |
| R9 | **README** with run instructions; all code and resources | S1 |
| R10 | **ZIP** all deliverables into a single file | S2 |

Constraints from the brief: any open-source tool or model is allowed; the code must be clean, modular
and runnable; the README must explain how to run it. **Estimated time: 3–5 hours.** That budget is a
design constraint on this plan, not a footnote — see §0.4.

### 0.2 The scene

All values below were read from the product's own metadata and verified against the pixels.

| Property | Value |
|---|---|
| Product | `S2C_MSIL1C_20251002T101851_N0511_R065_T32UPV_20251002T143120.SAFE` |
| Level / baseline | L1C, processing baseline 05.11 |
| Sensed | 2025-10-02 10:18:51 UTC (granule sensing time 10:27:28.089 UTC) |
| Tile / CRS | T32UPV, EPSG:32632 (UTM zone 32N) |
| Upper-left corner | E 600000, N 5500020 |
| Raster sizes | 10980² (10 m), 5490² (20 m), 1830² (60 m) |
| `QUANTIFICATION_VALUE` | 10000 |
| `RADIO_ADD_OFFSET` | −1000 (all 13 bands) |
| `Cloud_Coverage_Assessment` | 20.953686285049 % |
| `Snow_Coverage_Assessment` | 0.0 % |
| Mean sun zenith / azimuth | 53.5213° / 168.3974° |
| Footprint (`EXT_POS_LIST`) | latitude 48.62982 … 49.64444, longitude 10.35791 … 11.90457 |
| No-data (`DN == 0`) in B02 | 4 pixels of 120 560 400 — the tile is complete |

The four scene corners computed from the tile grid match `EXT_POS_LIST` to 5 × 10⁻¹⁰ degrees
(NW 49.64444, 10.38517 · NE 49.61627, 11.90457 · SE 48.62982, 11.84748 · SW 48.65703, 10.35791).

**Tiling is exact.** 10980 = 549 × 20, so the scene divides into a clean **20 × 20 grid of 400
tiles**, each 5.49 km across, with no remainder strip and no padding. If your tiler produces 401
tiles or a partial edge tile, it is a bug. (549 is not the only size that divides 10980 evenly; 610
and 732 do too. 549 gives the round 20 × 20 grid.)

This only holds on the 10 m grid. The same 5.49 km tile is 274.5 px at 20 m and 91.5 px at 60 m, so
on coarser grids tile boundaries fall *between* pixels. Every mask is therefore brought to 10 m
before tiles are cut.

**The scene is broken cumulus at about 21 % cloud, and that is what makes it a real test.** The 30 %
cut lands in the fat part of the per-tile distribution: **61 of the 400 tiles sit between 25 % and
35 %** cloud under ESA's mask, and 128 sit between 20 % and 40 %. For roughly 15 % of the grid the
keep-or-discard verdict turns on a few percentage points, so the choice of detector and of its
parameters decides the headline deliverable. Report the spread; do not present one number as the
answer.

### 0.3 Reference values

The **Done when** criteria quote these. Each names the grid and the resampling kernel it was measured
with, because both move the numbers.

**ESA `MSK_CLASSI_B00.jp2`, upsampled ×6 nearest to 10 m, cut into exact 549-px tiles:**

| Quantity | Value |
|---|---|
| Scene cloud | **20.9537 %** (19.559 opaque + 1.394 cirrus + 0.000 snow; equals the XML to 12 digits) |
| Tiles | **107 invalid, 293 valid** |
| Fully clear tiles | 49 at exactly 0.0000 %; the cloudiest tile is 93.0813 % |
| Closest to the cut | 29.8307 % (valid) and 30.2547 % (invalid) — no tile is near 30.0 by accident |
| In the 25–35 % band | **61 tiles**; in the 20–40 % band, 128 |
| Named tiles | 11,2 = **0.0000 %** · 6,5 = **40.9985 %** · 5,16 = **82.2927 %** · 12,19 = **16.2471 %** |
| Mean of the 400 tile percentages | 20.9537 % (identical to the scene figure) |
| 30 % of one tile | 90 420.3 of 301 401 pixels. **No tile can ever be exactly 30 %.** |

**Detector comparison.** Detection at 60 m, block-mean downsample, mask upsampled ×6 nearest, exact
549-px tiles. The invalid counts shift by a tile or two if you change the downsampling kernel; treat
them as ±3.

| Mask | scene cloud % | invalid tiles | validity differs from ESA on | tile 12,19 |
|---|---|---|---|---|
| ESA `MSK_CLASSI` | 20.95 | **107** | — | 16.2 % |
| `brightness > 0.33` (naive) | 13.6 | **35** | 72 tiles | 0.0 % |
| `b > 0.20 AND ndsi > −0.20 OR B10 > 0.005` | 19.8 | **83** | 30 tiles | 29.2 % |
| `b > 0.16 AND ndsi > −0.20 OR B10 > 0.005` | 21.5 | **100** | 25 tiles | **30.4 %** |
| s2cloudless, library defaults (0.4 / 1 / 1) | 39.3 | **259** | 151 tiles | 82.5 % |

Four methods put the same scene between 35 and 259 discarded tiles. That spread *is* the finding.

**Cost.** JPEG at quality 90: about 61 KB per tile, so about 18 MB for 293 tiles; mean decode error
against the TCI source 1.9 grey levels (max 2.2 over 24 tiles). GeoJSON: the 60 m ESA mask
vectorises to 602 polygons and 1.3 MB; a 10 m threshold mask to roughly 31 000 polygons and about
24 MB in UTM coordinates, more once reprojected to lat/lon.

**Speed and memory** (rasterio 1.4.4, this machine):

| Operation | Cost |
|---|---|
| Read one full 10 m band | 1.9 s, 241 MB as `uint16` |
| Read one 549 × 549 window | 0.013 s → 5.1 s for all 400 windows of one band |
| Read all 400 TCI windows (3 channels) | about 22 s |
| Read 13 bands at 60 m | 2.8 s |
| s2cloudless probability map, 1830² | about 14 s |
| One 10980² `float32` array | 482 MB (`bool`: 121 MB) |

Nothing here is slow. **Memory is the only real constraint**, and it is why detection runs tile by
tile by default (§1.8).

### 0.4 Architecture: one pipeline, three backends, one ZIP

The brief asks for one set of deliverables. R1 and R3–R6 are identical whatever produces the mask;
only R2 differs. So there is **one pipeline** and **three interchangeable detector backends**:

```
python -m pipeline.run --safe-dir <SAFE> --detector {esa,threshold,s2cloudless} --out output/<name>
```

| `--detector` | Mask source | Role |
|---|---|---|
| `esa` | ESA's shipped `MSK_CLASSI_B00.jp2` (60 m) | Baseline and reference. Not a detector you wrote. |
| `threshold` | Brightness, NDSI and B10 band tests | A detector with documented physics and three chosen thresholds. |
| `s2cloudless` | Pretrained LightGBM classifier | The ML bonus (R7). |

All three ship in the software and all three are runnable from the README. **One** of them produces
the deliverable, chosen by the rule in §5.3, which is fixed *now*, before the evaluation runs:

1. `esa` cannot be the submission. R2 asks you to *detect* clouds; reading the product's own mask is
   not detection. It ships as the reference.
2. Between `threshold` and `s2cloudless`, the winner is the one that agrees better with the visual
   audit (E2) on its fixed 16-tile sample. Agreement with ESA is reported next to it but does not
   decide.
3. If the audit cannot separate them — a difference of one tile in sixteen is within its own
   resolution — the simpler method wins. That is `threshold`.

The ML bonus is satisfied by shipping and evaluating `s2cloudless`, whether or not it is the detector
that produces the final tiles. Say so explicitly in the README.

**There is exactly one ZIP** (S2). It contains the chosen detector's `tiles/`, its `report.csv`, its
GeoJSON, the README, all code, and a `comparison/` folder holding the other two detectors' CSVs and
the evaluation write-up.

**What this plan deliberately does not build.** No bespoke HTML viewers, no browser-side threshold
sliders, no test suite for the validator. Visual checking is done with matplotlib contact sheets
(§1.9) and by opening the GeoJSON in QGIS. The budget is 3–5 hours; it belongs in the detectors and
the evaluation, not in tooling a reviewer never opens.

### 0.5 Environment

**Use Python 3.11 in a virtual environment.** The `py` launcher on this machine offers 3.8, 3.10,
3.11 and 3.14:

```bash
py -3.11 -m venv .venv
.venv\Scripts\activate           # Windows (PowerShell / cmd)
source .venv/Scripts/activate    # Git Bash
pip install -r requirements.txt
```

A virtual environment does not pick a Python version by itself. It copies whichever interpreter
created it, so the `-3.11` flag is the part that matters. Plain `py` defaults to **3.14** here.

Why not the system 3.8.10: Python 3.8 reached end of life in October 2024, and 3.8.10 dates from May
2021. Dependency resolution was checked on both:

| Package | on 3.8 | on 3.11 |
|---|---|---|
| rasterio | 1.3.11 | 1.4.4 |
| s2cloudless | 1.7.3 | 1.7.3 |
| pyproj | 3.5.0 | 3.7.2 |
| shapely | 2.0.7 | 2.1.2 |
| lightgbm | 4.6.0 | 4.7.0 |

3.14 also resolves the whole stack; 3.11 is the conservative choice for geospatial wheels.

**Install `opencv-python-headless`, not `opencv-python`.** Both provide the same `cv2` module and
shadow each other. s2cloudless requires the headless build, and nothing here needs `cv2.imshow`.

**numpy 2.x.** A fresh 3.11 environment installs numpy 2.x, which has breaking changes from 1.x.
Nothing in this spec relies on removed APIs, but watch for it when copying older snippets.

**Verified environment.** `scripts/smoke_test.py` exercises the pieces this spec depends on, against
the real product, in a 3.11 venv built from the pinned `requirements.txt`:

```bash
python scripts/smoke_test.py --safe-dir <path to the .SAFE folder>
```

It passed **10 / 10** checks (`pip check` clean): JPEG-2000 reads through rasterio, agreement with
OpenCV, the reflectance constants, `MSK_CLASSI` decoding, four-corner geocoding, mask vectorisation,
the resampling assumptions of §1.2, and s2cloudless on real data. It is a plumbing test, not an
evaluation of any detector.

**Known local gotcha: `pip install` can fail with `OSError: [WinError 53]` (network path not
found).** The cause is not this project. A stale entry on the *user* `PATH` that points at an offline
network drive makes pip fail during installation. The permanent fix is to remove or reconnect that
entry. Until then, drop it for one session without touching the stored `PATH` (adjust the `Z:*`
filter to the drive letter involved):

```powershell
$env:PATH = (($env:PATH -split ';') | Where-Object { $_ -and $_ -notlike 'Z:*' }) -join ';'
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

**Repository rules.** The 801 MB `.SAFE` folder is **not** committed: five band files exceed GitHub's
100 MB limit, and `.gitignore` already excludes it. The pipeline takes the product path as an
argument.

- [x] **F0 — Environment, pinned requirements and smoke test**
  **Done when:**
  - `py -3.11 -m venv .venv` and `pip install -r requirements.txt` succeed. *(verified)*
  - `python scripts/smoke_test.py` reports 10 / 10 and `pip check` is clean. *(verified)*
  - `.gitignore` excludes `.venv/`, the `.SAFE` folder and the task PDF. *(verified)*

---

## 1. Foundations

Implement once, in `pipeline/`. Every backend depends on these; no backend reimplements any of them.

### 1.1 Reading a band

```python
dn = src.read(1, window=...)                                  # uint16
reflectance = (dn.astype("float32") + OFFSET) / QUANT         # OFFSET = -1000, QUANT = 10000
```

Read `QUANTIFICATION_VALUE` and `RADIO_ADD_OFFSET` from `MTD_MSIL1C.xml` instead of hard-coding them,
so the code also works on pre-baseline-04.00 products where the offset is absent.

**The offset is the single most dangerous error in this dataset, because it fails silently.** Omit it
and every reflectance is 0.1 too high. Measured on the full scene: s2cloudless flags **100.00 % of
every pixel of all 400 tiles** without the offset, against 39.3 % with it. The pipeline then discards
the entire scene, writes an empty `tiles/` folder, and exits 0. Nothing crashes. This gets a hard
test, not a comment (F10).

Genuinely dark surfaces can land slightly below zero after the offset. That is sensor noise, not a
bug; clip at 0 only where a downstream consumer requires it, and say where.

- [x] **F1 — Product reader and constants** (`pipeline/metadata.py`, `pipeline/io.py`)
  **Done when:**
  - (auto) The offset and quantification value come from the XML. A copy of the XML with a different
    offset changes the result, and a missing offset yields 0.
  - (auto) `read_reflectance(band, window=None, out_shape=None)` returns offset-corrected `float32`
    reflectance. With `out_shape` it block-averages (§1.2). Raw digital numbers come from
    `read_dn`, so neither call can be mistaken for the other.
  - (auto) On the real product, the median of B04 over tile 6,5 is **0.092 ± 0.001**.
  - (auto) A unit test proves that skipping the offset shifts every value by exactly 0.1.

### 1.2 Resampling

| Direction | Continuous (reflectance) | Categorical (mask, class labels) |
|---|---|---|
| Down (10 → 60 m) | **block mean** (`cv2.INTER_AREA`) | cloud fraction per block, then threshold |
| Up (60 → 10 m) | `INTER_LINEAR` acceptable | `INTER_NEAREST`. **Never interpolate a label.** |

The three grids share the upper-left corner, so a 60 m pixel covers exactly a 6 × 6 block of 10 m
pixels and block operations are exact. Nearest-neighbour upsampling is exactly
`np.repeat(np.repeat(m, 6, 0), 6, 1)`.

**The downsampling kernel is part of the method, not an implementation detail.** Substituting
bilinear for the block mean when reducing bands to 60 m moves the threshold detector's F1 against ESA
by up to **0.005** and its invalid-tile count by a few tiles — the same size as the effect the
ablation in §3.2 exists to demonstrate. Pin `INTER_AREA` everywhere, record it in `run_summary.json`,
and do not compare numbers produced with different kernels.

> Caution: on a `uint8` mask, `cv2.INTER_LINEAR` does **not** produce visible fractions. It rounds
> back to 0/1 and silently moves edge pixels instead (486 pixels moved in a 100 × 100 test window).

- [x] **F2 — Resampling utilities** (`pipeline/io.py`)
  **Done when:**
  - (auto) `upsample_nearest(mask, 6)` equals `np.repeat(np.repeat(mask, 6, 0), 6, 1)` and never
    changes dtype or introduces a value that was not in the input.
  - (auto) The block-mean downsample equals `cv2.INTER_AREA` on random `float32` data.
  - (auto) A test documents the `uint8` `INTER_LINEAR` behaviour above, so nobody "fixes" it later.
  - (auto) `run_summary.json` records the kernel used for every resampling step.

### 1.3 Tiling and geocoding

Tile `(row, col)` for `row, col in range(20)` covers, in UTM 32N:

```
east_min  = 600000 + col * 5490          east_max = east_min + 5490
north_max = 5500020 - row * 5490         north_min = north_max - 5490
```

At 10 m its pixel window is `(col_off = col*549, row_off = row*549, width = 549, height = 549)`.
Convert the four corners to WGS84 lat/lon with `pyproj` (`always_xy=True`).

Because grid north is not true north, a tile is a slightly rotated quadrilateral in lat/lon, so
`min_latitude` and friends are the **bounding box of all four corners**, not just two of them. For
tile 6,5 a box built from only two opposite corners (SW and NE) is 0.00117° (about **130 m**) short at
each end in latitude. `pyproj` and an independent hand-written inverse transverse Mercator agree to
1e-9° on all four corners.

Reference for tile 6,5: `[49.29258, 10.75286, 49.34311, 10.83015]` (min lat, min lon, max lat, max lon).

- [ ] **F3 — Tile grid and geocoding** (`pipeline/tiling.py`) *(R1)*
  **Done when:**
  - (auto) The grid has exactly 400 tiles; every window is 549 × 549 at 10 m; painting all windows
    onto a 10980² array covers every pixel exactly once.
  - (auto) The union of the 400 lat/lon bounding boxes equals the scene footprint
    (lat 48.62982 … 49.64444, lon 10.35791 … 11.90457) within 1e-4°.
  - (auto) Tile 6,5's bounding box equals the reference above within 2e-5°, and a test rejects a
    two-corner box.
  - (manual) The README states that bounding boxes use all four corners.

### 1.4 Tile statistics, the 30 % rule and no-data

`cloud_cover_percent = 100 * cloud_pixels / 549²`, computed from the **10 m** mask, unrounded. A tile
is valid when `cloud_cover_percent <= 30`.

**The boundary is a trap only if you round first.** 30 % of a tile is 90 420.3 pixels, so no tile can
sit exactly on the cut: the two neighbouring counts give 29.9999 % (valid) and 30.0002 % (invalid).
`<=` and `<` are therefore equivalent **when applied to the unrounded value on the 10 m grid**. If you
round to one decimal *before* comparing, a tile at 29.97 % becomes "30.0" and its verdict depends on
which operator you wrote. Compute the rule with integers:
`valid = 100 * count <= 30 * 549 * 549`.

**No-data.** This product is a complete single tile: B02 has 4 zero pixels out of 120.6 million, so
no-data changes nothing here. It is still a one-line rule worth having, because the pipeline takes an
arbitrary `--safe-dir` and on a partial granule an all-no-data tile would report 0 % cloud, pass the
30 % rule, and ship as a black JPEG marked valid.

```
nodata     = (dn == 0)                       # per the product's SPECIAL_VALUE_INDEX
valid      = (cloud_pct <= 30) and (nodata_fraction <= MAX_NODATA)    # MAX_NODATA default 0.5
```

`nodata_fraction` goes in `tile_stats.csv`, not in `report.csv` — the brief fixes that schema.

- [ ] **F4 — Tile statistics and the 30 % rule** (`pipeline/tiling.py`) *(R3, R4)*
  **Done when:**
  - (auto) The mean of the 400 tile percentages equals the scene cloud percentage of the same mask.
  - (auto) `valid` is computed with the integer rule, and a test shows it equals `count < 90 420.3`
    for every count from 0 to 549².
  - (auto) A synthetic tile at 29.97 % is valid, and a test fails if rounding happens before the
    comparison.
  - (auto) A synthetic all-no-data tile is invalid even though its cloud percentage is 0, and
    `nodata_fraction` appears in `tile_stats.csv` for all 400 tiles.
  - (auto) On this product every tile has `nodata_fraction` below 1e-4, so the no-data rule changes
    no verdict here. The README says so.

### 1.5 CSV schema

Exactly 400 rows plus a header, in the column order the brief specifies, with **no extra columns**:

```
min_latitude,min_longitude,max_latitude,max_longitude,cloud_cover_percent,valid
```

Rows are **row-major** in `(row, col)`: line *i* + 2 is tile `(i // 20, i % 20)`. The CSV has no
identity columns, so row order, the JPEG file names and the world files (§1.6) are the only link to a
tile; document this in the README. Write `cloud_cover_percent` with at least 4 decimals and `valid`
as `True` / `False`.

Everything else — tile row and column, the opaque/cirrus split, mean cloud probability,
`nodata_fraction`, the per-tile agreement with the reference — goes in a sidecar `tile_stats.csv`
that carries explicit `tile_row` and `tile_col` columns.

- [x] **F5 — CSV writer** (`pipeline/report.py`) *(R6)*
  **Done when:**
  - (auto) Checks CK2 and CK3 pass (§1.9).
  - (auto) Row order is row-major and the first data row is tile (0,0).
  - (auto) `tile_stats.csv` has 400 rows and carries `tile_row`, `tile_col` and `nodata_fraction`.

### 1.6 JPEG export and world files

Write only valid tiles. The source is the 10 m `TCI.jp2`, which is already 8-bit RGB. **`TCI` is
display-stretched and must never be used for detection, only for output imagery.** Read a window with
rasterio, convert RGB to BGR for OpenCV, and write at quality 90 (about 61 KB per tile). Name files
`tile_rRR_cCC.jpg`.

**Write a `.jgw` world file beside each JPEG.** Six lines of plain text:

```
10.0            # pixel size in x
0.0
0.0
-10.0           # pixel size in y
<east_min + 5>  # UTM easting of the centre of the top-left pixel
<north_max - 5> # UTM northing of the centre of the top-left pixel
```

`report.csv` cannot carry a tile identifier — the brief fixes its six columns — so the world file is
what makes the imagery self-locating. It costs nothing, it is a standard, and dragging any tile into
QGIS lands it in the right place, which is a direct visual proof that the geocoding in F3 is correct.

- [ ] **F6 — JPEG and world-file writer** (`pipeline/export.py`) *(R5)*
  **Done when:**
  - (auto) Check CK5 passes.
  - (auto) A test catches a red/blue channel swap (per-channel means of the decoded JPEG match the
    TCI window within 2 grey levels).
  - (auto) Every JPEG has a matching `.jgw`, and the easting/northing it encodes equals the F3 grid
    for that tile exactly.
  - (manual) One tile dragged into QGIS lands on the correct part of the scene.

### 1.7 GeoJSON export

Vectorise the boolean mask with `rasterio.features.shapes`, reproject the coordinates from UTM to
EPSG:4326 with `always_xy=True` (GeoJSON is longitude, latitude), and keep the class in a `class`
property.

**Size is controlled by coordinate precision, not by simplification.** A 10 m threshold mask of this
scene is roughly 31 000 polygons and about 24 MB in UTM coordinates, and more once lat/lon values are
written at full float repr. Rounding lat/lon to **6 decimals** is about 0.1 m on the ground and
roughly halves the file without moving any vertex to a different pixel. Apply that first. Only if it
is still too large, either simplify (tolerance at most 10 m, stated in the README) or vectorise the
mask at 60 m instead of 10 m — GeoJSON is a bonus deliverable and nobody inspects 31 000 polygons.
Whichever you do, say it in the README and relax CK7's area tolerance to 0.5 %.

- [ ] **F7 — GeoJSON writer** (`pipeline/export.py`) *(R8, bonus)*
  **Done when:**
  - (auto) Check CK7 passes.
  - (auto) A test proves the coordinate order is (longitude, latitude).
  - (auto) Coordinates are rounded to 6 decimals, and the file is at most 15 MB.
  - (auto) `run_summary.json` records the vectorisation grid and any simplification tolerance.
  - (manual) It opens in QGIS or geojson.io and lands over Franconia (Nuremberg area).

### 1.8 The pipeline command

```
python -m pipeline.run --safe-dir <SAFE> --detector {esa,threshold,s2cloudless} --out output/<name>
```

| Flag | Meaning |
|---|---|
| `--t-bright`, `--t-ndsi`, `--t-cirrus` | threshold backend; override `config/thresholds.json` |
| `--prob-threshold`, `--average-over`, `--dilation-size` | s2cloudless backend |
| `--detection-resolution {10,60}` | where the tests run (D2) |
| `--save-tile-masks` | write `tile_masks/tile_rRR_cCC.png` for all 400 tiles |
| `--jpeg-quality` | default 90 |

Outputs, per run, into `--out`: `tiles/` (valid JPEGs plus `.jgw`), `report.csv`, `tile_stats.csv`,
`cloud_mask.geojson`, `run_summary.json`.

**Tile by tile is the default path.** Reading one 549 × 549 window costs 0.013 s, so all 400 windows
of one band take about 5 s — there is no speed argument for holding the scene in memory, and a
scene-wide `float32` band is 482 MB. Building the threshold mask scene-wide at 10 m needs B02, B03,
B04 plus upsampled B11 and B10 and peaks around 2.4 GB. Stream the tiles; keep a scene-wide assembly
behind an optional flag, used only for the GeoJSON and for the cross-check in B2.3.

- [ ] **F8 — Pipeline command** (`pipeline/run.py`)
  **Done when:**
  - (auto) The command above produces the five outputs listed, for each of the three detectors.
  - (auto) `run_summary.json` records: detector, all parameters actually used, the resampling kernels,
    scene cloud %, valid and invalid counts, package versions, runtime and peak memory.
  - (auto) It exits 0, `--help` documents every option, and it fails loudly with a clear message if
    `--safe-dir` is wrong.
  - (auto) Default path is streaming; peak resident memory stays under 1 GB for all three detectors.
  - (auto) CK8 passes (determinism).

### 1.9 Checks, tests and looking at the output

**Deliverable checks.** CK1–CK6 run as assertions **inside** the pipeline at the end of every run, so
a bad run fails instead of producing quiet nonsense. CK7 and CK8 run in `check_deliverables.py`.

| ID | Check |
|---|---|
| CK1 | **Grid.** 400 tiles, rows and columns 0–19, each a 549 × 549 window at 10 m, covering the 10980² raster exactly once. |
| CK2 | **CSV schema.** The header is exactly the six columns of §1.5; 400 data rows; no extra columns. |
| CK3 | **CSV values.** `cloud_cover_percent` is in [0, 100] with at least 4 decimals; `valid` is `True`/`False` and equals the integer 30 % rule combined with the no-data rule on every row. |
| CK4 | **Geography.** Every box has min < max and equals the recomputed four-corner box within 1e-6°; the union of the 400 boxes equals the scene footprint within 1e-4°. |
| CK5 | **JPEGs.** The files in `tiles/` are exactly the valid rows (`tile_rRR_cCC.jpg` plus `.jgw`); none for invalid tiles; each decodes to 549 × 549 × 3; mean absolute difference to the TCI window is at most 3 grey levels. |
| CK6 | **Consistency.** The mean of the 400 percentages equals the scene cloud percentage in `run_summary.json` within 1e-3; the count of `True` rows equals the number of JPEGs. |
| CK7 | **GeoJSON.** A valid FeatureCollection in (lon, lat) order, every coordinate inside the footprint, every geometry valid, and total polygon area (in UTM) equal to the mask's pixel area within 1e-6 relative — or 0.5 % if §1.7 simplification was applied. |
| CK8 | **Determinism.** Running the same detector twice gives a byte-identical `report.csv` and the same JPEG file list. |

**Looking at the output.** One script, `scripts/contact_sheet.py`, replaces any interactive viewer:

- `--tiles` mode: a grid of the named tiles (11,2 / 6,5 / 5,16 / 12,19) plus any tiles you name, in
  true colour with each detector's mask outlined in its own colour. One PNG.
- `--thresholds` mode: the same tiles rendered at four threshold settings, for choosing `T_bright`,
  `T_ndsi` and `T_cirrus` by eye. One PNG.
- `--scene` mode: the whole scene at 60 m with the chosen mask overlaid, plus the 20 × 20 grid and
  invalid tiles tinted. One PNG for the README.

- [ ] **F9 — Checks and contact sheets** (`pipeline/checks.py`, `scripts/check_deliverables.py`, `scripts/contact_sheet.py`)
  **Done when:**
  - (auto) CK1–CK6 run at the end of every pipeline run and abort it on failure.
  - (auto) `python scripts/check_deliverables.py --out <folder> [--safe-dir <SAFE>]` runs CK1–CK8 and
    exits non-zero on any failure. Without `--safe-dir` it runs the structural subset (CK1–CK4, CK5
    without the TCI comparison, CK6, CK7 without the area comparison), which is what runs on an
    unzipped deliverable.
  - (auto) `scripts/contact_sheet.py` produces the three PNGs above and each is at most 5 MB.
  - (manual) On the contact sheet, each detector's outline visibly follows the cloud in tiles 6,5 and
    5,16, and tile 12,19 shows the thin veil the detectors disagree about.

- [x] **F10 — Unit tests** (`tests/`)
  **Done when:**
  - (auto) `pytest` passes on a machine that does not have the `.SAFE` folder.
  - (auto) Tests cover the two places a silent error can hide: **four-corner geocoding** (F3,
    including rejection of a two-corner box) and the **integer 30 % rule with no-data** (F4). Both use
    small synthetic arrays; no band file is committed.
  - (auto) **The offset regression test**, run against the real product when `--safe-dir` is
    available and skipped otherwise: with the offset, s2cloudless flags 39.3 % ± 1 of the scene;
    without it, at least 99 %. This is the test that catches the failure mode of §1.1.

---

## 2. Backend `esa` — the product's own mask

**Role.** The reference every other detector is reported against, and the cheapest correct answer to
R1 and R3–R6. It is **not** the submission: R2 asks you to detect clouds, and this reads a mask
someone else computed.

### 2.1 What the product gives you

| Source | Content |
|---|---|
| `MTD_MSIL1C.xml` | `Cloud_Coverage_Assessment` 20.9537 %, `Snow_Coverage_Assessment` 0.0 % |
| `MTD_TL.xml` | `CLOUDY_PIXEL_PERCENTAGE` (same value), sun/viewing angles, mask filenames |
| `QI_DATA/MSK_CLASSI_B00.jp2` | **per-pixel classification mask, 60 m** |

`MSK_CLASSI_B00.jp2` is 1830 × 1830 × 3, `uint8`, three binary channels. In file (RGB) order these
are **opaque cloud, cirrus, snow/ice**. OpenCV returns BGR; rasterio returns them in file order:

```python
opaque = a[0] > 0     # 19.559 % of scene   (OpenCV: a[..., 2])
cirrus = a[1] > 0     #  1.394 %            (OpenCV: a[..., 1])
snow   = a[2] > 0     #  0.000 %            (OpenCV: a[..., 0]) matches Snow_Coverage_Assessment
cloud  = opaque | cirrus   # 20.9537 %, no overlap between the two
```

Channel order was confirmed two ways: the snow channel is empty, matching the metadata; and mean
visible brightness is 0.415 inside `opaque` versus 0.189 inside `cirrus`.

**The mask is 60 m but the tiles are 549 px at 10 m.** Upsample by 6 with nearest neighbour *first*,
then cut exact 549-pixel tiles. Cutting at 60 m would put tile edges half-way through a pixel
(91.5 px) and shift the counts.

### 2.2 Tasks

- [x] **B1.1 — Parse the scene metadata**
  **Done when:**
  - (auto) `scene_metadata.json` contains every constant in the §0.2 table (product, baseline, sensing
    time, CRS, upper-left corner, offset, quantification value, cloud and snow percentages, sun
    angles, footprint), each read from the XML.
  - (auto) Every value equals the §0.2 table.

- [x] **B1.2 — Load `MSK_CLASSI_B00.jp2` and check it**
  **Done when:**
  - (auto) Three boolean arrays of shape 1830 × 1830 are returned as `opaque`, `cirrus`, `snow`.
  - (auto) `(opaque | cirrus).mean() * 100` equals `Cloud_Coverage_Assessment` to 4 decimals
    (20.9537); `opaque & cirrus` is empty; `snow` is empty.
  - (auto) The loader **raises** if the channels are swapped. A test feeds it a swapped mask.

- [x] **B1.3 — Produce the 10 m mask** *(R2, as a baseline)*
  **Done when:**
  - (auto) The 10 m mask equals `np.repeat(np.repeat(cloud, 6, 0), 6, 1)`, is `bool`, and holds no
    interpolated value.
  - (auto) Reassembling the 400 tile masks gives back the full 10 m mask exactly.

- [x] **B1.4 — Full run and reference figures**
  **Done when:**
  - (auto) `--detector esa` produces all five outputs and passes CK1–CK8.
  - (auto) **107 tiles invalid, 293 valid**; mean of the 400 percentages **20.9537 % ± 0.0005**;
    exactly 49 tiles at 0.0000 %.
  - (auto) Named tiles match §0.3 within 0.001: 11,2 = 0.0000; 6,5 = 40.9985; 5,16 = 82.2927;
    12,19 = 16.2471.
  - (auto) The GeoJSON has separate features for `opaque` and `cirrus`, and their union area equals
    20.9537 % of the scene (about **2 526 km²**, ±0.1).
  - (auto) The scene percentage agrees between `MTD_MSIL1C.xml`, `MTD_TL.xml`, the mask and the mean
    of the CSV.

- [x] **B1.5 — Write down the reference's limits**
  **Done when:**
  - (manual) `output/comparison/reference_notes.md` states the limits with numbers: 60 m, binary, no
    shadow class, no probabilities, and tile 12,19 flagged at 16.2 % although the veil visibly covers
    the whole tile.
  - (manual) It states plainly that this mask is a catalogue-grade product for filtering archive
    searches, and is used here as a baseline, **not** as ground truth.

---

## 3. Backend `threshold` — band tests

**Role.** A detector with documented physics and three thresholds you chose. A candidate for the
submission.

### 3.1 The three tests

All operate on offset-corrected reflectance.

**Brightness.** Thick cloud is bright across the visible spectrum.

```
brightness = (B02 + B03 + B04) / 3
```

**NDSI.** Normalised difference snow index. Its textbook job is telling snow from cloud. This scene
has no snow (`Snow_Coverage_Assessment` = 0), so here it plays a different role.

```
NDSI = (B03 - B11) / (B03 + B11)
```

It is computed **per pixel**, so it is a raster the same shape as the bands. B11 is 20 m, so bring
both bands to one grid first, and guard the zero denominator.

**It measures a spectral slope:** how much brighter a surface is in the shortwave infrared (B11,
1610 nm) than in green (B03, 560 nm). Cloud is spectrally flat ("white"), so its NDSI is near zero.
Soil and vegetation get brighter toward the SWIR, so theirs is negative. Water and shadow absorb the
SWIR, so theirs is positive, and snow would be strongly positive. (Clouds do not absorb strongly at
1610 nm; cloud is brighter at B11 than at B03. The strong cloud absorption comes further out, at
2190 nm.) Class-mean spectra from the study window:

| Surface | B03 | B11 | NDSI |
|---|---|---|---|
| cloud | 0.553 | 0.605 | −0.044 |
| bare soil | 0.202 | 0.263 | −0.130 |
| vegetation | 0.088 | 0.143 | −0.238 |
| shadow | 0.069 | 0.045 | +0.217 |
| water | 0.062 | 0.015 | +0.610 |

Used as a **veto** on the brightness test (`NDSI > T_ndsi`), it removes bright pixels whose spectrum
is too SWIR-heavy to be a clean cloud pixel, and it passes cloud. Water and shadow also pass, but
they are too dark for the brightness test to have flagged them anyway. Justify it in the README as a
"whiteness veto", not as a snow test. The veto guards only the low side: snow has a high NDSI and
would pass it, so a snow scene would need an upper bound.

**B10 cirrus.** B10 sits inside a water-vapour absorption band, so it is blind to the ground by
design. Anything it sees is high in the atmosphere.

```
cirrus_flag = B10 > T_cirrus
```

Measured B10 means: clear tile 11,2 → 0.0014; hazy tile 12,19 → 0.0041; heavy tile 5,16 → 0.0135.
The **ratio to a clear-sky floor** is the signal, not the absolute value.

**Combined rule:**

```
cloud = (brightness > T_bright  AND  NDSI > T_ndsi)  OR  (B10 > T_cirrus)
```

### 3.2 Choosing the thresholds

**You choose them by inspection**, using the `--thresholds` contact sheet (F9), and record them in
`config/thresholds.json` with one sentence of justification each. ESA's mask is **not** the target.

**Starting values:** `T_bright` between 0.16 and 0.20, `T_ndsi` = −0.20, `T_cirrus` = 0.005.

Agreement with ESA's mask, at 60 m with the block-mean kernel, reported here **as a reference point
only**:

| Rule | Precision | Recall | F1 |
|---|---|---|---|
| `brightness > 0.33` (naive first attempt) | 0.81 | 0.53 | 0.64 |
| `brightness > 0.16` (best single threshold) | 0.66 | 0.76 | 0.71 |
| `b > 0.20 AND ndsi > −0.20 OR B10 > 0.005` | 0.75 | 0.69 | 0.72 |
| `b > 0.16 AND ndsi > −0.20 OR B10 > 0.005` (best over the widened grid) | 0.72 | 0.73 | 0.72 |

**Two decimals, deliberately.** Swapping the downsampling kernel for bilinear shifts every one of
these by up to 0.005, and the ablation below has an effect size of about 0.01 per test. The third
decimal is implementation noise, not signal, so this spec neither quotes it nor tests against it.

What each test contributes (best F1 over the widened grid, against `opaque | cirrus` at 60 m):

| Tests | Best F1 | at |
|---|---|---|
| brightness only | 0.71 | `> 0.16` |
| + NDSI veto | 0.72 | `> 0.14`, NDSI `> −0.20` |
| + B10 | 0.72 | `> 0.20`, B10 `> 0.005` |
| all three | 0.72 | `> 0.16`, NDSI `> −0.20`, B10 `> 0.005` |

**The honest reading: the three combined rules are indistinguishable at this resolution.** The extra
tests are justified physically — the NDSI veto removes bright non-white ground, B10 sees only the
upper atmosphere — not by F1. Say that, and do not present the ablation as showing an improvement it
does not show.

The widened grid was brightness 0.10–0.30, NDSI off and −0.40–0.10, B10 off and 0.003–0.008. An
earlier sweep started at brightness 0.20 and so had its best value on the edge of the grid; **a sweep
whose optimum sits on its own boundary has not found the optimum.** Check this for every sweep in
this document, including B3.3's.

Two caveats for the README:

- **F1 plateaus around 0.72, and that means less than it looks.** ESA's mask is coarse and misses
  thin cloud decks that a model flags (§4.4). This F1 measures agreement with a baseline, not
  accuracy; pushing it higher would mostly mean copying ESA's blind spots.
- **B10 is a weak standalone cirrus detector.** Against ESA's *cirrus channel alone* its best F1 is
  only 0.22 (precision 0.13), because B10 also responds to thick low cloud. Its value is marginal
  recall: at `T = 0.005` it recovers 10 % of the ESA cloud that brightness + NDSI miss, at a cost of
  0.40 % of the scene in new false positives. Include it, but do not oversell it.

### 3.3 Where to run the detection

Measured on this scene, comparing `brightness > 0.33` computed natively at 10 m against the same rule
computed at 60 m and upsampled back:

- pixel agreement **98.4 %**
- per-tile cloud percentage differs by **0.13 points on average**, 0.47 points at worst
- **0 of 400 tiles** change validity
- 41 % of cloud-edge 60 m pixels are internally mixed

**Recommendation: compute each test on the finest grid its bands allow, combine at 10 m, and cut
tiles there.** Brightness uses 10 m bands natively. NDSI comes from 20 m B11, so it is imprecise in a
ring roughly one 20 m pixel wide at cloud edges. B10 is **60 m**, so upsampled to 10 m the cirrus
branch is blocky in 6 × 6 squares across its whole extent — not just at edges. Say this rather than
implying all the coarse-band error is edge error.

Running everything at 60 m is also defensible: it is simpler and reads all bands on one grid. If you
take that route, say so and cite the agreement figures above. Do **not** justify either choice on
speed — a full 10 m band reads in 1.9 s and a single tile window in 0.013 s. The real constraint is
memory (§1.8), and streaming tiles removes it.

### 3.4 Tasks

- [ ] **B2.1 — Implement the three tests** (`pipeline/masks.py`)
  **Done when:**
  - (auto) `brightness`, `ndsi` and `cirrus_flag` follow the formulas in §3.1 exactly.
  - (auto) Unit tests on synthetic values: brightness of (0.3, 0.6, 0.9) = 0.6; NDSI of equal bands
    = 0; a zero denominator does not raise or emit `inf`.
  - (auto) A regression test shows the tests receive offset-corrected reflectance.
  - (auto) On the real product at 60 m with the **block-mean** kernel, `brightness > 0.33` flags
    **13.6 % ± 0.2** of the scene. The criterion names the kernel because a different one moves this.
  - (manual) The README states the brightness definition used, and what NDSI measures here (a
    green-to-SWIR slope, cloud near 0) on a scene with no snow.

- [ ] **B2.2 — Choose the thresholds and record them**
  **Done when:**
  - (manual) You have looked at the `--thresholds` contact sheet covering tiles 11,2 / 6,5 / 5,16 /
    12,19 and a handful of others, and recorded the three values in `config/thresholds.json` with one
    sentence of justification each.
  - (auto) The pipeline uses exactly those values, and `run_summary.json` echoes them.
  - (auto) **Reference only.** A sweep writes `output/comparison/threshold_sweep.csv` covering
    brightness 0.10–0.40, NDSI (off and −0.40–0.10) and B10 (off and 0.002–0.012), with precision,
    recall, F1 and IoU against the `esa` mask at 60 m, block-mean kernel. **The optimum must lie
    inside the grid, not on its edge**; the sweep asserts this and fails if it does not.
  - (auto) The sweep reproduces §3.2 within **±0.02**, and the ablation table, at two decimals.
  - (manual) The README reports the ESA-agreement optimum (`T_bright` 0.16, `T_ndsi` −0.20,
    `T_cirrus` 0.005) next to the values you chose, **with the comment below**.
  - (manual) The README reports how many tiles change validity when each threshold moves by a small
    step, so the sensitivity is visible.

  **The comment the README must contain (in your own words).** ESA's mask is the output of an
  algorithm, not a measurement of the truth. It is coarse (60 m), binary, has no shadow class, and
  can miss thin cloud that is visible in the imagery. Choosing thresholds to maximise agreement with
  it copies its errors and blind spots, and turns your detector into an imitation of ESA's mask
  instead of the independent detection the brief asks for. If ESA's mask were ground truth there
  would be nothing left to detect. Agreement with ESA is a consistency check reported after the
  choice; it is not accuracy.

  **No sanity band is imposed on your choice.** For orientation only: the ESA-agreement optimum
  produces a scene cloud around 21 % and about 100 invalid tiles, and `T_bright = 0.20` produces
  about 20 % and 83. A materially different choice is legitimate — explain it in the README rather
  than treating it as an error.

- [ ] **B2.3 — Build the mask and run** *(R2)*
  **Done when:**
  - (auto) `--detector threshold` produces all five outputs and passes CK1–CK8.
  - (auto) Brightness is computed natively at 10 m, NDSI and B10 from upsampled coarser bands, and
    the tests are combined at 10 m.
  - (auto) Streaming five random tiles window-by-window equals cutting them from a scene-wide mask
    built with the optional flag.
  - (auto) The 400 tile masks reassemble into the full 10 m mask exactly.
  - (auto) With `--save-tile-masks`, `tile_masks/tile_rRR_cCC.png` is written for all 400 tiles
    (549 × 549, values 0 and 255), each equal to the corresponding tile mask.
  - (auto) Two runs with different thresholds, written to two different `--out` folders, produce
    different `report.csv` files, and each folder passes CK1–CK8.
  - (manual) The resolution decision (D2) and the peak memory are recorded in the README.
  - (manual) On the contact sheet the detected cloud follows the visible cloud on tiles 11,2 / 6,5 /
    5,16 / 12,19.

### 3.5 The tile to write about

**Tile 12,19 changes verdict between two threshold sets that F1 cannot tell apart.** It is visibly
veiled in thin cirrus:

| Rule | Tile 12,19 | Verdict |
|---|---|---|
| `brightness > 0.33` (naive) | 0.0 % | kept |
| ESA `MSK_CLASSI` | 16.2 % | kept |
| `b > 0.20 AND ndsi > −0.20 OR B10 > 0.005` | 29.2 % | **kept** |
| `b > 0.16 AND ndsi > −0.20 OR B10 > 0.005` | 30.4 % | **discarded** |
| s2cloudless, defaults | 82.5 % | discarded |

The two tuned rules differ by 0.004 in F1 — inside the noise established in §3.2 — and they disagree
about whether this tile ships. That is the clearest single illustration of why this task has no
single right answer, and it is far more convincing than a pipeline presented as flawless. Write it
up, with this table.

---

## 4. Backend `s2cloudless` — pretrained model

**Role.** The ML bonus (R7), and the second candidate for the submission.

### 4.1 Model choice

`s2cloudless` is open source, is the model Sentinel Hub uses in production, takes Sentinel-2 **L1C**
bands directly, and returns a per-pixel **probability** rather than a hard edge.

**It is a LightGBM gradient-boosted classifier, not a deep neural network.** It satisfies "Machine
Learning" in the bonus, but not strictly "Deep Learning". Say so plainly in the README.

`s2cloudless` 1.7.3 installs on Python 3.11 and pulls `lightgbm`, `sentinelhub`, `pyproj`, `shapely`
and `opencv-python-headless` — which brings `pyproj` and `shapely`, needed by the other backends
anyway.

### 4.2 What was verified

From the installed source and by running it (Python 3.11.0, s2cloudless 1.7.3):

- **No download step.** The trained model ships inside the package
  (`pixel_s2_cloud_detector_lightGBM_v0.1.txt`, 11 MB). `pip install` is the download.
- **Band subset.** `MODEL_BAND_IDS = [0, 1, 3, 4, 7, 8, 9, 10, 11, 12]`, i.e. B01, B02, B04, B05,
  B08, B8A, B09, B10, B11, B12. With `all_bands=True` pass all 13 bands in order as `(N, H, W, 13)`;
  otherwise pass the 10-band subset. **`all_bands` defaults to `False`**, so passing 13 bands without
  setting it raises a `ValueError`.
- **Output.** A `float32` probability map in [0, 1] of shape `(N, H, W)`; `get_mask_from_prob`
  thresholds it.
- **Library defaults in this version are `threshold=0.4`, `average_over=1`, `dilation_size=1`.** The
  `average_over=4, dilation_size=2` pair that appears in the project's published examples is **not**
  the default, and the difference is large (§4.3). Whichever you use, set it explicitly and record it.
- **The offset matters, measured scene-wide.** With the offset the model flags 39.3 % of the scene;
  without it, **100.00 %** — every pixel of all 400 tiles. This is consistent with the model having
  been trained before baseline 04.00, when no offset existed; that reason is inference, not confirmed.
- **Clear sky stays clear.** On the genuinely clear tile 11,2 the model flags 0.45 %.
- **Runtime.** 13 bands at 60 m read in 2.8 s; the probability map takes about 14 s.

Not verified: what reflectance scale the library documents (its docstring is silent), which parameter
values Sentinel Hub recommends for a given ground resolution, and how the model was trained. Running
at 60 m is a choice made here for memory and runtime, not a documented requirement.

### 4.3 Parameters: the sweep and its grid

The library defaults flag 39.3 % of the scene against ESA's 21 % and discard **259** of 400 tiles.
That is the single biggest decision in this backend, so sweep it properly.

**Probability threshold**, 10-band stack, 60 m, no morphology, scored against the `esa` mask:

| threshold | scene cloud % | invalid tiles | F1 vs ESA | IoU |
|---|---|---|---|---|
| 0.3 | 36.8 | 240 | 0.66 | 0.49 |
| 0.4 | 32.5 | 205 | 0.69 | 0.52 |
| 0.5 | 29.8 | 181 | 0.70 | 0.54 |
| 0.6 | 27.6 | 156 | 0.71 | 0.55 |
| **0.7** | 25.6 | 145 | **0.71** | **0.55** |
| **0.8** | 23.5 | 117 | **0.71** | **0.55** |
| 0.9 | 20.6 | 91 | 0.70 | 0.54 |
| 0.95 | 17.6 | 61 | 0.68 | 0.51 |

**The agreement optimum is at 0.7–0.8.** A grid that stops at 0.6 has its best value on its own
boundary — exactly the mistake §3.2 warns about — so the sweep runs **0.3 to 0.95**.

**Morphology**, at threshold 0.4:

| `average_over` / `dilation_size` | scene cloud % | invalid tiles |
|---|---|---|
| none / none | 32.5 | 205 |
| 1 / 1 (library default) | 39.3 | 259 |
| 4 / 2 (published example) | 43.5 | 281 |
| 11 / 5 | 51.8 | 302 |

Every morphology setting only adds cloud, and the span from "none" to 4/2 is 11 points of scene
cloud — a bigger lever than the probability threshold. The grid must therefore include **none**, or
it cannot reach the region where the model and the reference are comparable.

### 4.4 What the extra flags are

The 19 % of the scene the model flags with defaults and ESA does not has median brightness 0.137
(clear ground 0.079, ESA cloud 0.354) and median B10 0.0019 (clear 0.0013, ESA cloud 0.0042). That is
intermediate: thin cloud, haze or cloud edges. On the crop around tile 12,19 — an extensive thin
cumulus deck — the model flags essentially all of it while ESA outlines only the denser patches. The
model also draws wide halos around small isolated clouds.

So **precision 0.51 against ESA does not mean half the model's flags are wrong.** Many are visible
cloud that ESA missed. Breaking that tie is what the visual audit (E2) is for.

### 4.5 A result to be ready for

Best F1 against ESA, both tuned as far as their sweeps allow:

- `s2cloudless`: **0.71** (threshold 0.7–0.8)
- `threshold`: **0.72**

On the only quantitative axis available, an 11 MB gradient-boosted model does not beat three band
comparisons. That is a legitimate and interesting finding, and E1 must be able to report it without
flinching. It is also why this backend is a backend and not a separate pipeline.

### 4.6 Tasks

- [ ] **B3.1 — Set up the model** *(R7)*
  **Done when:**
  - (auto) `S2PixelCloudDetector` loads from the installed package with no network access; the smoke
    test's `check_s2cloudless` passes.
  - (auto) A test shows that passing the wrong number of bands raises.
  - (auto) The band order of the stack equals the library's `S2_BANDS` order (an assertion).
  - (manual) The README states that `s2cloudless` is gradient-boosted trees (ML), not a neural
    network, and that this satisfies the bonus.

- [ ] **B3.2 — Run the model on the scene** *(R7)*
  **Done when:**
  - (auto) An offset-corrected reflectance stack is built on the 60 m grid with the block-mean kernel,
    and the probability map is saved as `cloud_probability_60m.npy` (`float32`, 1830 × 1830, [0, 1]).
  - (auto) With library defaults the run reproduces §4.3: scene cloud **39.3 % ± 0.5** and **259 ± 3**
    invalid tiles, in well under a minute.
  - (auto) The F10 offset regression test passes: without the offset the model flags at least 99 % of
    the scene.

- [ ] **B3.3 — Choose the parameters deliberately** *(R7, decision D6)*
  **Done when:**
  - (auto) `output/comparison/s2cloudless_sweep.csv` covers `threshold` **0.3–0.95**,
    `average_over` **{none, 1, 2, 4}** and `dilation_size` **{none, 1, 2}**, with scene cloud %, tiles
    above 30 %, agreement with the `esa` mask and agreement with the audit verdicts (E2).
  - (auto) The sweep asserts that its best agreement value does **not** sit on a grid boundary.
  - (auto) It reproduces the §4.3 tables within 0.02 F1 and ±3 tiles.
  - (manual) The chosen parameters and the reason are in the README, and were **not** tuned against
    ESA alone.

- [ ] **B3.4 — Full run** *(R2)*
  **Done when:**
  - (auto) `--detector s2cloudless` produces all five outputs and passes CK1–CK8.
  - (auto) The model mask is upsampled ×6 with nearest neighbour first, then cut into 549 × 549 tiles
    at 10 m; the mask holds only 0 and 1 (no interpolated probability was thresholded after
    upsampling).
  - (auto) The 400 tile masks reassemble into the full 10 m mask exactly.
  - (auto) `cloud_cover_percent` is the hard-mask fraction (decision D8); the **mean probability** per
    tile is written to `tile_stats.csv` so both definitions can be compared.
  - (manual) The README states which definition feeds the 30 % rule.
  - (manual) On the contact sheet the overlay looks right on tiles 11,2 / 6,5 / 5,16 / 12,19,
    including the halos the model draws around small clouds.

---

## 5. Evaluation and selection

### 5.1 Comparison protocol

The reference is always the `esa` mask, resampled to whatever grid is being compared. It is a
**baseline, not ground truth**; say so wherever these numbers appear.

**Pixel level:** precision, recall, F1, IoU. Report all four at two decimals; F1 alone hides the
precision/recall trade-off that the threshold sweep is all about.

**Tile level** (the one that matters for the brief): the count of tiles above 30 %, the number of
tiles whose `valid` flag disagrees, and the direction of disagreement. A method that drops tiles ESA
keeps behaves very differently from one that keeps tiles ESA drops.

**Named tiles**, across all three detectors:

| Tile | Character | ESA (exact 10 m tiles) |
|---|---|---|
| 11,2 | genuinely clear | 0.0000 % |
| 6,5 | moderate cloud | 40.9985 % |
| 5,16 | heavy cloud | 82.2927 % |
| 12,19 | thin cirrus veil, the tile that flips (§3.5) | 16.2471 % |

**The spread**, which is the headline result: invalid-tile counts across all three detectors and the
naive rule, plus the number of tiles in the 25–35 % band (61 under ESA).

- [ ] **E1 — Comparison module** (`pipeline/compare.py`)
  **Done when:**
  - (auto) Unit tests check precision, recall, F1 and IoU on tiny hand-made masks whose answers are
    known.
  - (auto) `python -m pipeline.compare --runs output/esa output/threshold output/s2cloudless`
    writes `output/comparison/comparison.md` containing every item of §5.1.
  - (auto) It works for any subset of runs, in any order.
  - (manual) The report states plainly that ESA is a baseline, not ground truth.

### 5.2 Visual audit

ESA is a baseline and §4.4 found places where it visibly under-flags, so the tie between `threshold`
and `s2cloudless` is broken by eye, on a small fixed sample, with the sampling rule stated in advance.

- [ ] **E2 — Visual audit** (`audit/verdicts.csv`)
  **Done when:**
  - (auto) The sample is the four named tiles plus the 12 tiles where the two candidate detectors
    disagree most in `cloud_cover_percent`, chosen by that stated rule and written out before you look
    at them.
  - (manual) For each of the 16 tiles you record your own verdict (`cloud`, `thin_cloud` or `clear`)
    in `audit/verdicts.csv`, viewing true colour with every detector's outline overlaid on the
    contact sheet.
  - (manual) Each detector's agreement with your verdicts is reported **next to** its agreement with
    ESA, with a plain statement that this is a small, subjective sample whose resolution is one tile
    in sixteen.

### 5.3 Choosing the detector that ships

The rule was fixed in §0.4, before any of this was measured. Apply it as written:

1. `esa` is excluded — it is a baseline, not detection.
2. The winner is whichever of `threshold` and `s2cloudless` agrees better with the E2 audit.
3. If they are within one tile of sixteen, `threshold` wins on simplicity.

- [ ] **E3 — Name the deliverable detector** *(decision D5)*
  **Done when:**
  - (manual) `output/comparison/decision.md` states which detector ships, which of the three rules
    decided it, and the audit and ESA numbers for both candidates.
  - (manual) It records that the rule was fixed before the evaluation ran, and does not invent a new
    criterion after seeing the results.
  - (manual) If `s2cloudless` is not the detector that ships, the README says explicitly that the ML
    bonus is satisfied by shipping and evaluating it as a backend.

---

## 6. Submission

- [ ] **S1 — README** *(R9)*
  **Done when:**
  - (manual) Install instructions, including the `PATH` gotcha of §0.5, and where to get the input
    product (the `.SAFE` folder is not in the repository).
  - (manual) The exact command for each of the three detectors, and for the validator and the contact
    sheets.
  - (manual) The results table for all three detectors, led by **the spread**: invalid-tile counts of
    107 / your threshold count / your s2cloudless count, and the 61 tiles sitting between 25 % and
    35 %.
  - (manual) Which detector ships and why (E3); that the ML bonus is satisfied by B3 either way.
  - (manual) The method: formulas, thresholds and how they were chosen; the offset and why it matters;
    the resolution decision; why NDSI is used on a snow-free scene; four-corner bounding boxes;
    row-major CSV order and the `.jgw` world files as the link from a JPEG to its place on the ground.
  - (manual) The tile 12,19 write-up from §3.5, with its table.
  - (manual) Limitations: no ground truth, ESA is a baseline, no shadow class (D4), one scene, a
    16-tile subjective audit.
  - (manual) Every open decision D1–D8 answered in a sentence.
  - (manual) A fresh clone plus venv reproduces the outputs by following it verbatim.

- [ ] **S2 — The single deliverable ZIP** *(R10)*
  **Done when:**
  - (auto) Exactly **one** ZIP exists, at most 50 MB, containing:
    - `tiles/` — the chosen detector's valid JPEGs and their `.jgw` files
    - `report.csv` — the chosen detector's report, 400 rows, the six columns of §1.5
    - `tile_stats.csv`, `cloud_mask.geojson`, `run_summary.json`
    - `comparison/` — the other two detectors' `report.csv` files, `comparison.md`, `decision.md`,
      `reference_notes.md`, the sweeps, the audit verdicts and the contact-sheet PNGs
    - `README.md`, `pipeline/`, `scripts/`, `tests/`, `config/`, `requirements.txt`
  - (auto) `python scripts/check_deliverables.py --out <unzipped copy>` passes the structural checks
    in a temporary folder, with no `.SAFE` present.
  - (manual) A colleague, or you on a clean machine, follows the README from the unzipped copy alone
    and gets the same numbers.

---

## 7. Suggested layout

```
pipeline/
  metadata.py      # XML parsing, constants                                   (F1)
  io.py            # band reading, reflectance, resampling                    (F1, F2)
  tiling.py        # 20x20 grid, four-corner boxes, tile stats, 30 % rule     (F3, F4)
  report.py        # report.csv + tile_stats.csv                              (F5)
  export.py        # JPEG + .jgw + GeoJSON                                    (F6, F7)
  checks.py        # CK1-CK6 as in-pipeline assertions                        (F9)
  detectors/
    esa.py         # MSK_CLASSI                                               (B1)
    threshold.py   # brightness / NDSI / B10                                  (B2)
    s2cloudless.py # model wrapper                                            (B3)
  compare.py       # precision, recall, F1, IoU, tile agreement, the spread   (E1)
  run.py           # CLI: --safe-dir --detector --out                         (F8)
scripts/
  smoke_test.py            # environment check (already exists)
  check_deliverables.py    # CK1-CK8, with a structural-only mode             (F9)
  contact_sheet.py         # the three PNGs that replace a viewer             (F9)
tests/                     # geocoding, 30 % rule, offset regression          (F10)
audit/verdicts.csv         # your hand verdicts                               (E2)
config/thresholds.json     # your chosen T_bright, T_ndsi, T_cirrus           (B2.2)
SPEC.md  README.md  requirements.txt
```

Tests use small synthetic arrays or a tiny crop, never a committed band file.

---

## 8. Open decisions

Each needs a choice and one sentence of justification in the README.

- [ ] **D1 — Threshold strategy:** chosen by you by inspection on the contact sheet (B2.2). ESA's
  agreement optimum is reported for reference only and is never the target.
- [ ] **D2 — Detection resolution:** native 10 m or everything at 60 m? Cite the agreement figures in
  §3.3. Not a speed question.
- [ ] **D3 — The 30 % boundary:** never round before comparing. On the 10 m grid no tile can be
  exactly 30 %, so `<=` and `<` are identical on unrounded values (§1.4). Record it.
- [ ] **D4 — Cloud shadow: the default answer is to ignore it, and to say why in one sentence.** The
  CSV the brief specifies has one cloud column and no shadow column, so a shadow class has nowhere to
  go, and detecting shadow is a second threshold-choice problem as hard as the first. A check on this
  scene shows why: shifting the cloud mask by the solar geometry (zenith 53.52°, azimuth 168.40°, so a
  displacement of 1.354 × cloud height toward 348.4°) gives candidate shadow regions whose mean
  brightness is 0.102–0.113 for assumed cloud bases of 1000–3000 m, against 0.110 for clear land —
  not measurably darker. A direct dark-pixel proxy is no better behaved: non-cloud, non-water pixels
  below brightness 0.06 are 2.70 % of the scene, below 0.07 are 16.8 %, below 0.08 are 30.4 %.
  Detecting shadow anyway is defensible if you want it; silence is not.
- [ ] **D5 — Which detector ships:** decided by the rule fixed in §0.4 and applied in E3.
- [ ] **D6 — s2cloudless parameters:** `threshold`, `average_over` and `dilation_size` move the result
  a lot (§4.3). Sweep the full grid including "none", choose deliberately, and tune against the audit,
  not against ESA alone.
- [ ] **D7 — How to treat thin cloud.** For tile 12,19 the methods give 0 %, 16 %, 29 %, 30 % and
  83 %, and the verdict flips between two threshold sets that F1 cannot separate (§3.5). Decide and
  state the definition.
- [ ] **D8 — What `cloud_cover_percent` means for the model:** the hard-mask fraction or the mean
  probability. Default is the hard-mask fraction in `report.csv`, with the mean probability in
  `tile_stats.csv`.

---

## 9. Provenance

**Verified against the product in this repository:** every §0.2 constant; the `MSK_CLASSI` channel
order and percentages; all four scene corners against `EXT_POS_LIST` (5 × 10⁻¹⁰°); the exact 10 m
tile statistics of the ESA mask (107 invalid, 293 valid, 20.9537 % mean, 49 clear tiles, 61 tiles in
the 25–35 % band, closest tile to the cut 29.8307 %); the four named tiles to 4 decimals; JPEG size
and decode error on 24 random valid tiles; every threshold sweep and F1 in §3.2; the resolution
comparison in §3.3; the s2cloudless threshold and morphology sweeps in §4.3; the scene-wide offset
behaviour (100.00 % without, 39.3 % with); the read timings and array sizes in §0.3; the GeoJSON
polygon counts and file sizes; and the no-data count in B02.

Also verified: the interpreters available via `py`; that `s2cloudless` + `rasterio` resolve on Python
3.8, 3.11 and 3.14; and, by actually installing and running, that the pinned `requirements.txt` builds
a working Python 3.11 environment (`pip check` clean) which passes all 10 checks of
`scripts/smoke_test.py` on the real product.

**Corrected during planning, so nobody repeats it:**

- The two-corner tile bounding box was 130 m short in latitude (§1.3).
- "Tile 6,5 sits exactly on 30.0 %" was a rounding artefact of 29.969 %; no tile can be exactly 30 %
  (§1.4).
- ESA's invalid-tile count is **107** on exact 10 m tiles, not 108 (§0.3).
- 549 is not the largest square that divides the scene (§0.2).
- An earlier threshold sweep had its optimum on the edge of its own grid; the same error was present
  in the s2cloudless parameter grid, whose agreement optimum sits at threshold 0.7–0.8 and not inside
  0.3–0.6 (§4.3).
- F1 values were quoted to three decimals, but swapping the downsampling kernel moves them by up to
  0.005 — half the ablation's own effect size. They are now quoted and tested to two (§1.2, §3.2).
- The threshold backend's "sanity bands" on scene cloud and invalid-tile count were ESA-derived, and
  would have failed an honest by-eye threshold choice. They are now orientation figures, not criteria
  (§3.4).
- "Shadow covers roughly as much ground as cloud in this scene" was asserted without measurement and
  does not survive a check (§8, D4).
- §3.3 justified windowed reads by comparing rasterio on one window against OpenCV on a whole band —
  two variables at once. rasterio reads a full band in 1.9 s; the real constraint is memory, not
  speed (§0.3, §1.8).
- The B10 branch was described as imprecise "in a thin ring at cloud edges"; at 60 m it is blocky in
  6 × 6 squares throughout (§3.3).

**Not verified:** the reflectance scale s2cloudless documents; the parameter values Sentinel Hub
recommends for a given ground resolution; how the model was trained; any deep-learning model other
than `s2cloudless`; and any result on an interpreter other than 3.11. The threshold and resolution
measurements in §3 were run on Python 3.8 with numpy 1.x and were not re-run on 3.11; they are
expected to reproduce. The visual judgements in §4.4 are single crops judged by eye.
