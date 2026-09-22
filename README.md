# sentinel2-cloud-filtering

Hi FlyPix AI team,

Thank you for giving me this task. I have to admit that it took me more than the estimated five hours, as I got a little lost in exploring the topic of multispectral data. However, I found it really interesting and learned a lot in the process!

I would like to briefly explain how I approached the task.

First, I used Claude Code (Opus 5) to generate a tutorial based on the provided dataset to help me understand multispectral data in general. You can find the resulting artifact here:

https://claude.ai/artifact/8zD4YCfyGrj9a4QPSR5dJo

I learned two particularly important things:

1. The provided dataset already contains everything needed to solve the task.

In MTD_MSIL1C.xml, I found the average cloud coverage of 20.9537% (and a snow coverage of 0.0%, which makes sense since the data was recorded in October last year and there was no snow in Bavaria at that time).

Furthermore, I found QI_DATA/MSK_CLASSI_B00.jp2, which contains a per-pixel classification mask at 60 m resolution for opaque clouds, cirrus, and snow. I built a small viewer to inspect this mask (output/comparison/esa_mask_viewer.html).

Although I could have solved the task using the provided classification mask, I also wanted to try creating such a mask myself. My initial intuition was simply to calculate the brightness and classify pixels as cloud or non-cloud based on their brightness. This led me to the second important thing I learned:

2. Brightness alone is not sufficient to identify clouds.

There can also be snow (although there is none in this particular example) or bright soil. This is where the Normalized Difference Snow Index (NDSI) can help, using the shortwave infrared band (B11). At this wavelength, soil continues to reflect radiation, whereas cloud ice/water particles absorb it.

Another challenge is cirrus, which is more or less invisible in most bands but can be detected using B10, the cirrus band. This band is essentially insensitive to the Earth's surface, so anything it detects is likely to be high in the atmosphere, such as cirrus clouds.

Taking all of this into account, I used the following basic classification approach:

cloud = (brightness > T_bright AND NDSI > T_ndsi) OR (B10 > T_cirrus)

This brings me to the biggest problem I encountered: How should the thresholds be chosen?

I built a viewer to inspect the impact of changing these thresholds (output/comparison/threshold_viewer.html). However, the problem is that the human eye cannot reliably detect cirrus clouds or necessarily distinguish clouds from bright soil in the original imagery.

Therefore, I believe that determining optimal thresholds—or training a neural network to detect cloud pixels—would require some form of additional ground-truth data beyond the image data itself. I would be very interested in discussing this aspect with you in more detail!

As a bonus, you mentioned that a Machine Learning or Deep Learning approach could be used. I therefore looked for existing models for cloud detection and found s2cloudless, an open-source LightGBM gradient-boosted classifier. The model takes Sentinel-2 L1C bands directly as input and returns a per-pixel cloud probability.

However, the same issue applies here: I still need to determine an appropriate threshold for converting the predicted probabilities into a binary cloud mask.

In the end, I decided to solve the task using all three approaches:

- Use the ESA-provided classification mask from the dataset
- Build my own rule-based classifier based on brightness, NDSI, and the cirrus band
- Use the pretrained s2cloudless model for cloud classification

This allowed me not only to complete the task, but also to explore the underlying multispectral data and compare different approaches to the problem.

---

## Technical README

### Install

Use Python 3.11 in a virtual environment — the `py` launcher on Windows also offers 3.8, 3.10 and
3.14, and a bare venv silently inherits whichever interpreter created it (3.14 by default here), so
the `-3.11` flag matters:

```bash
py -3.11 -m venv .venv
.venv\Scripts\activate           # Windows (PowerShell / cmd)
source .venv/Scripts/activate    # Git Bash
pip install -r requirements.txt
```

**Known local gotcha:** `pip install` can fail with `OSError: [WinError 53]` (network path not
found). The cause isn't this project — a stale entry on the *user* `PATH` points at an offline
network drive and breaks pip during dependency resolution. Fix it permanently by removing or
reconnecting that PATH entry, or drop it for one session without touching the stored PATH:

```powershell
$env:PATH = (($env:PATH -split ';') | Where-Object { $_ -and $_ -notlike 'Z:*' }) -join ';'
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Install `opencv-python-headless`, not `opencv-python` — both provide the `cv2` module and shadow
each other, and `s2cloudless` requires the headless build (already pinned in `requirements.txt`).

**The input product is not in this repository.** The 801 MB `.SAFE` folder
(`S2C_MSIL1C_20251002T101851_N0511_R065_T32UPV_20251002T143120.SAFE`) exceeds GitHub's 100 MB
per-file limit on five of its band files and is excluded via `.gitignore`. Every command below takes
its location as `--safe-dir <path to the .SAFE folder>`.

Verify the environment against the real product before anything else:

```bash
python scripts/smoke_test.py --safe-dir <path to the .SAFE folder>
```

### Running the pipeline

Each of the three detectors is a separate `--detector` value of the same CLI:

```bash
python -m pipeline.run --safe-dir <path> --detector esa          --out output/esa
python -m pipeline.run --safe-dir <path> --detector threshold    --out output/threshold
python -m pipeline.run --safe-dir <path> --detector s2cloudless  --out output/s2cloudless
```

Each run writes `tiles/*.jpg` (+ `.jgw`/`.prj`), `report.csv`, `tile_stats.csv`, `cloud_mask.geojson`
and `run_summary.json` into its `--out` folder.

Validate a run's structure and content:

```bash
python scripts/check_deliverables.py --out output/threshold --safe-dir <path>
```

Build the contact sheets used to inspect results without an interactive viewer:

```bash
python scripts/contact_sheet.py --safe-dir <path> --runs output/esa output/threshold output/s2cloudless --tiles
python scripts/contact_sheet.py --safe-dir <path> --runs output/esa --scene
python scripts/contact_sheet.py --safe-dir <path> --thresholds
```

Compare all three detectors and re-run the audit/decision:

```bash
python -m pipeline.compare --runs output/esa output/threshold output/s2cloudless --audit-verdicts audit/verdicts.csv
python scripts/decide.py --runs output/esa output/threshold output/s2cloudless --audit-verdicts audit/verdicts.csv
```

Two exception viewers exist beyond the spec's own no-interactive-viewer scope decision, built on
request to manually sanity-check the two backends that don't ship an official mask: `pipeline/view_mask.py`
(a static ESA-mask viewer) and `pipeline/view_thresholds.py` (a live-slider threshold viewer).

### Results — the spread

The headline number for each method is how many of the 400 tiles it discards (cloud > 30 %), and the
resulting scene-wide cloud percentage:

| Method | Invalid tiles | Scene cloud % |
|---|---|---|
| esa (baseline, `MSK_CLASSI`) | 107 | 20.9537 |
| **threshold (ships)** | **83** | **20.1517** |
| s2cloudless | 161 | 27.7646 |
| `brightness > 0.33` (naive, reference only) | 35 | 13.6 |

61 tiles sit between 25 % and 35 % cloud under ESA — close enough to the 30 % cut that small,
defensible changes in method move them across it. That's the reason a single tile (12,19, below) is
worth a full write-up rather than a footnote.

### Pixel- and tile-level agreement with ESA

| Detector | Precision | Recall | F1 | IoU |
|---|---|---|---|---|
| threshold | 0.73 | 0.70 | 0.71 | 0.55 |
| s2cloudless | 0.62 | 0.82 | 0.71 | 0.55 |

| Detector | Invalid tiles | Disagreements with ESA | ESA keeps, detector discards | ESA discards, detector keeps |
|---|---|---|---|---|
| esa (reference) | 107 | — | — | — |
| threshold | 83 | 30 | 3 | 27 |
| s2cloudless | 161 | 56 | 55 | 1 |

`threshold` leans lenient against ESA (it keeps 27 tiles ESA discards, against only 3 the other way).
`s2cloudless` leans strict (it discards 55 tiles ESA keeps, and keeps only 1 ESA discards). Full
detail: `output/comparison/comparison.md`.

**ESA's `MSK_CLASSI` mask is a baseline, not ground truth.** It is coarse (60 m), binary, has no
shadow class, and can miss thin cloud that is visible in the imagery. Maximising agreement with it
copies its errors and turns a detector into an imitation of ESA instead of the independent detection
the brief asks for — this caution applies to every "agreement with ESA" number in this README.

### Visual audit (E2) and which detector ships (E3, D5)

A 16-tile sample (the 4 named tiles below, plus the 12 tiles where `threshold` and `s2cloudless`
disagree most on `cloud_cover_percent`) was verdicted by eye against the true-colour imagery —
**not** against any detector's output — recorded in `audit/verdicts.csv`. One tile came back
`thin_cloud` and is excluded from scoring (decision D7 is open on how to treat it; scoring against an
undefined right answer would just hide that decision).

| Detector | Agreement with ESA (400 tiles) | Agreement with audit (15 scored) |
|---|---|---|
| esa (reference) | 100.0 % | 100.0 % |
| threshold | 92.5 % | 86.7 % (13/15) |
| s2cloudless | 86.0 % | 80.0 % (12/15) |

**The rule for which detector ships was fixed in advance, before any of this was measured** (this
README's own Decision D5 section repeats it): `esa` is excluded as a baseline, not a detector; the
winner is whichever of `threshold`/`s2cloudless` agrees better with the audit; a gap within 1 tile of
15 scored is a tie, and `threshold` wins ties on simplicity.

**Result: `threshold` ships.** The gap (13/15 vs. 12/15) is within the 1-tile tie margin, so
simplicity decided it — not raw ESA-agreement, where threshold also happens to lead (92.5 % vs.
86.0 %, reported but not decisive per the rule). `s2cloudless` is fully implemented, deliberately
tuned (see D6 below) and evaluated on equal footing, so the ML/DL bonus is satisfied regardless of
which detector ships. Full detail: `output/comparison/decision.md`.

**A genuine judgment call worth being explicit about:** three of the 16 audit tiles (11,17 / 10,16 /
11,19) show real cloud plus a lot of cloud shadow. Read strictly — counting only the bright cloud
itself, not its shadow, since the task asks for cloud coverage — the cloud alone covers well under a
third of each tile, so they're verdicted `clear`. A shadow-inclusive reading would call all three
`cloud` instead, which would reverse this section's outcome in favour of `s2cloudless`. This
sensitivity is documented per-tile in `audit/verdicts.csv`'s notes column rather than resolved
silently.

### Named tiles

| Tile | Character | esa | threshold | s2cloudless |
|---|---|---|---|---|
| 11,2 | genuinely clear | 0.0000 % | 0.0932 % | 0.1971 % |
| 6,5 | moderate cloud | 40.9985 % | 36.5709 % | 43.3814 % |
| 5,16 | heavy cloud | 82.2927 % | 74.7473 % | 79.7307 % |
| 12,19 | thin cirrus veil (see below) | 16.2471 % | 29.7613 % | 51.0884 % |

### The method

**Brightness alone is not enough** — bright soil and snow both trip a brightness-only rule (this
scene has no snow, recorded in the product's own metadata at 0.0 %, but bare soil is present). Two
more band tests fix that:

- **NDSI** (Normalized Difference Snow Index, using B11 shortwave infrared): soil keeps reflecting at
  this wavelength while cloud ice/water particles absorb it, so NDSI separates "bright and reflective
  in SWIR" (soil) from "bright and absorptive in SWIR" (cloud), even though there's no snow on this
  scene to actually distinguish from cloud — it works here as a general brightness veto.
- **B10 (cirrus band)** is essentially insensitive to the Earth's surface, so anything it detects is
  high in the atmosphere — this is what catches thin cirrus that's nearly invisible in the other
  bands.

The shipped rule (`pipeline/detectors/threshold.py`):

```
cloud = (brightness > T_bright AND NDSI > T_ndsi) OR (B10 > T_cirrus)
```

**Radiometric offset.** This product uses processing baseline ≥ 04.00, which subtracts a fixed
`RADIO_ADD_OFFSET = -1000` before scaling by `QUANTIFICATION_VALUE = 10000`. Skipping that offset
correction shifts every reflectance value and silently biases both the brightness and NDSI tests —
it's applied once, centrally, in `pipeline/io.py`'s band-reading path, not per-detector.

**Decision D2 — detection resolution: native per-band resolution, combined at 10 m.** Brightness uses
10 m bands natively; NDSI comes from 20 m B11 (imprecise in roughly a 20 m-wide ring at cloud edges);
B10 is 60 m (blocky in 6×6 10 m blocks across its whole extent when upsampled, not just at edges).
Running everything at 60 m instead is also defensible — simpler, one shared grid — but was not the
choice made here; either way, this is not a speed question (a full 10 m band reads in 1.9 s, a single
tile window in 0.013 s — the real constraint is memory, and streaming tiles removes it).

**Four-corner bounding boxes.** Each tile's `min_latitude,min_longitude,max_latitude,max_longitude`
comes from reprojecting all four UTM corners of the tile, not just two — the scene is rotated enough
in lat/lon space that a two-corner box would be wrong.

**Row-major CSV order, no identity columns.** `report.csv`'s 400 data rows are row-major in
`(row, col)`: row *i* (0-indexed, after the header) is tile `(i // 20, i % 20)`. The CSV itself has no
tile-id column — row order, the JPEG filenames (`tile_rRR_cCC.jpg`) and their `.jgw`/`.prj` sidecars
are the only link between a CSV row and a place on the ground. A `.jgw` world file alone has no CRS
information — GIS tools will happily misread its UTM-metre pixel size as degrees without the paired
`.prj` file, which is why both are written together.

**The 30 % rule.** `valid = (cloud_pct <= 30) AND (nodata_fraction <= 0.5)`, compared on the unrounded
value — on the 10 m grid no tile is ever exactly 30.0000 %, so `<=` and `<` give identical results in
practice, but the comparison is still written unrounded to avoid a boundary bug (Decision D3). No-data:
every tile on this product has `nodata_fraction` below 1e-4, so the no-data rule changes zero verdicts
here — it exists for robustness on other products, not because this scene needed it.

### The threshold choice (`config/thresholds.json`, Decision D1)

Chosen by visual inspection on the four-tile contact sheet (`output/comparison/thresholds.png`), not
by fitting to the ESA mask:

| Parameter | Chosen | ESA-agreement optimum (reference only) |
|---|---|---|
| `T_bright` | 0.18 | 0.16 |
| `T_ndsi` | −0.20 | −0.20 |
| `T_cirrus` | 0.005 | 0.005 |

0.18 is the midpoint of the recommended 0.16–0.20 range: at 0.20 visible haze on tile 12,19 is still
outside the outline, while at 0.14 the genuinely clear named tile (11,2) starts picking up false
positives (flagged fraction rises from ~0.1–0.2 % at 0.16–0.20 to 0.3 % at 0.14, on ground with no
cloud at all) — 0.18 avoids that creep while still catching more of the veil than 0.20 alone.
`T_ndsi = −0.20` vetoes vegetation (NDSI −0.238, correctly excluded) while still allowing bright bare
soil (NDSI −0.130) to be called cloud if it's also bright enough. `T_cirrus = 0.005` sits above the
clear-sky B10 floor (mean 0.0014 on tile 11,2) but below individual cirrus pixels within the hazy
tile 12,19, even though that tile's own B10 *mean* (0.0041) sits just under it — the threshold is
per-pixel.

**Sensitivity.** Moving `T_bright` by 0.02 either side of 0.18 (to 0.20 or 0.16) changes tile 12,19's
own cloud percentage from 29.26 % to 30.59 % — spanning the 30 % cut and flipping that one tile's
verdict; see the tile 12,19 write-up below. This is the clearest evidence in the whole project of how
close a "reasonable" threshold choice can sit to the boundary.

**Why not maximise ESA-agreement:** ESA's mask is the output of an algorithm, not ground truth. It's
coarse, binary, has no shadow class, and under-detects thin cloud (it scores tile 12,19 at only
16.2 %, against 29–30 % from the threshold rule and 51–82 % from s2cloudless). Fitting thresholds to
match it would copy those blind spots and turn this backend into an imitation of ESA rather than the
independent detection the brief asks for — if ESA's mask were ground truth, there'd be nothing left
to detect.

### The s2cloudless choice (`config/s2cloudless.json`, Decision D6)

`s2cloudless` is gradient-boosted trees (LightGBM), not a neural network — this still satisfies the
ML/DL bonus regardless of whether it ships (Decision D5, above).

| Parameter | Chosen | Library default | ESA-agreement optimum (reference only) |
|---|---|---|---|
| `prob_threshold` | 0.6 | 0.4 | 0.7–0.8 (F1 0.768) |
| `average_over` | off | 1 | 4 |
| `dilation_size` | off | 1 | 2 |

Chosen via the ESA-agreement sweep (`scripts/s2cloudless_sweep.py`) for orientation, plus a visual
check on the four named tiles — **not** by fitting to ESA alone. The library default (0.4) discards
65 % of the scene (259/400 tiles); the ESA-agreement optimum (0.8) is explicitly rejected: at that
setting, coverage of the visibly-veiled tile 12,19 drops from 82.5 % (default) to 33.1 %, converging
toward ESA's own under-detection of thin cloud — exactly the failure mode this backend exists to
correct, not reproduce. At 0.6 the tile still scores 51.1 % (more than 3× ESA's figure) while the
scene-wide discard rate (40 %, 160/400) is far more defensible than the default's 65 %. Both
morphology options are left off: averaging pre-smooths the raw probability before thresholding, and
dilation only inflates already-flagged regions — both work against seeing the model's own raw
judgement.

**Caveat on this tuning:** Decision D6 asks these parameters to be tuned against the visual audit
(E2), not against ESA alone. They were originally chosen before E2 existed, using an informal check
on the four named tiles instead of the full 16-tile audit — and were not revisited after E2 landed,
since D5 had already settled that `s2cloudless` doesn't ship regardless of this choice. If
`s2cloudless` were ever chosen to ship, this tuning should be redone against `audit/verdicts.csv`
directly.

`cloud_cover_percent` for this backend is the hard-mask fraction (Decision D8) — the same definition
as every other backend's `report.csv` column — with the mean per-tile cloud *probability* available
separately in `tile_stats.csv` for anyone who wants the continuous value instead.

### Tile 12,19 — the tile that flips

Visibly veiled in thin cirrus. Measured on the real product, natively at 10 m:

| Rule | Tile 12,19 | Verdict |
|---|---|---|
| `brightness > 0.33` (naive) | 0.0551 % | kept |
| ESA `MSK_CLASSI` | 16.2471 % | kept |
| `T_bright = 0.20` | 29.2613 % | **kept** |
| **shipped config** (`T_bright = 0.18`) | **29.7613 %** | **kept** |
| `T_bright = 0.16` | 30.5898 % | **discarded** |
| s2cloudless, library defaults | 82.4719 % | discarded |
| s2cloudless, shipped config | 51.0884 % | discarded |

Three threshold settings just 0.02 apart in `T_bright` span 29.26 %–30.59 % on this single tile, and
the lowest flips its verdict — that gap is inside the F1 noise floor established during threshold
tuning (§3.2 of SPEC.md), so none of the three is more "correct" by that measure. Meanwhile
`s2cloudless` discards this tile outright at either of its own settings, by a wide margin either way.
Four detectors, five configurations, three different verdicts, on one tile — this is the clearest
single illustration in the whole project of why this task has no single right answer.

### Limitations

- **No ground truth.** ESA's mask is a baseline algorithm output, not a measurement of truth; the
  16-tile visual audit is one person's subjective judgement, not an independent ground truth either.
- **Cloud shadow is ignored** (Decision D4). The CSV schema the brief specifies has one cloud column
  and no shadow column, so a shadow class has nowhere to go, and detecting shadow reliably is its own
  threshold-choice problem, as hard as the cloud one. A check on this scene shows why a simple
  approach doesn't work: shifting the cloud mask by the sun geometry (zenith 53.52°, azimuth 168.40°)
  toward 348.4° gives candidate shadow regions with mean brightness 0.102–0.113 for assumed cloud
  bases of 1000–3000 m, against 0.110 for clear land — not measurably darker. A direct dark-pixel
  proxy fares no better: non-cloud, non-water pixels below brightness 0.06 are 2.70 % of the scene,
  below 0.07 are 16.8 %, below 0.08 are 30.4 % — no clean cutoff exists. Detecting shadow anyway would
  be defensible; staying silent about the decision would not.
- **One scene.** Every threshold, sweep result and audit sample is specific to this one product; none
  of it is validated to generalise to other scenes, seasons or tiles of Sentinel-2 imagery.
- **A 16-tile subjective audit** out of 400 tiles (4 %) is a small, spot-check sample, not a
  statistically powered evaluation — read the audit agreement gap between `threshold` and
  `s2cloudless` (86.7 % vs. 80.0 %) as suggestive of which detector is better, not as a precise,
  decisive measurement.

### Open decisions (D1–D8)

- **D1 — Threshold strategy:** chosen by visual inspection on the contact sheet, not by fitting to
  ESA (`config/thresholds.json`; the ESA-agreement optimum is reported only as a reference point).
- **D2 — Detection resolution:** each band test runs at its own native resolution, combined at 10 m,
  because B10 upsampled from 60 m is blocky across its whole extent, not just at edges.
- **D3 — The 30 % boundary:** compared on the unrounded cloud percentage, so `<=` and `<` are
  equivalent in practice, but written unrounded on principle to avoid a boundary bug.
- **D4 — Cloud shadow:** ignored, because the brief's CSV schema has no shadow column and a naive
  shadow proxy on this scene (solar-geometry shift or a dark-pixel cutoff) isn't reliably darker than
  clear land — see Limitations above.
- **D5 — Which detector ships:** `threshold` — 13/15 scored audit tiles against `s2cloudless`'s 12/15,
  within the pre-fixed 1-tile tie margin, decided by simplicity (`output/comparison/decision.md`).
- **D6 — s2cloudless parameters:** `prob_threshold = 0.6`, no morphology (`config/s2cloudless.json`),
  chosen by the ESA-agreement sweep for orientation plus a visual check on the named tiles; the
  ESA-agreement optimum was deliberately rejected because it suppresses most of the thin cloud on tile
  12,19 that this backend exists to catch.
- **D7 — How to treat thin cloud:** treated as its own category, `thin_cloud`, in the visual audit,
  and excluded from strict pass/fail scoring rather than forced into `cloud`/`clear` — for tile 12,19
  the five configurations above give three different verdicts, so there is no single defensible answer
  to score against.
- **D8 — What `cloud_cover_percent` means for the model:** the hard-mask fraction, same definition as
  every other backend, in `report.csv`; the mean per-tile probability is available separately in
  `tile_stats.csv`.

### Reproducing this from a fresh clone

```bash
git clone <this repository>
cd sentinel2-cloud-filtering
py -3.11 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python scripts/smoke_test.py --safe-dir <path to the .SAFE folder>
python -m pipeline.run --safe-dir <path> --detector esa         --out output/esa
python -m pipeline.run --safe-dir <path> --detector threshold   --out output/threshold
python -m pipeline.run --safe-dir <path> --detector s2cloudless --out output/s2cloudless
python -m pipeline.compare --runs output/esa output/threshold output/s2cloudless --audit-verdicts audit/verdicts.csv
python scripts/decide.py --runs output/esa output/threshold output/s2cloudless --audit-verdicts audit/verdicts.csv
```

This reproduces every number in this README from the source product and the checked-in
`audit/verdicts.csv` (the audit's own hand judgements aren't recomputed, only scored).
