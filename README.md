# sentinel2-cloud-filtering

Repository: https://github.com/JohannesErfurt/sentinel2-cloud-filtering

## How I approached the task

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

The ZIP file contains the results of my own rule-based classifier (the threshold backend). I chose the thresholds T_bright = 0.18, T_ndsi = −0.20 and T_cirrus = 0.005 by looking at four characteristic tiles (clear, moderate cloud, heavy cloud and a thin cirrus veil) under several settings (comparison/thresholds.jpg). With T_bright = 0.18 the classifier still catches most of the visible haze on the veiled tile, while lower values start flagging villages and roads on the clear tile as cloud. T_ndsi = −0.20 keeps vegetation from being classified as cloud, and T_cirrus = 0.005 lies just above the clear-sky level of the cirrus band. I deliberately did not tune the thresholds to match the ESA mask, as I am not sure if this can be treated as ground truth. 

---

## Technical README

**How Claude Code was used.** I used Claude Code to critique and rewrite the plan in `SPEC.md` and
then to implement it task by task: the pipeline, the three detector backends, the tests, the
comparison, the viewers, this technical README and the ZIP build. I set the direction and made the
decisions (which approaches to build, how to read the brief), and Claude Code checked each task
against the real product before it was ticked off. 

### Install

Use Python 3.11 in a virtual environment 

```bash
py -3.11 -m venv .venv
.venv\Scripts\activate           # Windows (PowerShell / cmd)
source .venv/Scripts/activate    # Git Bash
pip install -r requirements.txt
```

**The input product (the data) is not in this repository.** 

Verify the environment against the real product before anything else:

```bash
python scripts/smoke_test.py --safe-dir <path to the .SAFE folder>
```

This runs 10 checks against the .SAFE dataset. If all pass, installation worked and the pipeline commands will run. 

### Running the pipeline

Each of the three detectors is a separate `--detector` value of the same CLI:

```bash
python -m pipeline.run --safe-dir <path> --detector esa          --out output/esa
python -m pipeline.run --safe-dir <path> --detector threshold    --out output/threshold
python -m pipeline.run --safe-dir <path> --detector s2cloudless  --out output/s2cloudless
```

Each run writes `tiles/` (one JPEG per valid tile, nothing else), `report.csv`,
`cloud_mask.geojson` and `run_summary.json` into its `--out` folder.

Build the contact sheets used to inspect results without an interactive viewer (each run needs
`--save-tile-masks` for `--tiles` and `--scene`):

```bash
python scripts/contact_sheet.py --safe-dir <path> --runs output/esa output/threshold output/s2cloudless --tiles
python scripts/contact_sheet.py --safe-dir <path> --runs output/threshold --scene
python scripts/contact_sheet.py --safe-dir <path> --thresholds
```

They are written to `output/comparison/` as PNG. The ZIP carries the same three images as JPEG, in
`comparison/`:

| File in the ZIP | What it shows |
|---|---|
| `comparison/scene.jpg` | The whole scene with the shipped `threshold` mask in red (T_bright 0.18, T_ndsi −0.20, T_cirrus 0.005) and the 83 discarded tiles shaded — the tiles missing from `tiles/`. |
| `comparison/tiles.jpg` | The four named tiles with all three detectors' mask outlines (ESA red, threshold cyan, s2cloudless yellow). |
| `comparison/thresholds.jpg` | The four named tiles under five threshold settings for the `threshold` detector, with the shipped row (0.18) marked **SHIPPED**, plus each tile's cloud %. |


### Viewers for inspecting the masks

Three viewers show a cloud mask over the true-colour scene, with the 20×20 tile grid, zoom and pan,
and each tile's cloud % and verdict on hover. Each command writes one self-contained HTML file; open
it in any browser (no server or internet connection needed).

**ESA mask** — ESA's own classification mask, with its opaque-cloud, cirrus and snow layers:

```bash
python -m pipeline.view_mask --safe-dir <path> --run output/esa --out output/comparison/esa_mask_viewer.html
```

**Threshold masks** — three sliders for `T_bright`, `T_ndsi` and `T_cirrus`; the mask, tile verdicts
and invalid-tile count update as you drag. It opens at the values in `config/thresholds.json`:

```bash
python -m pipeline.view_thresholds --safe-dir <path> --out output/comparison/threshold_viewer.html
```

**s2cloudless masks** — one slider for the probability threshold, plus two curves showing how the
invalid-tile count and scene cloud % change with it. It opens at the value in
`config/s2cloudless.json`:

```bash
python -m pipeline.view_s2cloudless --safe-dir <path> --run output/s2cloudless --out output/comparison/s2cloudless_viewer.html
```

`--run` is optional for the ESA and s2cloudless viewers. When given, the viewer checks its per-tile
numbers against that run's `report.csv`, and the s2cloudless viewer reuses the run's saved
probability map instead of running the model again. The threshold viewer stores its values at 8 bits,
so its cloud % can differ from a real pipeline run by a few tenths of a point; the other two match
the pipeline exactly.

### Results — the spread

The headline number for each method is how many of the 400 tiles it discards (cloud > 30 %), and the
resulting scene-wide cloud percentage:

| Method | Invalid tiles | Scene cloud % |
|---|---|---|
| esa (baseline, `MSK_CLASSI`) | 107 | 20.9537 |
| **threshold (ships)** | **83** | **20.1517** |
| s2cloudless | 161 | 27.7646 |

`comparison/scene.jpg` shows where the shipped detector's 83 discarded tiles are.

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

**Detection resolution: native per-band resolution, combined at 10 m.** Brightness uses
10 m bands natively; NDSI comes from 20 m B11 and was upsampled to 10 m; 
B10 is 60 m and was upsampled to 10 m aswell. For the upsampling nearest-neighbour was used (plain pixel replication).

**The tiles are RGB, the detection is not.** Cloud detection uses the spectral bands (B11 and B10 for
the threshold rule, all 13 for s2cloudless), but the saved tiles are cut from ESA's 8-bit true-colour
image (`TCI.jp2`, B04/B03/B02), because a JPEG can hold only three 8-bit channels. 

