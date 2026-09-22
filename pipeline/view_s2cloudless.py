"""Self-contained HTML viewer with a **live** probability-threshold slider for
the ``s2cloudless`` backend.

    python -m pipeline.view_s2cloudless --safe-dir <SAFE> --run output/s2cloudless \
        --out output/comparison/s2cloudless_viewer.html

One HTML file: no server, no CDN, no network. The page ships the model's 60 m
cloud-probability map and recomputes ``cloud = probability > threshold`` in the
browser on every slider move, together with each tile's cloud percentage, the
number of invalid tiles, and two curves showing how both move with the
threshold.

The probability is packed into the three 8-bit PNG channels as 24-bit fixed
point (``round(p * 2**24)``). float32 probabilities in [0.5, 1) are exact
multiples of 2**-24, so unlike the threshold viewer this one is not an
approximation: per-tile percentages match
``pipeline.run --detector s2cloudless`` exactly whenever averaging and dilation
are off, which is the shipped configuration. The slider does not apply
``average_over``/``dilation_size``.

Per-tile counts are computed on the 60 m grid with overlap weights (6 or 3 per
axis, since a tile edge falls half-way through every other 60 m pixel), which is
arithmetically identical to upsampling ×6 and cutting 549 px tiles -- checked
against the pipeline's own ``report.csv`` when ``--run`` is given.
"""
from __future__ import annotations

import argparse
import base64
import csv
import json
import os
import sys

import cv2
import numpy as np

from .constants import GRID, TILE_PIXELS, TILE_PX
from .detectors.esa import read_classi_mask
from .detectors.s2cloudless import load_config
from .io import read_tci
from .metadata import ProductError, read_product
from .view_mask import MASK_SIZE, TILE_PX_60M, _data_uri_jpeg
from .view_thresholds import _outline

FACTOR = 6
PROB_SCALE = 2 ** 24
CURVE_STEP = 0.01
MAX_BYTES = 25 * 1024 * 1024

#: Threshold values offered as buttons. Only the threshold is applied; the
#: library default and the ESA-agreement optimum also carry averaging and
#: dilation, which this viewer leaves off.
PRESETS = [
    {"label": "0.4 library default", "value": 0.4},
    {"label": "0.6 config/s2cloudless.json", "value": None},  # filled from the config
    {"label": "0.8 ESA-agreement optimum", "value": 0.8},
]


def overlap_weights(n60: int = MASK_SIZE, grid: int = GRID) -> np.ndarray:
    """``W[y, r]`` = how many 10 m rows of 60 m row ``y`` fall inside tile row ``r``."""
    w = np.zeros((n60, grid), dtype=np.int64)
    for y in range(n60):
        lo, hi = y * FACTOR, (y + 1) * FACTOR
        for r in range(grid):
            overlap = min(hi, (r + 1) * TILE_PX) - max(lo, r * TILE_PX)
            if overlap > 0:
                w[y, r] = overlap
    return w


def tile_counts(mask60: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Cloud pixels per tile at 10 m, as a ``grid x grid`` integer array."""
    return weights.T @ mask60.astype(np.int64) @ weights


def encode_probability(prob: np.ndarray) -> np.ndarray:
    scaled = np.round(np.clip(prob.astype(np.float64), 0.0, 1.0) * PROB_SCALE)
    return np.minimum(scaled, PROB_SCALE - 1).astype(np.uint32)


def threshold_code(t: float) -> float:
    """``prob > t`` on the encoded map is ``q > t * 2**24``; the page does the same."""
    return t * PROB_SCALE


def _data_uri_prob_png(q: np.ndarray) -> str:
    # cv2 writes BGR: red carries the high byte, green the middle, blue the low.
    bgr = np.zeros(q.shape + (3,), dtype=np.uint8)
    bgr[..., 2] = (q >> 16) & 0xFF
    bgr[..., 1] = (q >> 8) & 0xFF
    bgr[..., 0] = q & 0xFF
    ok, buf = cv2.imencode(".png", bgr)
    if not ok:
        raise IOError("failed to encode the probability map")
    decoded = cv2.imdecode(np.frombuffer(buf.tobytes(), np.uint8), cv2.IMREAD_UNCHANGED)
    if not np.array_equal(decoded, bgr):
        raise AssertionError("probability PNG round-trip does not match the source array")
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


def scene_stats(qprob: np.ndarray, weights: np.ndarray, t: float) -> dict:
    counts = tile_counts(qprob > threshold_code(t), weights)
    invalid = int(np.count_nonzero(100 * counts > 30 * TILE_PIXELS))
    return {
        "invalid": invalid,
        "scene_percent": 100.0 * float(counts.sum()) / (TILE_PIXELS * GRID * GRID),
        "tile_percent": 100.0 * counts / TILE_PIXELS,
    }


def _check_against_run(run_dir: str, qprob: np.ndarray, weights: np.ndarray) -> dict | None:
    """Compare the viewer's arithmetic with the pipeline's own report.csv."""
    summary_path = os.path.join(run_dir, "run_summary.json")
    report_path = os.path.join(run_dir, "report.csv")
    with open(summary_path, encoding="utf8") as handle:
        params = json.load(handle)["parameters"]
    if params.get("average_over") or params.get("dilation_size"):
        return None  # morphology on: the slider cannot reproduce that run
    with open(report_path, encoding="utf8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    expected = np.array([float(r["cloud_cover_percent"]) for r in rows]).reshape(GRID, GRID)
    got = scene_stats(qprob, weights, params["prob_threshold"])
    gap = float(np.abs(got["tile_percent"] - expected).max())
    if gap > 1e-4:
        raise AssertionError(
            "viewer tile percentages differ from %s by up to %.6f pp at threshold %s"
            % (report_path, gap, params["prob_threshold"])
        )
    expected_invalid = sum(r["valid"] == "False" for r in rows)
    if got["invalid"] != expected_invalid:
        raise AssertionError(
            "viewer counts %d invalid tiles, %s has %d" % (got["invalid"], report_path, expected_invalid)
        )
    return {"threshold": params["prob_threshold"], "invalid": expected_invalid, "max_gap_pp": gap}


def _compute_probability(meta) -> np.ndarray:
    from s2cloudless import S2PixelCloudDetector

    from .detectors.s2cloudless import build_reflectance_stack

    stack = build_reflectance_stack(meta)
    return S2PixelCloudDetector(all_bands=True).get_cloud_probability_maps(stack[None, ...])[0]


def build_viewer_html(safe_dir: str, run_dir: str | None = None, config_path: str | None = None) -> str:
    meta = read_product(safe_dir)
    config = load_config(config_path)

    npy = os.path.join(run_dir, "cloud_probability_60m.npy") if run_dir else None
    prob = np.load(npy) if npy and os.path.isfile(npy) else _compute_probability(meta)
    if prob.shape != (MASK_SIZE, MASK_SIZE):
        raise AssertionError("probability map is %s, expected %dx%d" % (prob.shape, MASK_SIZE, MASK_SIZE))

    qprob = encode_probability(prob)
    weights = overlap_weights()

    run_check = _check_against_run(run_dir, qprob, weights) if run_dir else None

    thresholds = np.round(np.arange(0.0, 1.0 + CURVE_STEP / 2, CURVE_STEP), 2)
    curve = []
    for t in thresholds:
        stats = scene_stats(qprob, weights, float(t))
        full_counts = tile_counts(prob > t, weights)
        full_invalid = int(np.count_nonzero(100 * full_counts > 30 * TILE_PIXELS))
        if full_invalid != stats["invalid"]:
            raise AssertionError(
                "threshold %.2f: 24-bit map gives %d invalid tiles, full precision %d"
                % (t, stats["invalid"], full_invalid)
            )
        curve.append({"t": float(t), "invalid": stats["invalid"], "scene": round(stats["scene_percent"], 4)})

    presets = [dict(p) for p in PRESETS]
    for preset in presets:
        if preset["value"] is None:
            preset["value"] = config["prob_threshold"]
            preset["label"] = "%g config/s2cloudless.json" % config["prob_threshold"]
        stats = scene_stats(qprob, weights, preset["value"])
        preset["invalid"] = stats["invalid"]
        preset["scene"] = stats["scene_percent"]

    opaque, cirrus, _snow = read_classi_mask(meta)
    esa_mask = opaque | cirrus
    esa_counts = tile_counts(esa_mask, weights)

    payload = {
        "product": meta.product_uri,
        "maskSize": MASK_SIZE,
        "tilePx60": TILE_PX_60M,
        "tilePx10": TILE_PX,
        "factor": FACTOR,
        "grid": GRID,
        "tilePixels": TILE_PIXELS,
        "probScale": PROB_SCALE,
        "initial": config["prob_threshold"],
        "presets": presets,
        "curve": curve,
        "esaReference": {
            "invalid": int(np.count_nonzero(100 * esa_counts > 30 * TILE_PIXELS)),
            "scene": 100.0 * float(esa_counts.sum()) / (TILE_PIXELS * GRID * GRID),
        },
        "runCheck": run_check,
        "base": _data_uri_jpeg(read_tci(meta, out_shape=(MASK_SIZE, MASK_SIZE))),
        "prob": _data_uri_prob_png(qprob),
        "esaOutline": "data:image/png;base64,"
        + base64.b64encode(cv2.imencode(".png", _outline(esa_mask).astype(np.uint8) * 255)[1].tobytes()).decode("ascii"),
    }
    html = _TEMPLATE.replace("__PAYLOAD__", json.dumps(payload, separators=(",", ":")))
    size = len(html.encode("utf8"))
    if size > MAX_BYTES:
        raise AssertionError("viewer is %.1f MB, over the %.0f MB ceiling" % (size / 1e6, MAX_BYTES / 1e6))
    return html


_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>s2cloudless threshold viewer</title>
<style>
  :root { color-scheme: dark;
    --bg: #12151a; --panel: #1a1e25; --line: #2a2f38; --ink: #e6e8eb; --ink-2: #aab2c0; --muted: #8b93a1;
    --series: #60a5fa; --mark: #ffd166; --ref: #8b93a1; --mask: 34,211,238; }
  * { box-sizing: border-box; }
  html, body { margin: 0; height: 100%; background: var(--bg); color: var(--ink);
    font: 13px/1.4 -apple-system, Segoe UI, Arial, sans-serif; }
  body { display: flex; }
  #sidebar { width: 360px; flex: none; padding: 14px; overflow-y: auto;
    background: var(--panel); border-right: 1px solid var(--line); }
  #sidebar h1 { font-size: 15px; margin: 0 0 2px; }
  #sidebar .sub { color: var(--muted); margin-bottom: 10px; word-break: break-all; }
  fieldset { border: 1px solid var(--line); border-radius: 6px; margin: 0 0 12px; padding: 8px 10px; }
  legend { padding: 0 4px; color: var(--ink-2); font-size: 11px; text-transform: uppercase; letter-spacing: .04em; }
  label { display: flex; align-items: center; gap: 7px; margin: 5px 0; }
  input[type=range] { width: 100%; }
  .top { display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 3px; }
  .big { font-size: 22px; font-weight: 600; font-variant-numeric: tabular-nums; }
  .stat { display: flex; justify-content: space-between; margin: 3px 0; }
  .stat b { font-variant-numeric: tabular-nums; }
  .k { color: var(--muted); }
  .hint { color: #6b7280; font-size: 11px; margin-top: 2px; }
  .presets { display: flex; gap: 4px; flex-wrap: wrap; margin-top: 6px; }
  button { background: #262b34; color: var(--ink); border: 1px solid #363c47; border-radius: 5px;
    padding: 5px 8px; cursor: pointer; font-size: 11px; flex: 1 1 auto; }
  button:hover { background: #2f3540; }
  button.wide { width: 100%; margin-top: 6px; }
  .chart { position: relative; }
  .chart canvas { display: block; width: 100%; height: 150px; cursor: crosshair; }
  .chart .title { font-size: 12px; color: var(--ink-2); margin: 4px 0 2px; }
  .tip { position: absolute; pointer-events: none; background: rgba(10,12,16,.92); border: 1px solid var(--line);
    border-radius: 5px; padding: 3px 6px; font-size: 11px; white-space: nowrap; display: none; }
  details { margin-top: 6px; }
  summary { cursor: pointer; color: var(--ink-2); font-size: 11px; }
  table { width: 100%; border-collapse: collapse; font-size: 11px; font-variant-numeric: tabular-nums; margin-top: 4px; }
  th, td { text-align: right; padding: 2px 4px; border-bottom: 1px solid var(--line); }
  th:first-child, td:first-child { text-align: left; }
  #stageWrap { position: relative; flex: 1; overflow: hidden; background: #05070a; cursor: grab; }
  #stageWrap.dragging { cursor: grabbing; }
  #stage { position: absolute; left: 0; top: 0; image-rendering: pixelated; }
  .badge { position: absolute; background: rgba(10,12,16,.88); border: 1px solid var(--line);
    border-radius: 6px; padding: 6px 10px; }
  #tileBadge { left: 10px; top: 10px; font-weight: 600; display: none; }
  .invalid { color: #f87171; } .valid { color: #4ade80; }
  #readout { left: 10px; bottom: 10px; min-width: 240px; pointer-events: none; }
  #readout .stat { gap: 12px; }
  #selfCheck { right: 10px; top: 10px; font-size: 11px; font-weight: 600; }
  #selfCheck.pass { color: #4ade80; border-color: #16652f; }
  #selfCheck.fail { color: #f87171; border-color: #7f1d1d; }
  .tile-jump { display: flex; gap: 4px; flex-wrap: wrap; }
</style>
</head>
<body>
<div id="sidebar">
  <h1>s2cloudless: probability threshold</h1>
  <div class="sub" id="productName"></div>

  <fieldset>
    <legend>Threshold</legend>
    <div class="top"><span class="k">cloud = probability &gt;</span><span class="big" id="valT"></span></div>
    <input type="range" id="slider" min="0" max="1" step="0.005">
    <div class="presets" id="presets"></div>
    <div class="hint">Threshold only: averaging and dilation are not applied (the shipped config has both off).</div>
  </fieldset>

  <fieldset>
    <legend>At this threshold</legend>
    <div class="stat"><span class="k">Scene cloud</span><b id="scenePct"></b></div>
    <div class="stat"><span class="k">Invalid tiles (&gt; 30 %)</span><b id="invalidCount"></b></div>
    <div class="stat"><span class="k">ESA baseline (reference, not truth)</span><b id="esaRef"></b></div>
  </fieldset>

  <fieldset>
    <legend>How the threshold moves the result</legend>
    <div class="chart">
      <div class="title">Invalid tiles (of 400)</div>
      <canvas id="chartInvalid"></canvas>
      <div class="tip" id="tipInvalid"></div>
    </div>
    <div class="chart">
      <div class="title">Scene cloud %</div>
      <canvas id="chartScene"></canvas>
      <div class="tip" id="tipScene"></div>
    </div>
    <div class="hint">Click a chart to set the threshold. Dashed line: ESA baseline.</div>
    <details>
      <summary>Table</summary>
      <table id="curveTable"><thead><tr><th>threshold</th><th>invalid tiles</th><th>scene cloud %</th></tr></thead><tbody></tbody></table>
    </details>
  </fieldset>

  <fieldset>
    <legend>Layers</legend>
    <label><input type="checkbox" id="toggleBase" checked> True colour (TCI, 60 m)</label>
    <label><input type="checkbox" id="toggleMask" checked> Cloud mask at this threshold</label>
    <label><input type="checkbox" id="toggleProb"> Probability (brighter = more likely cloud)</label>
    <label><input type="checkbox" id="toggleGrid" checked> 20&times;20 tile grid</label>
    <label><input type="checkbox" id="toggleTint" checked> Darken invalid tiles</label>
    <label><input type="checkbox" id="toggleEsa"> ESA outline (reference only)</label>
    <label for="opacity" class="k" style="margin-top:8px">Overlay opacity</label>
    <input type="range" id="opacity" min="0" max="100" value="55">
  </fieldset>

  <fieldset>
    <legend>Named tiles</legend>
    <div class="tile-jump">
      <button data-row="11" data-col="2">11,2 clear</button>
      <button data-row="6" data-col="5">6,5 moderate</button>
      <button data-row="5" data-col="16">5,16 heavy</button>
      <button data-row="12" data-col="19">12,19 veil</button>
    </div>
    <button class="wide" id="resetView">Reset zoom / pan</button>
  </fieldset>
</div>

<div id="stageWrap">
  <canvas id="stage"></canvas>
  <div class="badge" id="selfCheck">checking...</div>
  <div class="badge" id="tileBadge"></div>
  <div class="badge" id="readout">
    <div class="stat"><span class="k">60&nbsp;m pixel</span><span id="roPixel">&ndash;</span></div>
    <div class="stat"><span class="k">cloud probability</span><span id="roProb">&ndash;</span></div>
  </div>
</div>

<script>
const DATA = __PAYLOAD__;
const N = DATA.maskSize, G = DATA.grid, F = DATA.factor, T10 = DATA.tilePx10;
const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
document.getElementById('productName').textContent = DATA.product;

const state = { t: DATA.initial, opacity: 0.55,
  show: { base: true, mask: true, prob: false, grid: true, tint: true, esa: false },
  scale: 1, minScale: 1, panX: 0, panY: 0, _initialised: false };

// Per 60 m row/column: the tile index it starts in, and how many of its 6
// 10 m rows fall in that tile vs the next (6/0 or 3/3 at a half-pixel edge).
const firstTile = new Int32Array(N), firstW = new Int32Array(N);
for (let y = 0; y < N; y++) {
  const lo = y * F, r = Math.floor(lo / T10);
  firstTile[y] = r;
  firstW[y] = Math.min(lo + F, (r + 1) * T10) - lo;
}

let qprob, esaOutline, baseImg;
const mask = new Uint8Array(N * N);
let tileCounts = new Float64Array(G * G);

function recompute() {
  const code = state.t * DATA.probScale;
  tileCounts = new Float64Array(G * G);
  for (let y = 0; y < N; y++) {
    const ry = firstTile[y], wy = firstW[y], wy2 = F - wy;
    for (let x = 0; x < N; x++) {
      const i = y * N + x;
      const c = qprob[i] > code ? 1 : 0;
      mask[i] = c;
      if (!c) continue;
      const rx = firstTile[x], wx = firstW[x], wx2 = F - wx;
      tileCounts[ry * G + rx] += wy * wx;
      if (wx2) tileCounts[ry * G + rx + 1] += wy * wx2;
      if (wy2) {
        tileCounts[(ry + 1) * G + rx] += wy2 * wx;
        if (wx2) tileCounts[(ry + 1) * G + rx + 1] += wy2 * wx2;
      }
    }
  }
  let total = 0, invalid = 0;
  for (const c of tileCounts) { total += c; if (100 * c > 30 * DATA.tilePixels) invalid++; }
  return { invalid, scene: 100 * total / (DATA.tilePixels * G * G) };
}

// ---------------------------------------------------------------- map
const off = document.createElement('canvas'); off.width = N; off.height = N;
const offCtx = off.getContext('2d');
let baseData;

function composite() {
  if (state.show.base) offCtx.drawImage(baseImg, 0, 0, N, N);
  else { offCtx.fillStyle = '#0a0c10'; offCtx.fillRect(0, 0, N, N); }
  const img = offCtx.getImageData(0, 0, N, N), d = img.data;
  const a = state.opacity;
  const [mr, mg, mb] = css('--mask').split(',').map(Number);
  for (let i = 0, p = 0; i < mask.length; i++, p += 4) {
    if (state.show.prob) {
      const v = qprob[i] / DATA.probScale;
      d[p] = d[p] * (1 - v) + 255 * v; d[p + 1] = d[p + 1] * (1 - v) + 255 * v; d[p + 2] = d[p + 2] * (1 - v) + 255 * v;
    }
    if (state.show.mask && mask[i]) {
      d[p] = d[p] * (1 - a) + mr * a; d[p + 1] = d[p + 1] * (1 - a) + mg * a; d[p + 2] = d[p + 2] * (1 - a) + mb * a;
    }
    if (state.show.esa && esaOutline[i]) { d[p] = 255; d[p + 1] = 60; d[p + 2] = 90; }
  }
  offCtx.putImageData(img, 0, 0);
}

const stage = document.getElementById('stage'), ctx = stage.getContext('2d');
const wrap = document.getElementById('stageWrap');
const edge60 = (i) => i * DATA.tilePx60;

function resizeCanvas() {
  const w = wrap.clientWidth, h = wrap.clientHeight;
  if (w <= 0 || h <= 0) return;
  stage.width = w; stage.height = h;
  state.minScale = Math.min(w / N, h / N) * 0.98;
  if (!state._initialised) {
    state.scale = state.minScale;
    state.panX = (w - N * state.scale) / 2; state.panY = (h - N * state.scale) / 2;
    state._initialised = true;
  }
  draw();
}
new ResizeObserver(resizeCanvas).observe(wrap);

function draw() {
  ctx.imageSmoothingEnabled = false;
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.fillStyle = '#05070a'; ctx.fillRect(0, 0, stage.width, stage.height);
  ctx.setTransform(state.scale, 0, 0, state.scale, state.panX, state.panY);
  ctx.drawImage(off, 0, 0);
  if (state.show.tint) {
    ctx.fillStyle = 'rgba(0,0,0,0.5)';
    for (let r = 0; r < G; r++) for (let c = 0; c < G; c++) {
      if (100 * tileCounts[r * G + c] > 30 * DATA.tilePixels)
        ctx.fillRect(edge60(c), edge60(r), DATA.tilePx60, DATA.tilePx60);
    }
  }
  if (state.show.grid) {
    ctx.lineWidth = 1 / state.scale; ctx.strokeStyle = 'rgba(255,255,255,0.35)';
    ctx.beginPath();
    for (let i = 0; i <= G; i++) { const p = edge60(i); ctx.moveTo(p, 0); ctx.lineTo(p, N); ctx.moveTo(0, p); ctx.lineTo(N, p); }
    ctx.stroke();
  }
}

// -------------------------------------------------------------- charts
function makeChart(canvasId, tipId, key, refValue, fmt) {
  const canvas = document.getElementById(canvasId), tip = document.getElementById(tipId);
  const pad = { l: 34, r: 8, t: 8, b: 20 };
  const values = DATA.curve.map((p) => p[key]);
  const yMax = Math.max(...values, refValue) * 1.05 || 1;
  let hoverT = null;

  function geom() {
    const w = canvas.clientWidth, h = canvas.clientHeight;
    return { w, h, x: (t) => pad.l + t * (w - pad.l - pad.r), y: (v) => h - pad.b - (v / yMax) * (h - pad.t - pad.b) };
  }
  function valueAt(t) {
    const i = Math.round(t / 0.01);
    return DATA.curve[Math.max(0, Math.min(DATA.curve.length - 1, i))];
  }
  function render(current) {
    const dpr = window.devicePixelRatio || 1, g = geom();
    canvas.width = g.w * dpr; canvas.height = g.h * dpr;
    const c = canvas.getContext('2d'); c.setTransform(dpr, 0, 0, dpr, 0, 0);
    c.clearRect(0, 0, g.w, g.h);
    c.font = '10px -apple-system, Segoe UI, Arial, sans-serif';
    // recessive axes and ticks
    c.strokeStyle = css('--line'); c.fillStyle = css('--muted'); c.lineWidth = 1;
    c.beginPath(); c.moveTo(pad.l, g.h - pad.b); c.lineTo(g.w - pad.r, g.h - pad.b); c.stroke();
    c.textAlign = 'center';
    for (const t of [0, 0.2, 0.4, 0.6, 0.8, 1]) c.fillText(t.toFixed(1), g.x(t), g.h - 6);
    c.textAlign = 'right';
    for (const f of [0, 0.5, 1]) {
      const v = yMax / 1.05 * f;
      c.fillText(fmt(v), pad.l - 4, g.y(v) + 3);
      c.beginPath(); c.moveTo(pad.l, g.y(v)); c.lineTo(g.w - pad.r, g.y(v)); c.globalAlpha = 0.4; c.stroke(); c.globalAlpha = 1;
    }
    // ESA reference
    c.setLineDash([4, 3]); c.strokeStyle = css('--ref');
    c.beginPath(); c.moveTo(pad.l, g.y(refValue)); c.lineTo(g.w - pad.r, g.y(refValue)); c.stroke();
    c.setLineDash([]); c.textAlign = 'left'; c.fillStyle = css('--muted');
    c.fillText('ESA ' + fmt(refValue), pad.l + 3, g.y(refValue) - 3);
    // series
    c.strokeStyle = css('--series'); c.lineWidth = 2; c.lineJoin = 'round';
    c.beginPath();
    DATA.curve.forEach((p, i) => { const X = g.x(p.t), Y = g.y(p[key]); i ? c.lineTo(X, Y) : c.moveTo(X, Y); });
    c.stroke();
    // current threshold
    const cx = g.x(current.t), cy = g.y(current.v);
    c.strokeStyle = css('--mark'); c.lineWidth = 1;
    c.beginPath(); c.moveTo(cx, pad.t); c.lineTo(cx, g.h - pad.b); c.stroke();
    c.fillStyle = css('--mark'); c.strokeStyle = css('--panel'); c.lineWidth = 2;
    c.beginPath(); c.arc(cx, cy, 4.5, 0, 2 * Math.PI); c.fill(); c.stroke();
    // hover crosshair
    if (hoverT !== null) {
      const p = valueAt(hoverT), hx = g.x(p.t), hy = g.y(p[key]);
      c.strokeStyle = css('--ink-2'); c.lineWidth = 1; c.globalAlpha = 0.5;
      c.beginPath(); c.moveTo(hx, pad.t); c.lineTo(hx, g.h - pad.b); c.stroke(); c.globalAlpha = 1;
      c.fillStyle = css('--series'); c.strokeStyle = css('--panel'); c.lineWidth = 2;
      c.beginPath(); c.arc(hx, hy, 4, 0, 2 * Math.PI); c.fill(); c.stroke();
    }
  }
  function tFromEvent(e) {
    const g = geom(), rect = canvas.getBoundingClientRect();
    const t = (e.clientX - rect.left - pad.l) / (g.w - pad.l - pad.r);
    return Math.max(0, Math.min(1, Math.round(t * 100) / 100));
  }
  canvas.addEventListener('mousemove', (e) => {
    hoverT = tFromEvent(e);
    const p = valueAt(hoverT), rect = canvas.getBoundingClientRect();
    tip.style.display = 'block';
    tip.textContent = 'threshold ' + p.t.toFixed(2) + ': ' + fmt(p[key]);
    const left = e.clientX - rect.left + 10;
    tip.style.left = Math.min(left, rect.width - tip.offsetWidth - 2) + 'px';
    tip.style.top = (e.clientY - rect.top + 22) + 'px';
    refreshCharts();
  });
  canvas.addEventListener('mouseleave', () => { hoverT = null; tip.style.display = 'none'; refreshCharts(); });
  canvas.addEventListener('click', (e) => setThreshold(tFromEvent(e)));
  new ResizeObserver(() => refreshCharts()).observe(canvas);
  return render;
}

let latest = { invalid: 0, scene: 0 };
const renderInvalid = makeChart('chartInvalid', 'tipInvalid', 'invalid', DATA.esaReference.invalid, (v) => String(Math.round(v)));
const renderScene = makeChart('chartScene', 'tipScene', 'scene', DATA.esaReference.scene, (v) => v.toFixed(1) + '%');
function refreshCharts() {
  renderInvalid({ t: state.t, v: latest.invalid });
  renderScene({ t: state.t, v: latest.scene });
}

const tbody = document.querySelector('#curveTable tbody');
for (const p of DATA.curve) {
  if (Math.round(p.t * 100) % 5) continue;
  const tr = document.createElement('tr');
  tr.innerHTML = '<td>' + p.t.toFixed(2) + '</td><td>' + p.invalid + '</td><td>' + p.scene.toFixed(2) + '</td>';
  tbody.appendChild(tr);
}

// ----------------------------------------------------------- interaction
function refreshAll() {
  latest = recompute();
  document.getElementById('valT').textContent = state.t.toFixed(3);
  document.getElementById('scenePct').textContent = latest.scene.toFixed(4) + ' %';
  document.getElementById('invalidCount').textContent = latest.invalid + ' / ' + G * G;
  composite(); draw(); refreshCharts();
}
function setThreshold(t) {
  state.t = t; document.getElementById('slider').value = t; refreshAll();
}
document.getElementById('slider').addEventListener('input', (e) => { state.t = Number(e.target.value); refreshAll(); });
document.getElementById('esaRef').textContent =
  DATA.esaReference.invalid + ' tiles, ' + DATA.esaReference.scene.toFixed(2) + ' %';

const presetBox = document.getElementById('presets');
for (const p of DATA.presets) {
  const b = document.createElement('button');
  b.textContent = p.label;
  b.addEventListener('click', () => setThreshold(p.value));
  presetBox.appendChild(b);
}

function toImageXY(cx, cy) {
  const r = wrap.getBoundingClientRect();
  return [(cx - r.left - state.panX) / state.scale, (cy - r.top - state.panY) / state.scale];
}
wrap.addEventListener('wheel', (e) => {
  e.preventDefault();
  const [ix, iy] = toImageXY(e.clientX, e.clientY);
  const next = Math.min(Math.max(state.scale * Math.exp(-e.deltaY * 0.0015), state.minScale), state.minScale * 60);
  state.panX -= ix * (next - state.scale); state.panY -= iy * (next - state.scale); state.scale = next;
  draw();
}, { passive: false });
let dragging = false, lastX = 0, lastY = 0;
wrap.addEventListener('mousedown', (e) => { dragging = true; lastX = e.clientX; lastY = e.clientY; wrap.classList.add('dragging'); });
window.addEventListener('mouseup', () => { dragging = false; wrap.classList.remove('dragging'); });
window.addEventListener('mousemove', (e) => {
  if (dragging) { state.panX += e.clientX - lastX; state.panY += e.clientY - lastY; lastX = e.clientX; lastY = e.clientY; draw(); }
  updateReadout(e);
});

function updateReadout(e) {
  if (!qprob) return;
  const [ix, iy] = toImageXY(e.clientX, e.clientY);
  const px = Math.floor(ix), py = Math.floor(iy), badge = document.getElementById('tileBadge');
  if (px < 0 || py < 0 || px >= N || py >= N || !(e.target instanceof Node && wrap.contains(e.target))) {
    document.getElementById('roPixel').textContent = '–';
    document.getElementById('roProb').textContent = '–';
    badge.style.display = 'none';
    return;
  }
  document.getElementById('roPixel').textContent = px + ', ' + py;
  document.getElementById('roProb').textContent = (qprob[py * N + px] / DATA.probScale).toFixed(3);
  const row = Math.floor(py / DATA.tilePx60), col = Math.floor(px / DATA.tilePx60);
  const pct = 100 * tileCounts[row * G + col] / DATA.tilePixels;
  const valid = 100 * tileCounts[row * G + col] <= 30 * DATA.tilePixels;
  badge.style.display = 'block';
  badge.innerHTML = 'tile ' + row + ',' + col + ' &mdash; ' + pct.toFixed(4) + '% &mdash; ' +
    '<span class="' + (valid ? 'valid">valid' : 'invalid">invalid') + '</span>';
}

function jumpToTile(row, col) {
  const w = wrap.clientWidth, h = wrap.clientHeight, s = DATA.tilePx60;
  state.scale = Math.max(Math.min(w / s, h / s) * 0.7, state.minScale);
  state.panX = w / 2 - (col + 0.5) * s * state.scale;
  state.panY = h / 2 - (row + 0.5) * s * state.scale;
  draw();
}
document.querySelectorAll('.tile-jump button').forEach((b) =>
  b.addEventListener('click', () => jumpToTile(Number(b.dataset.row), Number(b.dataset.col))));
document.getElementById('resetView').addEventListener('click', () => { state._initialised = false; resizeCanvas(); });

function wireToggle(id, key) {
  document.getElementById(id).addEventListener('change', (e) => { state.show[key] = e.target.checked; composite(); draw(); });
}
wireToggle('toggleBase', 'base'); wireToggle('toggleMask', 'mask'); wireToggle('toggleProb', 'prob');
wireToggle('toggleGrid', 'grid'); wireToggle('toggleTint', 'tint'); wireToggle('toggleEsa', 'esa');
document.getElementById('opacity').addEventListener('input', (e) => { state.opacity = e.target.value / 100; composite(); draw(); });

// The page must reproduce Python's numbers exactly: same 24-bit map, same
// integer tile weights. A mismatch means the browser altered the PNG's pixels.
function runSelfCheck() {
  const banner = document.getElementById('selfCheck'), saved = state.t, detail = [];
  let ok = true;
  const checks = DATA.presets.map((p) => ({ label: p.label, t: p.value, invalid: p.invalid, scene: p.scene }));
  for (const c of checks) {
    state.t = c.t;
    const got = recompute();
    const pass = got.invalid === c.invalid && Math.abs(got.scene - c.scene) < 1e-6;
    ok = ok && pass;
    detail.push(c.label + ': ' + got.invalid + ' invalid (expected ' + c.invalid + '), ' + got.scene.toFixed(4) + '%');
  }
  if (DATA.runCheck) detail.push('matches the pipeline report.csv at ' + DATA.runCheck.threshold + ' (' + DATA.runCheck.invalid + ' invalid)');
  state.t = saved;
  banner.className = 'badge ' + (ok ? 'pass' : 'fail');
  banner.textContent = ok ? 'self-check passed (exact)' : 'self-check FAILED';
  banner.title = detail.join('\n');
}

function loadImage(src) {
  return new Promise((res, rej) => { const i = new Image(); i.onload = () => res(i); i.onerror = rej; i.src = src; });
}
function pixels(img) {
  const c = document.createElement('canvas'); c.width = N; c.height = N;
  const cx = c.getContext('2d'); cx.drawImage(img, 0, 0);
  return cx.getImageData(0, 0, N, N).data;
}
Promise.all([loadImage(DATA.base), loadImage(DATA.prob), loadImage(DATA.esaOutline)]).then(([base, probImg, esaImg]) => {
  baseImg = base;
  const pd = pixels(probImg), ed = pixels(esaImg);
  qprob = new Uint32Array(N * N); esaOutline = new Uint8Array(N * N);
  for (let i = 0, p = 0; i < N * N; i++, p += 4) { qprob[i] = (pd[p] << 16) | (pd[p + 1] << 8) | pd[p + 2]; esaOutline[i] = ed[p] > 127 ? 1 : 0; }
  runSelfCheck();
  setThreshold(state.t);
  resizeCanvas();
});
</script>
</body>
</html>
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build a self-contained HTML viewer with a live s2cloudless threshold slider."
    )
    parser.add_argument("--safe-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--run",
        default=None,
        help="an s2cloudless output folder: reuses its cloud_probability_60m.npy (skipping the model) "
        "and checks the viewer against its report.csv",
    )
    parser.add_argument("--config", default=None, help="s2cloudless.json to seed the slider from")
    args = parser.parse_args(argv)

    try:
        html = build_viewer_html(args.safe_dir, args.run, args.config)
    except ProductError as error:
        print("error: %s" % error, file=sys.stderr)
        return 2
    except AssertionError as error:
        print("error: %s" % error, file=sys.stderr)
        return 3

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf8") as handle:
        handle.write(html)
    print("%s  (%.1f MB)" % (args.out, len(html.encode("utf8")) / 1e6))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
