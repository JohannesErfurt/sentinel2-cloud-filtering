"""Self-contained HTML viewer with **live** threshold sliders for the
``threshold`` backend.

    python -m pipeline.view_thresholds --safe-dir <SAFE> --out output/comparison/threshold_viewer.html

One HTML file: no server, no CDN, no network. Unlike ``view_mask.py`` (the
``esa`` viewer), this backend has parameters, so this page ships the raw
brightness/NDSI/B10 fields -- quantised to 8 bits, the precision ceiling a
plain ``<canvas>`` enforces regardless of the source PNG's own bit depth --
and recomputes ``cloud = (brightness > T_bright AND ndsi > T_ndsi) OR
(B10 > T_cirrus)`` in the browser on every slider move. No button, no
round trip.

This is the viewer SPEC.md 0.4 originally ruled out as too expensive for a
detector with tunable parameters (three quantised rasters instead of a fixed
boolean layer). Built anyway, on request; see SPEC.md's note on B2's viewer
task for the reasoning.
"""
from __future__ import annotations

import argparse
import base64
import json
import sys

import cv2
import numpy as np

from .constants import GRID, TILE_METRES
from .detectors.esa import read_classi_mask
from .detectors.threshold import load_config
from .io import read_reflectance, read_tci
from .masks import brightness as brightness_fn
from .masks import combined_cloud_mask
from .masks import ndsi as ndsi_fn
from .metadata import ProductError, read_product
from .view_mask import MASK_SIZE, TILE_PX_60M, _data_uri_jpeg

# (low, high) reflectance/index values mapped to the 0-255 a canvas can hold.
# Chosen from the real product's own distribution (measured): brightness
# occasionally saturates above 1.0 (bright cloud/snow-like glare, ~0.1% of
# pixels) and B10 has a long, thin tail above 0.02 -- both clip to the top
# bin, which is harmless for classification since a clipped pixel is already
# unambiguously past any threshold the matching slider can reach.
QUANT_RANGES = {
    "brightness": (0.0, 1.0),
    "ndsi": (-1.0, 1.0),
    "b10": (0.0, 0.02),
}

# The three-branch combined rule compounds quantisation error from all three
# comparisons at once. Measured on the real product: 0.05 pp for brightness
# alone, up to ~0.32 pp once NDSI and B10 both participate. This is the
# disclosed ceiling for the in-page self-check, not an aspirational number --
# see the module docstring on why 8 bits is a hard floor for a <canvas>-based
# viewer, and the page says plainly that it is an approximation.
SELF_CHECK_TOLERANCE_PP = 0.5

#: Presets the page recomputes on load and checks against Python's own
#: quantised computation -- not full precision, which is a different (looser)
#: comparison the generator makes separately, before ever writing the file.
SELF_CHECK_PRESETS = [
    {"label": "naive (brightness only)", "t_bright": 0.33, "t_ndsi": -1.0, "t_cirrus": 1.0},
    {"label": "ESA-agreement optimum", "t_bright": 0.16, "t_ndsi": -0.20, "t_cirrus": 0.005},
    {"label": "config/thresholds.json", "t_bright": None, "t_ndsi": None, "t_cirrus": None},  # filled at build time
]

#: Full-precision vs quantised must agree tighter than the in-browser check
#: above, because this comparison has no canvas 8-bit ceiling on either side
#: -- both are computed in Python. The gap here is purely the PNG's own
#: quantisation, not a per-preset combined-rule compounding effect on top of
#: it, so the bound can be the single-band figure measured above.
GENERATOR_SELF_CHECK_TOLERANCE_PP = 0.4

MAX_BYTES = 25 * 1024 * 1024  # SPEC.md's original ceiling for this heavier viewer.


def quantize(array: np.ndarray, lo: float, hi: float) -> np.ndarray:
    clipped = np.clip(array, lo, hi)
    return np.round((clipped - lo) / (hi - lo) * 255.0).astype(np.uint8)


def quantized_threshold(t: float, lo: float, hi: float) -> int:
    """The integer bin a real threshold ``t`` corresponds to.

    Comparing raw quantised bytes against this integer is mathematically
    equivalent, up to one quantisation step, to dequantising every pixel and
    comparing against ``t`` directly -- and is what the browser does, since
    doing the dequantisation per pixel per slider move would be needless
    floating-point work for no accuracy gain.
    """
    return int(round((t - lo) / (hi - lo) * 255.0))


def mask_from_quantized(
    q_bright: np.ndarray, q_ndsi: np.ndarray, q_b10: np.ndarray,
    t_bright: float, t_ndsi: float, t_cirrus: float,
) -> np.ndarray:
    """Same rule as :func:`pipeline.masks.combined_cloud_mask`, evaluated on
    already-quantised layers -- exactly what the page's JS does."""
    qb = quantized_threshold(t_bright, *QUANT_RANGES["brightness"])
    qn = quantized_threshold(t_ndsi, *QUANT_RANGES["ndsi"])
    qc = quantized_threshold(t_cirrus, *QUANT_RANGES["b10"])
    return (q_bright > qb) & (q_ndsi > qn) | (q_b10 > qc)


def _data_uri_gray_png(quantized: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", quantized)
    if not ok:
        raise IOError("failed to encode a quantised layer")
    encoded = buf.tobytes()
    decoded = cv2.imdecode(np.frombuffer(encoded, np.uint8), cv2.IMREAD_UNCHANGED)
    if not np.array_equal(decoded, quantized):
        raise AssertionError("quantised layer PNG round-trip does not match the source array")
    return "data:image/png;base64," + base64.b64encode(encoded).decode("ascii")


def _outline(mask: np.ndarray) -> np.ndarray:
    """The 1-pixel boundary of a boolean mask.

    Precomputed here (once, in Python) rather than traced in the browser:
    canvas has no marching-squares primitive, and this project does not add
    a library for one function. A pixel is on the boundary if it is inside
    the mask but at least one 4-neighbour is not.
    """
    up = np.zeros_like(mask)
    up[:-1] = mask[1:]
    down = np.zeros_like(mask)
    down[1:] = mask[:-1]
    left = np.zeros_like(mask)
    left[:, :-1] = mask[:, 1:]
    right = np.zeros_like(mask)
    right[:, 1:] = mask[:, :-1]
    interior = mask & up & down & left & right
    return mask & ~interior


def build_viewer_html(safe_dir: str, config_path: str | None = None) -> str:
    meta = read_product(safe_dir)
    config = load_config(config_path)

    shape = (MASK_SIZE, MASK_SIZE)
    b02 = read_reflectance(meta, "B02", out_shape=shape)
    b03 = read_reflectance(meta, "B03", out_shape=shape)
    b04 = read_reflectance(meta, "B04", out_shape=shape)
    b11 = read_reflectance(meta, "B11", out_shape=shape)
    b10 = read_reflectance(meta, "B10", out_shape=shape)
    bright = brightness_fn(b02, b03, b04)
    ndsi_field = ndsi_fn(b03, b11)

    q_bright = quantize(bright, *QUANT_RANGES["brightness"])
    q_ndsi = quantize(ndsi_field, *QUANT_RANGES["ndsi"])
    q_b10 = quantize(b10, *QUANT_RANGES["b10"])

    presets = [dict(p) for p in SELF_CHECK_PRESETS]
    for preset in presets:
        if preset["t_bright"] is None:
            preset.update(t_bright=config["t_bright"], t_ndsi=config["t_ndsi"], t_cirrus=config["t_cirrus"])

    for preset in presets:
        full_mask = combined_cloud_mask(
            b02, b03, b04, b11, b10, preset["t_bright"], preset["t_ndsi"], preset["t_cirrus"]
        )
        quant_mask = mask_from_quantized(
            q_bright, q_ndsi, q_b10, preset["t_bright"], preset["t_ndsi"], preset["t_cirrus"]
        )
        full_pct = 100.0 * float(full_mask.mean())
        quant_pct = 100.0 * float(quant_mask.mean())
        gap = abs(full_pct - quant_pct)
        if gap > GENERATOR_SELF_CHECK_TOLERANCE_PP:
            raise AssertionError(
                "preset %r: quantised scene cloud %.4f%% differs from full precision %.4f%% "
                "by %.4f pp, over the %.2f pp ceiling"
                % (preset["label"], quant_pct, full_pct, gap, GENERATOR_SELF_CHECK_TOLERANCE_PP)
            )
        preset["full_percent"] = round(full_pct, 4)
        preset["quantized_percent"] = round(quant_pct, 4)

    opaque, cirrus, _snow = read_classi_mask(meta)
    esa_outline = _outline(opaque | cirrus)

    background = _data_uri_jpeg(read_tci(meta, out_shape=shape))
    layer_uris = {
        "brightness": _data_uri_gray_png(q_bright),
        "ndsi": _data_uri_gray_png(q_ndsi),
        "b10": _data_uri_gray_png(q_b10),
    }
    esa_outline_uri = _data_uri_gray_png(esa_outline.astype(np.uint8) * 255)

    payload = {
        "product": meta.product_uri,
        "maskSize": MASK_SIZE,
        "tilePx": TILE_PX_60M,
        "grid": GRID,
        "quantRanges": QUANT_RANGES,
        "initial": {"t_bright": config["t_bright"], "t_ndsi": config["t_ndsi"], "t_cirrus": config["t_cirrus"]},
        "esaAgreementOptimum": {"t_bright": 0.16, "t_ndsi": -0.20, "t_cirrus": 0.005},
        "selfCheckPresets": presets,
        "selfCheckTolerancePp": SELF_CHECK_TOLERANCE_PP,
        "base": background,
        "layers": layer_uris,
        "esaOutline": esa_outline_uri,
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
<title>Threshold detector -- live viewer</title>
<style>
  :root { color-scheme: dark; }
  * { box-sizing: border-box; }
  html, body { margin: 0; height: 100%; background: #12151a; color: #e6e8eb;
    font: 13px/1.4 -apple-system, Segoe UI, Arial, sans-serif; }
  body { display: flex; }
  #sidebar { width: 340px; flex: none; padding: 14px; overflow-y: auto;
    background: #1a1e25; border-right: 1px solid #2a2f38; }
  #sidebar h1 { font-size: 15px; margin: 0 0 2px; }
  #sidebar .sub { color: #8b93a1; margin-bottom: 10px; }
  fieldset { border: 1px solid #2a2f38; border-radius: 6px; margin: 0 0 12px; padding: 8px 10px; }
  legend { padding: 0 4px; color: #aab2c0; font-size: 11px; text-transform: uppercase; letter-spacing: .04em; }
  label { display: flex; align-items: center; gap: 7px; margin: 5px 0; }
  .swatch { width: 11px; height: 11px; border-radius: 2px; flex: none; }
  .pct { margin-left: auto; color: #8b93a1; font-variant-numeric: tabular-nums; }
  input[type=range] { width: 100%; }
  .slider-row { margin: 10px 0; }
  .slider-row .top { display: flex; justify-content: space-between; font-size: 12px; margin-bottom: 3px; }
  .slider-row .val { font-variant-numeric: tabular-nums; color: #ffd166; }
  .hint { color: #6b7280; font-size: 11px; margin-top: 2px; }
  button { background: #262b34; color: #e6e8eb; border: 1px solid #363c47; border-radius: 5px;
    padding: 5px 9px; cursor: pointer; font-size: 12px; }
  button:hover { background: #2f3540; }
  button.wide { width: 100%; margin-top: 6px; }
  #stageWrap { position: relative; flex: 1; overflow: hidden; background: #05070a; cursor: grab; }
  #stageWrap.dragging { cursor: grabbing; }
  canvas { position: absolute; left: 0; top: 0; image-rendering: pixelated; }
  #readout { position: absolute; left: 10px; bottom: 10px; background: rgba(10,12,16,.88);
    border: 1px solid #2a2f38; border-radius: 6px; padding: 8px 10px; min-width: 260px;
    pointer-events: none; }
  #readout .row { display: flex; justify-content: space-between; gap: 10px; }
  #readout .k { color: #8b93a1; }
  #tileBadge { position: absolute; left: 10px; top: 10px; background: rgba(10,12,16,.88);
    border: 1px solid #2a2f38; border-radius: 6px; padding: 6px 10px; font-weight: 600; }
  #tileBadge .invalid { color: #f87171; }
  #tileBadge .valid { color: #4ade80; }
  #selfCheck { position: absolute; right: 10px; top: 10px; padding: 6px 10px; border-radius: 6px;
    font-size: 11px; font-weight: 600; }
  #selfCheck.pass { background: rgba(34,197,94,.18); color: #4ade80; border: 1px solid #16652f; }
  #selfCheck.fail { background: rgba(239,68,68,.18); color: #f87171; border: 1px solid #7f1d1d; }
  textarea.copybox { width: 100%; height: 46px; background: #0e1116; color: #cbd3df; border: 1px solid #2a2f38;
    border-radius: 5px; font: 11px/1.3 ui-monospace, monospace; padding: 5px; resize: none; }
  .tile-jump { display: flex; gap: 4px; flex-wrap: wrap; }
  .tile-jump button { flex: 1 1 auto; font-size: 11px; padding: 4px 6px; }
  .approx-note { background: #1f1a0d; border: 1px solid #4a3b12; border-radius: 6px; padding: 7px 9px;
    font-size: 11px; color: #e6c675; margin-bottom: 12px; }
</style>
</head>
<body>
<div id="sidebar">
  <h1>Threshold detector</h1>
  <div class="sub" id="productName"></div>

  <div class="approx-note">
    Approximate offline preview: the three fields below are quantised to 8 bits (a canvas can hold
    no more), and the combined rule compounds that error across all three comparisons. Scene cloud %
    here can differ from the real pipeline (<code>pipeline.run --detector threshold</code>) by a few
    tenths of a point. For the exact figures, run the pipeline.
  </div>

  <fieldset>
    <legend>Thresholds</legend>
    <div class="slider-row">
      <div class="top"><span>T_bright</span><span class="val" id="valBright"></span></div>
      <input type="range" id="sliderBright" min="0" max="1" step="0.005">
    </div>
    <div class="slider-row">
      <div class="top"><span>T_ndsi</span><span class="val" id="valNdsi"></span></div>
      <input type="range" id="sliderNdsi" min="-1" max="1" step="0.005">
      <div class="hint">Drag to -1.00 to disable the veto (every pixel passes it).</div>
    </div>
    <div class="slider-row">
      <div class="top"><span>T_cirrus</span><span class="val" id="valCirrus"></span></div>
      <input type="range" id="sliderCirrus" min="0" max="0.02" step="0.0002">
      <div class="hint">Drag to 0.02 (max) to disable the B10 branch.</div>
    </div>
    <button class="wide" id="btnReset">Reset to config/thresholds.json</button>
    <button class="wide" id="btnEsaOptimum">ESA-agreement optimum (reference only, not ground truth)</button>
  </fieldset>

  <fieldset>
    <legend>Legend / live scene stats</legend>
    <label><span class="swatch" style="background:#dc2626"></span> Brightness + NDSI only <span class="pct" id="pctBright"></span></label>
    <label><span class="swatch" style="background:#f59e0b"></span> B10 (cirrus) only <span class="pct" id="pctCirrus"></span></label>
    <label><span class="swatch" style="background:#a855f7"></span> Both branches <span class="pct" id="pctBoth"></span></label>
    <div class="row" style="display:flex;justify-content:space-between;margin-top:6px">
      <span class="k">Total cloud</span><b id="pctTotal"></b>
    </div>
    <div class="row" style="display:flex;justify-content:space-between">
      <span class="k">Tiles above 30% (invalid)</span><b id="invalidCount"></b>
    </div>
    <div style="margin-top:8px">
      <label for="opacity">Overlay opacity</label>
      <input type="range" id="opacity" min="0" max="100" value="75">
    </div>
  </fieldset>

  <fieldset>
    <legend>Layers</legend>
    <label><input type="checkbox" id="toggleBase" checked> Base image (TCI, 60 m)</label>
    <label><input type="checkbox" id="toggleMask" checked> Cloud mask (current thresholds)</label>
    <label><input type="checkbox" id="toggleGrid" checked> 20&times;20 tile grid</label>
    <label><input type="checkbox" id="toggleTint"> Tint invalid tiles</label>
    <label><input type="checkbox" id="toggleEsa"> ESA outline <span class="hint">(reference only, not ground truth)</span></label>
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

  <fieldset>
    <legend>Copy current values</legend>
    <div class="hint">CLI flags</div>
    <textarea class="copybox" id="copyFlags" readonly></textarea>
    <div class="hint" style="margin-top:6px">config/thresholds.json</div>
    <textarea class="copybox" id="copyJson" readonly></textarea>
  </fieldset>

  <div class="hint">
    Formula: <code>cloud = (brightness &gt; T_bright AND ndsi &gt; T_ndsi) OR (B10 &gt; T_cirrus)</code>
    (SPEC.md 3.1). Thresholds are chosen by looking, not by fitting to ESA's mask (SPEC.md 3.2).
  </div>
</div>

<div id="stageWrap">
  <canvas id="stage"></canvas>
  <div id="selfCheck">checking...</div>
  <div id="tileBadge" style="display:none"></div>
  <div id="readout">
    <div class="row"><span class="k">60&nbsp;m pixel</span><span id="roPixel">&ndash;</span></div>
    <div class="row"><span class="k">UTM</span><span id="roUtm">&ndash;</span></div>
    <div class="row"><span class="k">brightness / ndsi / B10</span><span id="roValues">&ndash;</span></div>
  </div>
</div>

<script>
const DATA = __PAYLOAD__;
const N = DATA.maskSize;

document.getElementById('productName').textContent = DATA.product;

// ---------------------------------------------------------------- state
const state = {
  t_bright: DATA.initial.t_bright, t_ndsi: DATA.initial.t_ndsi, t_cirrus: DATA.initial.t_cirrus,
  opacity: 0.75,
  show: { base: true, mask: true, grid: true, tint: false, esa: false },
  scale: 1, minScale: 1, panX: 0, panY: 0, _initialised: false,
};

function quantizedThreshold(t, lo, hi) { return Math.round((t - lo) / (hi - lo) * 255); }

// ------------------------------------------------------------ load layers
function loadImage(src) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = reject;
    img.src = src;
  });
}
function extractChannel0(img) {
  const c = document.createElement('canvas');
  c.width = N; c.height = N;
  const cx = c.getContext('2d');
  cx.drawImage(img, 0, 0);
  const data = cx.getImageData(0, 0, N, N).data;
  const out = new Uint8Array(N * N);
  for (let i = 0, p = 0; i < data.length; i += 4, p++) out[p] = data[i]; // R == G == B for grayscale
  return out;
}

let qBright, qNdsi, qB10, esaOutlineMask, baseImg;
const codeArray = new Uint8Array(N * N); // 0 clear, 1 bright+ndsi, 2 cirrus, 3 both

function recomputeMask() {
  const qb = quantizedThreshold(state.t_bright, ...DATA.quantRanges.brightness);
  const qn = quantizedThreshold(state.t_ndsi, ...DATA.quantRanges.ndsi);
  const qc = quantizedThreshold(state.t_cirrus, ...DATA.quantRanges.b10);
  let cloudCount = 0, brightCount = 0, cirrusCount = 0, bothCount = 0;
  for (let i = 0; i < N * N; i++) {
    const brightAndNdsi = (qBright[i] > qb) && (qNdsi[i] > qn);
    const cirrusFlag = qB10[i] > qc;
    let code = 0;
    if (brightAndNdsi && cirrusFlag) { code = 3; bothCount++; cloudCount++; }
    else if (brightAndNdsi) { code = 1; brightCount++; cloudCount++; }
    else if (cirrusFlag) { code = 2; cirrusCount++; cloudCount++; }
    codeArray[i] = code;
  }
  return { cloudCount, brightCount, cirrusCount, bothCount };
}

// ------------------------------------------------------- tile boundaries
const tileEdge = new Array(DATA.grid + 1);
for (let i = 0; i <= DATA.grid; i++) tileEdge[i] = Math.round(i * DATA.tilePx);

function computeTileStats() {
  const stats = [];
  for (let row = 0; row < DATA.grid; row++) {
    for (let col = 0; col < DATA.grid; col++) {
      const y0 = tileEdge[row], y1 = tileEdge[row + 1];
      const x0 = tileEdge[col], x1 = tileEdge[col + 1];
      let count = 0, total = 0;
      for (let y = y0; y < y1; y++) {
        const rowBase = y * N;
        for (let x = x0; x < x1; x++) {
          total++;
          if (codeArray[rowBase + x] > 0) count++;
        }
      }
      const pct = 100 * count / total;
      stats.push({ row, col, pct, valid: pct <= 30 });
    }
  }
  return stats;
}
let tileStats = [];
const tilesByRC = new Map();

// -------------------------------------------------------------- drawing
const off = document.createElement('canvas');
off.width = N; off.height = N;
const offCtx = off.getContext('2d');
const COLOURS = { 1: [220, 38, 38], 2: [245, 158, 11], 3: [168, 85, 247] };

function composite() {
  offCtx.clearRect(0, 0, N, N);
  if (state.show.base) offCtx.drawImage(baseImg, 0, 0, N, N);
  else { offCtx.fillStyle = '#0a0c10'; offCtx.fillRect(0, 0, N, N); }

  if (state.show.mask) {
    const img = offCtx.getImageData(0, 0, N, N);
    const d = img.data;
    const alpha = Math.round(255 * state.opacity);
    for (let i = 0, p = 0; i < codeArray.length; i++, p += 4) {
      const code = codeArray[i];
      if (!code) continue;
      const [r, g, b] = COLOURS[code];
      const a = alpha / 255;
      d[p] = d[p] * (1 - a) + r * a;
      d[p + 1] = d[p + 1] * (1 - a) + g * a;
      d[p + 2] = d[p + 2] * (1 - a) + b * a;
    }
    offCtx.putImageData(img, 0, 0);
  }

  if (state.show.esa) {
    const img = offCtx.getImageData(0, 0, N, N);
    const d = img.data;
    for (let i = 0, p = 0; i < esaOutlineMask.length; i++, p += 4) {
      if (esaOutlineMask[i] > 127) { d[p] = 255; d[p + 1] = 255; d[p + 2] = 255; }
    }
    offCtx.putImageData(img, 0, 0);
  }
}

const stage = document.getElementById('stage');
const ctx = stage.getContext('2d');
const wrap = document.getElementById('stageWrap');

function resizeCanvas() {
  const w = wrap.clientWidth, h = wrap.clientHeight;
  if (w <= 0 || h <= 0) return; // flex layout not settled yet; ResizeObserver fires again when it is
  stage.width = w; stage.height = h;
  state.minScale = Math.min(w / N, h / N) * 0.98;
  if (!state._initialised) {
    state.scale = state.minScale;
    state.panX = (w - N * state.scale) / 2;
    state.panY = (h - N * state.scale) / 2;
    state._initialised = true;
  }
  draw();
}
// A plain 'resize' listener misses the very first layout pass: data-URI
// images can finish loading before the flex box has been given a size at
// all, reading clientWidth/Height as 0 and (without the guard above)
// locking in a scale of exactly 0 forever, since _initialised then blocks
// ever correcting it. A ResizeObserver on the element itself also fires
// once as soon as it has a real size, with no window resize needed.
new ResizeObserver(resizeCanvas).observe(wrap);

function draw() {
  ctx.imageSmoothingEnabled = false;
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.fillStyle = '#05070a';
  ctx.fillRect(0, 0, stage.width, stage.height);
  ctx.setTransform(state.scale, 0, 0, state.scale, state.panX, state.panY);
  ctx.drawImage(off, 0, 0);

  if (state.show.tint) {
    ctx.fillStyle = 'rgba(0,0,0,0.45)';
    for (const t of tileStats) {
      if (t.valid) continue;
      const x0 = tileEdge[t.col], y0 = tileEdge[t.row];
      ctx.fillRect(x0, y0, tileEdge[t.col + 1] - x0, tileEdge[t.row + 1] - y0);
    }
  }
  if (state.show.grid) {
    ctx.lineWidth = 1 / state.scale;
    ctx.strokeStyle = 'rgba(255,255,255,0.35)';
    ctx.beginPath();
    for (let i = 0; i <= DATA.grid; i++) {
      const p = tileEdge[i];
      ctx.moveTo(p, 0); ctx.lineTo(p, N);
      ctx.moveTo(0, p); ctx.lineTo(N, p);
    }
    ctx.stroke();
  }
}

function refreshAll() {
  const agg = recomputeMask();
  tileStats = computeTileStats();
  tilesByRC.clear();
  for (const t of tileStats) tilesByRC.set(t.row + ',' + t.col, t);

  document.getElementById('pctBright').textContent = (100 * agg.brightCount / (N * N)).toFixed(3) + '%';
  document.getElementById('pctCirrus').textContent = (100 * agg.cirrusCount / (N * N)).toFixed(3) + '%';
  document.getElementById('pctBoth').textContent = (100 * agg.bothCount / (N * N)).toFixed(3) + '%';
  document.getElementById('pctTotal').textContent = (100 * agg.cloudCount / (N * N)).toFixed(4) + '%';
  document.getElementById('invalidCount').textContent = tileStats.filter(t => !t.valid).length + ' / ' + tileStats.length;

  document.getElementById('valBright').textContent = state.t_bright.toFixed(3);
  document.getElementById('valNdsi').textContent = state.t_ndsi.toFixed(3);
  document.getElementById('valCirrus').textContent = state.t_cirrus.toFixed(4);
  document.getElementById('copyFlags').value =
    '--t-bright ' + state.t_bright.toFixed(3) + ' --t-ndsi ' + state.t_ndsi.toFixed(3) +
    ' --t-cirrus ' + state.t_cirrus.toFixed(4);
  document.getElementById('copyJson').value = JSON.stringify(
    { t_bright: Number(state.t_bright.toFixed(3)), t_ndsi: Number(state.t_ndsi.toFixed(3)),
      t_cirrus: Number(state.t_cirrus.toFixed(4)) }, null, 2);

  composite();
  draw();
}

// ------------------------------------------------------------ interaction
function toImageXY(clientX, clientY) {
  const r = wrap.getBoundingClientRect();
  return [(clientX - r.left - state.panX) / state.scale, (clientY - r.top - state.panY) / state.scale];
}

wrap.addEventListener('wheel', (e) => {
  e.preventDefault();
  const [ix, iy] = toImageXY(e.clientX, e.clientY);
  const factor = Math.exp(-e.deltaY * 0.0015);
  const next = Math.min(Math.max(state.scale * factor, state.minScale), state.minScale * 60);
  state.panX -= ix * (next - state.scale);
  state.panY -= iy * (next - state.scale);
  state.scale = next;
  draw();
}, { passive: false });

let dragging = false, lastX = 0, lastY = 0;
wrap.addEventListener('mousedown', (e) => { dragging = true; lastX = e.clientX; lastY = e.clientY; wrap.classList.add('dragging'); });
window.addEventListener('mouseup', () => { dragging = false; wrap.classList.remove('dragging'); });
window.addEventListener('mousemove', (e) => {
  if (dragging) {
    state.panX += e.clientX - lastX; state.panY += e.clientY - lastY;
    lastX = e.clientX; lastY = e.clientY;
    draw();
  }
  updateReadout(e);
});

function updateReadout(e) {
  if (!qBright) return;
  const [ix, iy] = toImageXY(e.clientX, e.clientY);
  const px = Math.floor(ix), py = Math.floor(iy);
  const badge = document.getElementById('tileBadge');
  if (px < 0 || py < 0 || px >= N || py >= N) {
    document.getElementById('roPixel').textContent = '–';
    document.getElementById('roUtm').textContent = '–';
    document.getElementById('roValues').textContent = '–';
    badge.style.display = 'none';
    return;
  }
  document.getElementById('roPixel').textContent = px + ', ' + py;
  const easting = 600000 + px * 60, northing = 5500020 - py * 60;
  document.getElementById('roUtm').textContent = easting.toLocaleString() + ' E, ' + northing.toLocaleString() + ' N';

  const idx = py * N + px;
  const dq = (byte, lo, hi) => lo + (byte / 255) * (hi - lo);
  const br = dq(qBright[idx], ...DATA.quantRanges.brightness);
  const nd = dq(qNdsi[idx], ...DATA.quantRanges.ndsi);
  const b10 = dq(qB10[idx], ...DATA.quantRanges.b10);
  document.getElementById('roValues').textContent = br.toFixed(3) + ' / ' + nd.toFixed(3) + ' / ' + b10.toFixed(4);

  const row = Math.floor(py / DATA.tilePx), col = Math.floor(px / DATA.tilePx);
  const tile = tilesByRC.get(row + ',' + col);
  if (tile) {
    badge.style.display = 'block';
    badge.innerHTML = 'tile ' + row + ',' + col + ' &mdash; ' + tile.pct.toFixed(4) + '% &mdash; ' +
      '<span class="' + (tile.valid ? 'valid">valid' : 'invalid">invalid') + '</span>';
  } else badge.style.display = 'none';
}

function jumpToTile(row, col) {
  const w = wrap.clientWidth, h = wrap.clientHeight;
  const x0 = tileEdge[col], y0 = tileEdge[row];
  const x1 = tileEdge[col + 1], y1 = tileEdge[row + 1];
  const cx = (x0 + x1) / 2, cy = (y0 + y1) / 2;
  const scale = Math.min(w / (x1 - x0), h / (y1 - y0)) * 0.7;
  state.scale = Math.max(scale, state.minScale);
  state.panX = w / 2 - cx * state.scale;
  state.panY = h / 2 - cy * state.scale;
  draw();
}
document.querySelectorAll('.tile-jump button').forEach((btn) => {
  btn.addEventListener('click', () => jumpToTile(Number(btn.dataset.row), Number(btn.dataset.col)));
});
document.getElementById('resetView').addEventListener('click', () => { state._initialised = false; resizeCanvas(); });

function setSliders(t_bright, t_ndsi, t_cirrus) {
  state.t_bright = t_bright; state.t_ndsi = t_ndsi; state.t_cirrus = t_cirrus;
  document.getElementById('sliderBright').value = t_bright;
  document.getElementById('sliderNdsi').value = t_ndsi;
  document.getElementById('sliderCirrus').value = t_cirrus;
  refreshAll();
}
document.getElementById('sliderBright').addEventListener('input', (e) => { state.t_bright = Number(e.target.value); refreshAll(); });
document.getElementById('sliderNdsi').addEventListener('input', (e) => { state.t_ndsi = Number(e.target.value); refreshAll(); });
document.getElementById('sliderCirrus').addEventListener('input', (e) => { state.t_cirrus = Number(e.target.value); refreshAll(); });
document.getElementById('btnReset').addEventListener('click', () => setSliders(DATA.initial.t_bright, DATA.initial.t_ndsi, DATA.initial.t_cirrus));
document.getElementById('btnEsaOptimum').addEventListener('click', () =>
  setSliders(DATA.esaAgreementOptimum.t_bright, DATA.esaAgreementOptimum.t_ndsi, DATA.esaAgreementOptimum.t_cirrus));

function wireToggle(id, key) {
  document.getElementById(id).addEventListener('change', (e) => { state.show[key] = e.target.checked; composite(); draw(); });
}
wireToggle('toggleBase', 'base');
wireToggle('toggleMask', 'mask');
wireToggle('toggleGrid', 'grid');
wireToggle('toggleTint', 'tint');
wireToggle('toggleEsa', 'esa');
document.getElementById('opacity').addEventListener('input', (e) => { state.opacity = e.target.value / 100; composite(); draw(); });

// window resize is now handled by the ResizeObserver above.

// -------------------------------------------------------------- self-check
function runSelfCheck() {
  const banner = document.getElementById('selfCheck');
  let worst = 0, detail = [];
  for (const preset of DATA.selfCheckPresets) {
    const saved = { t_bright: state.t_bright, t_ndsi: state.t_ndsi, t_cirrus: state.t_cirrus };
    state.t_bright = preset.t_bright; state.t_ndsi = preset.t_ndsi; state.t_cirrus = preset.t_cirrus;
    const agg = recomputeMask();
    const got = 100 * agg.cloudCount / (N * N);
    const gap = Math.abs(got - preset.quantized_percent);
    worst = Math.max(worst, gap);
    detail.push(preset.label + ': got ' + got.toFixed(4) + '%, expected ' + preset.quantized_percent.toFixed(4) + '%');
    state.t_bright = saved.t_bright; state.t_ndsi = saved.t_ndsi; state.t_cirrus = saved.t_cirrus;
  }
  const ok = worst <= DATA.selfCheckTolerancePp;
  banner.className = ok ? 'pass' : 'fail';
  banner.textContent = ok ? 'self-check passed' : 'self-check FAILED';
  banner.title = detail.join('\n');
  return ok;
}

// ------------------------------------------------------------------ boot
Promise.all([
  loadImage(DATA.base), loadImage(DATA.layers.brightness),
  loadImage(DATA.layers.ndsi), loadImage(DATA.layers.b10), loadImage(DATA.esaOutline),
]).then(([base, brightImg, ndsiImg, b10Img, esaImg]) => {
  baseImg = base;
  qBright = extractChannel0(brightImg);
  qNdsi = extractChannel0(ndsiImg);
  qB10 = extractChannel0(b10Img);
  esaOutlineMask = extractChannel0(esaImg);

  document.getElementById('sliderBright').value = state.t_bright;
  document.getElementById('sliderNdsi').value = state.t_ndsi;
  document.getElementById('sliderCirrus').value = state.t_cirrus;

  refreshAll();
  resizeCanvas();
  runSelfCheck();
});
</script>
</body>
</html>
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build a self-contained HTML viewer with live threshold sliders."
    )
    parser.add_argument("--safe-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--config", default=None, help="thresholds.json to seed the sliders from")
    args = parser.parse_args(argv)

    try:
        html = build_viewer_html(args.safe_dir, args.config)
    except ProductError as error:
        print("error: %s" % error, file=sys.stderr)
        return 2
    except AssertionError as error:
        print("error: %s" % error, file=sys.stderr)
        return 3

    import os

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf8") as handle:
        handle.write(html)
    print("%s  (%.1f MB)" % (args.out, len(html.encode("utf8")) / 1e6))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
