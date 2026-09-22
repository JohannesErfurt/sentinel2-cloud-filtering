"""Self-contained HTML viewer for the ``esa`` backend's cloud mask.

    python -m pipeline.view_mask --safe-dir <SAFE> --out output/esa/mask_viewer.html

One HTML file: no server, no CDN, no network. Everything -- the true-colour
background, the three MSK_CLASSI layers, and every tile's statistics -- is
embedded, so it opens offline in any browser and can be zipped with the
deliverables.

This exists because reading numbers off a CSV is a poor way to sanity-check a
cloud mask against the imagery it came from; seeing the two overlaid is not.
It is scoped to what that job needs -- a fixed mask, toggled layers, a tile
grid -- and deliberately does not attempt the general-purpose, threshold-
recomputing viewer this project's plan otherwise chose not to build (SPEC.md
0.4): there is nothing to recompute here, `esa` has no parameters.
"""
from __future__ import annotations

import argparse
import base64
import json
import sys

import cv2
import numpy as np

from .constants import GRID, TILE_METRES
from .detectors.esa import EsaDetector
from .io import read_tci
from .metadata import ProductError, read_product
from .report import read_report_csv
from .tiling import compute_tile_stats, iter_tiles

MASK_RESOLUTION = 60  # MSK_CLASSI's native grid; the viewer never upsamples it.
MASK_SIZE = 1830  # 20 tiles * 91.5 px/tile at 60 m.
TILE_PX_60M = TILE_METRES / MASK_RESOLUTION  # 91.5

LAYER_COLOURS = {
    # (R, G, B) -- also used as the legend swatch colour.
    "opaque": (220, 38, 38),
    "cirrus": (245, 158, 11),
    "snow": (34, 211, 238),
}
INVALID_TINT = (0, 0, 0)

MAX_BYTES = 15 * 1024 * 1024  # SPEC.md G1.3's original ceiling; still the sane one.


def _data_uri_jpeg(rgb: np.ndarray, quality: int = 87) -> str:
    ok, buf = cv2.imencode(".jpg", rgb[:, :, ::-1], [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise IOError("failed to encode the background image")
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


def _data_uri_layer_png(mask: np.ndarray, colour: tuple[int, int, int], alpha: int = 210) -> str:
    """A mask as an RGBA PNG: transparent where false, ``colour`` where true.

    Layering these as plain ``<img>`` elements with CSS opacity, rather than
    compositing in a canvas, is what lets a single slider control every
    layer's intensity with no redraw logic at all.
    """
    height, width = mask.shape
    rgba = np.zeros((height, width, 4), dtype=np.uint8)
    rgba[mask, 0] = colour[2]  # BGRA for cv2.imencode
    rgba[mask, 1] = colour[1]
    rgba[mask, 2] = colour[0]
    rgba[mask, 3] = alpha
    ok, buf = cv2.imencode(".png", rgba)
    if not ok:
        raise IOError("failed to encode a mask layer")
    encoded = buf.tobytes()

    # The self-check SPEC.md G1.3 asked for: decode what was just encoded and
    # confirm the pixel count survived the round trip, before it ever reaches
    # the page. A silently-wrong PNG here would look fine and mean nothing.
    decoded = cv2.imdecode(np.frombuffer(encoded, np.uint8), cv2.IMREAD_UNCHANGED)
    if not np.array_equal(decoded[:, :, 3] > 0, mask):
        raise AssertionError("mask layer PNG round-trip does not match the source mask")

    return "data:image/png;base64," + base64.b64encode(encoded).decode("ascii")


def _tile_records(meta, detector: EsaDetector, run_dir: str | None) -> list[dict]:
    """Per-tile stats for all 400 tiles, via the same code path as report.csv.

    Takes an already-loaded detector rather than building its own, so the mask
    is read and verified once for the whole viewer, not once per consumer.

    If ``run_dir`` is given, this cross-checks against its report.csv rather
    than trusting a second computation -- catching the generator and the
    pipeline silently drifting apart.
    """
    records = []
    unrounded_percent = []
    for row, col in iter_tiles():
        mask = detector.tile_mask(meta, row, col)
        stats = compute_tile_stats(meta, row, col, mask)
        unrounded_percent.append(stats.cloud_cover_percent)
        records.append(
            {
                "row": row,
                "col": col,
                "cloud_cover_percent": round(stats.cloud_cover_percent, 4),
                "valid": stats.valid,
                "min_latitude": round(stats.min_latitude, 6),
                "min_longitude": round(stats.min_longitude, 6),
                "max_latitude": round(stats.max_latitude, 6),
                "max_longitude": round(stats.max_longitude, 6),
            }
        )

    if run_dir:
        import os

        _, csv_rows = read_report_csv(os.path.join(run_dir, "report.csv"))
        if len(csv_rows) != len(records):
            raise AssertionError(
                "%s has %d rows, expected %d" % (run_dir, len(csv_rows), len(records))
            )
        # Compared against the unrounded value, not record["cloud_cover_percent"]:
        # that field and report.csv's own "%.6f" formatting round the same float
        # independently, so a value sitting on a .5 boundary can land on either
        # side by exactly one part in 1e4 without either number being wrong.
        for record, csv_row, exact in zip(records, csv_rows, unrounded_percent):
            got = float(csv_row["cloud_cover_percent"])
            if abs(got - exact) > 1e-3:
                raise AssertionError(
                    "tile (%d,%d): viewer computed %.6f%% but %s/report.csv says %.6f%%"
                    % (record["row"], record["col"], exact, run_dir, got)
                )
            if (csv_row["valid"] == "True") != record["valid"]:
                raise AssertionError(
                    "tile (%d,%d): valid disagrees with %s/report.csv"
                    % (record["row"], record["col"], run_dir)
                )
    return records


def build_viewer_html(safe_dir: str, run_dir: str | None = None) -> str:
    meta = read_product(safe_dir)
    detector = EsaDetector()
    detector._ensure_loaded(meta)  # one read + one brightness check, shared below
    opaque, cirrus, snow = detector._opaque60, detector._cirrus60, detector._snow60
    if opaque.shape != (MASK_SIZE, MASK_SIZE):
        raise AssertionError("MSK_CLASSI is %s, expected %dx%d" % (opaque.shape, MASK_SIZE, MASK_SIZE))

    union = opaque | cirrus
    counts = {
        "opaque": 100.0 * float(opaque.mean()),
        "cirrus": 100.0 * float(cirrus.mean()),
        "snow": 100.0 * float(snow.mean()),
        "total": 100.0 * float(union.mean()),
    }
    # The self-check SPEC.md G1.3 asked for, at the scene level: these must
    # match Cloud_Coverage_Assessment before anything is written to disk.
    if abs(counts["total"] - meta.cloud_coverage_assessment) > 5e-3:
        raise AssertionError(
            "embedded mask is %.4f%% cloud but the product says %.4f%%"
            % (counts["total"], meta.cloud_coverage_assessment)
        )

    background = read_tci(meta, out_shape=(MASK_SIZE, MASK_SIZE))
    base_uri = _data_uri_jpeg(background)
    layer_uris = {
        "opaque": _data_uri_layer_png(opaque, LAYER_COLOURS["opaque"]),
        "cirrus": _data_uri_layer_png(cirrus, LAYER_COLOURS["cirrus"]),
        "snow": _data_uri_layer_png(snow, LAYER_COLOURS["snow"]),
    }

    tiles = _tile_records(meta, detector, run_dir)
    detector.close()

    payload = {
        "product": meta.product_uri,
        "maskSize": MASK_SIZE,
        "tilePx": TILE_PX_60M,
        "grid": GRID,
        "counts": {k: round(v, 4) for k, v in counts.items()},
        "colours": {k: list(v) for k, v in LAYER_COLOURS.items()},
        "tiles": tiles,
        "base": base_uri,
        "layers": layer_uris,
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
<title>ESA cloud mask viewer</title>
<style>
  :root { color-scheme: dark; }
  * { box-sizing: border-box; }
  html, body { margin: 0; height: 100%; background: #12151a; color: #e6e8eb;
    font: 13px/1.4 -apple-system, Segoe UI, Arial, sans-serif; }
  body { display: flex; }
  #sidebar { width: 300px; flex: none; padding: 14px; overflow-y: auto;
    background: #1a1e25; border-right: 1px solid #2a2f38; }
  #sidebar h1 { font-size: 15px; margin: 0 0 2px; }
  #sidebar .sub { color: #8b93a1; margin-bottom: 14px; }
  fieldset { border: 1px solid #2a2f38; border-radius: 6px; margin: 0 0 12px; padding: 8px 10px; }
  legend { padding: 0 4px; color: #aab2c0; font-size: 11px; text-transform: uppercase; letter-spacing: .04em; }
  label { display: flex; align-items: center; gap: 7px; margin: 5px 0; cursor: pointer; }
  .swatch { width: 11px; height: 11px; border-radius: 2px; flex: none; }
  .pct { margin-left: auto; color: #8b93a1; font-variant-numeric: tabular-nums; }
  input[type=range] { width: 100%; }
  #stageWrap { position: relative; flex: 1; overflow: hidden; background: #05070a; cursor: grab; }
  #stageWrap.dragging { cursor: grabbing; }
  canvas { position: absolute; left: 0; top: 0; image-rendering: pixelated; }
  #readout { position: absolute; left: 10px; bottom: 10px; background: rgba(10,12,16,.85);
    border: 1px solid #2a2f38; border-radius: 6px; padding: 8px 10px; min-width: 220px;
    pointer-events: none; }
  #readout .row { display: flex; justify-content: space-between; gap: 10px; }
  #readout .k { color: #8b93a1; }
  #tileBadge { position: absolute; left: 10px; top: 10px; background: rgba(10,12,16,.85);
    border: 1px solid #2a2f38; border-radius: 6px; padding: 6px 10px; font-weight: 600; }
  #tileBadge .invalid { color: #f87171; }
  #tileBadge .valid { color: #4ade80; }
  button { background: #262b34; color: #e6e8eb; border: 1px solid #363c47; border-radius: 5px;
    padding: 5px 9px; cursor: pointer; font-size: 12px; }
  button:hover { background: #2f3540; }
  .hint { color: #6b7280; font-size: 11px; margin-top: 2px; }
</style>
</head>
<body>
<div id="sidebar">
  <h1>ESA cloud mask</h1>
  <div class="sub" id="productName"></div>

  <fieldset>
    <legend>Layers</legend>
    <label><input type="checkbox" id="toggleBase" checked> Base image (TCI, 60 m)</label>
    <label><input type="checkbox" id="toggleOpaque" checked>
      <span class="swatch" style="background:rgb(220,38,38)"></span> Opaque cloud
      <span class="pct" id="pctOpaque"></span></label>
    <label><input type="checkbox" id="toggleCirrus" checked>
      <span class="swatch" style="background:rgb(245,158,11)"></span> Cirrus
      <span class="pct" id="pctCirrus"></span></label>
    <label><input type="checkbox" id="toggleSnow" checked>
      <span class="swatch" style="background:rgb(34,211,238)"></span> Snow / ice
      <span class="pct" id="pctSnow"></span></label>
    <div class="row" style="display:flex;justify-content:space-between;margin-top:8px">
      <span class="k">Total cloud</span><b id="pctTotal"></b>
    </div>
    <div style="margin-top:8px">
      <label for="opacity">Layer opacity</label>
      <input type="range" id="opacity" min="0" max="100" value="80">
    </div>
  </fieldset>

  <fieldset>
    <legend>Tile grid</legend>
    <label><input type="checkbox" id="toggleGrid" checked> Show 20&times;20 grid</label>
    <label><input type="checkbox" id="toggleTint"> Tint invalid tiles (&gt;30% cloud)</label>
  </fieldset>

  <fieldset>
    <legend>View</legend>
    <button id="resetView">Reset zoom / pan</button>
    <div class="hint">Scroll to zoom, drag to pan. The mask never smooths.</div>
  </fieldset>

  <div class="hint" style="margin-top:14px">
    Reference: <b>MSK_CLASSI_B00.jp2</b>, ESA's own classification mask.
    This is the reference baseline, not a detector (SPEC.md section&nbsp;2).
  </div>
</div>

<div id="stageWrap">
  <canvas id="stage"></canvas>
  <div id="tileBadge" style="display:none"></div>
  <div id="readout">
    <div class="row"><span class="k">60&nbsp;m pixel</span><span id="roPixel">&ndash;</span></div>
    <div class="row"><span class="k">UTM</span><span id="roUtm">&ndash;</span></div>
    <div class="row"><span class="k">opaque / cirrus / snow</span><span id="roChannels">&ndash;</span></div>
  </div>
</div>

<script>
const DATA = __PAYLOAD__;

document.getElementById('productName').textContent = DATA.product;
document.getElementById('pctOpaque').textContent = DATA.counts.opaque.toFixed(3) + '%';
document.getElementById('pctCirrus').textContent = DATA.counts.cirrus.toFixed(3) + '%';
document.getElementById('pctSnow').textContent = DATA.counts.snow.toFixed(3) + '%';
document.getElementById('pctTotal').textContent = DATA.counts.total.toFixed(4) + '%';

const N = DATA.maskSize;
const tilesByRC = new Map();
for (const t of DATA.tiles) tilesByRC.set(t.row + ',' + t.col, t);

// ---- load images, then build a full-resolution offscreen composite -------
function loadImage(src) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = reject;
    img.src = src;
  });
}

const state = { scale: 1, minScale: 1, panX: 0, panY: 0, opacity: 0.8,
  show: { base: true, opaque: true, cirrus: true, snow: true, grid: true, tint: false } };

let images = {};
const off = document.createElement('canvas');
off.width = N; off.height = N;
const offCtx = off.getContext('2d');

const stage = document.getElementById('stage');
const ctx = stage.getContext('2d');
const wrap = document.getElementById('stageWrap');

function composite() {
  offCtx.clearRect(0, 0, N, N);
  if (state.show.base) offCtx.drawImage(images.base, 0, 0, N, N);
  else { offCtx.fillStyle = '#0a0c10'; offCtx.fillRect(0, 0, N, N); }
  offCtx.globalAlpha = state.opacity;
  if (state.show.opaque) offCtx.drawImage(images.opaque, 0, 0, N, N);
  if (state.show.cirrus) offCtx.drawImage(images.cirrus, 0, 0, N, N);
  if (state.show.snow) offCtx.drawImage(images.snow, 0, 0, N, N);
  offCtx.globalAlpha = 1;
}

function resizeCanvas() {
  const w = wrap.clientWidth, h = wrap.clientHeight;
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

function draw() {
  ctx.imageSmoothingEnabled = false;
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.fillStyle = '#05070a';
  ctx.fillRect(0, 0, stage.width, stage.height);
  ctx.setTransform(state.scale, 0, 0, state.scale, state.panX, state.panY);
  ctx.drawImage(off, 0, 0);

  if (state.show.tint) {
    ctx.fillStyle = 'rgba(0,0,0,0.45)';
    for (const t of DATA.tiles) {
      if (t.valid) continue;
      ctx.fillRect(t.col * DATA.tilePx, t.row * DATA.tilePx, DATA.tilePx, DATA.tilePx);
    }
  }
  if (state.show.grid) {
    ctx.lineWidth = 1 / state.scale;
    ctx.strokeStyle = 'rgba(255,255,255,0.35)';
    ctx.beginPath();
    for (let i = 0; i <= DATA.grid; i++) {
      const p = i * DATA.tilePx;
      ctx.moveTo(p, 0); ctx.lineTo(p, N);
      ctx.moveTo(0, p); ctx.lineTo(N, p);
    }
    ctx.stroke();
  }
}

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
  if (!maskAt.opaque) return; // layers not decoded yet
  const [ix, iy] = toImageXY(e.clientX, e.clientY);
  const px = Math.floor(ix), py = Math.floor(iy);
  const badge = document.getElementById('tileBadge');
  if (px < 0 || py < 0 || px >= N || py >= N) {
    document.getElementById('roPixel').textContent = '–';
    document.getElementById('roUtm').textContent = '–';
    document.getElementById('roChannels').textContent = '–';
    badge.style.display = 'none';
    return;
  }
  document.getElementById('roPixel').textContent = px + ', ' + py;

  const easting = 600000 + px * 60, northing = 5500020 - py * 60;
  document.getElementById('roUtm').textContent = easting.toLocaleString() + ' E, ' + northing.toLocaleString() + ' N';

  const flags = ['opaque', 'cirrus', 'snow'].map(name => maskAt[name][py * N + px] ? name[0].toUpperCase() : '·');
  document.getElementById('roChannels').textContent = flags.join(' ');

  const row = Math.floor(py / DATA.tilePx), col = Math.floor(px / DATA.tilePx);
  const tile = tilesByRC.get(row + ',' + col);
  if (tile) {
    badge.style.display = 'block';
    badge.innerHTML = 'tile ' + row + ',' + col + ' &mdash; ' +
      tile.cloud_cover_percent.toFixed(4) + '% &mdash; ' +
      '<span class="' + (tile.valid ? 'valid">valid' : 'invalid">invalid') + '</span>' +
      '<br><span class="hint">' + tile.min_latitude.toFixed(4) + ', ' + tile.min_longitude.toFixed(4) +
      ' → ' + tile.max_latitude.toFixed(4) + ', ' + tile.max_longitude.toFixed(4) + '</span>';
  } else {
    badge.style.display = 'none';
  }
}

// Precompute boolean membership arrays once, from the loaded layer images,
// so hovering doesn't decode a PNG's alpha channel every mousemove.
let maskAt = { opaque: null, cirrus: null, snow: null };
function extractAlphaMask(img) {
  const c = document.createElement('canvas');
  c.width = N; c.height = N;
  const cx = c.getContext('2d');
  cx.drawImage(img, 0, 0);
  const data = cx.getImageData(0, 0, N, N).data;
  const out = new Uint8Array(N * N);
  for (let i = 0, p = 0; i < data.length; i += 4, p++) out[p] = data[i + 3] > 0 ? 1 : 0;
  return out;
}
document.getElementById('resetView').addEventListener('click', () => {
  state._initialised = false;
  resizeCanvas();
});

function wireToggle(id, key) {
  document.getElementById(id).addEventListener('change', (e) => {
    state.show[key] = e.target.checked;
    composite(); draw();
  });
}
wireToggle('toggleBase', 'base');
wireToggle('toggleOpaque', 'opaque');
wireToggle('toggleCirrus', 'cirrus');
wireToggle('toggleSnow', 'snow');
wireToggle('toggleGrid', 'grid');
wireToggle('toggleTint', 'tint');
document.getElementById('opacity').addEventListener('input', (e) => {
  state.opacity = e.target.value / 100;
  composite(); draw();
});

window.addEventListener('resize', resizeCanvas);

Promise.all([
  loadImage(DATA.base), loadImage(DATA.layers.opaque),
  loadImage(DATA.layers.cirrus), loadImage(DATA.layers.snow),
]).then(([base, opaque, cirrus, snow]) => {
  images = { base, opaque, cirrus, snow };
  maskAt.opaque = extractAlphaMask(opaque);
  maskAt.cirrus = extractAlphaMask(cirrus);
  maskAt.snow = extractAlphaMask(snow);
  composite();
  resizeCanvas();
});
</script>
</body>
</html>
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build a self-contained HTML viewer for the esa backend's cloud mask."
    )
    parser.add_argument("--safe-dir", required=True, help="path to the .SAFE product folder")
    parser.add_argument("--out", required=True, help="where to write the HTML file")
    parser.add_argument(
        "--run", default=None,
        help="an existing esa run folder (with report.csv) to cross-check tile stats against",
    )
    args = parser.parse_args(argv)

    try:
        html = build_viewer_html(args.safe_dir, args.run)
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
    size_mb = len(html.encode("utf8")) / 1e6
    print("%s  (%.1f MB)" % (args.out, size_mb))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
