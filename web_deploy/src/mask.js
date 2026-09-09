/**
 * mask.js — the binary-raster toolbox that replaces OpenCV.
 *
 * A mask is a flat `Uint8Array(w * h)` holding 0 or 255, laid out row-major,
 * exactly like a single-channel cv2 Mat. Every function here mirrors the cv2
 * call it stands in for, so regions.js reads almost line-for-line like the
 * Python original.
 *
 * Deliberately dependency-free: pulling in opencv.js would mean ~9 MB of WASM
 * for the six operations below, which is a poor trade on a phone.
 */

export function createMask(w, h) {
  return new Uint8Array(w * h);
}

/** cv2.fillPoly — even-odd scanline fill. `pts` is a flat [x0,y0,x1,y1,...]. */
export function fillPoly(mask, w, h, pts) {
  const n = pts.length / 2;
  if (n < 3) return mask;

  let minY = Infinity;
  let maxY = -Infinity;
  for (let i = 1; i < pts.length; i += 2) {
    if (pts[i] < minY) minY = pts[i];
    if (pts[i] > maxY) maxY = pts[i];
  }
  const yStart = Math.max(0, Math.floor(minY));
  const yEnd = Math.min(h - 1, Math.ceil(maxY));

  const xs = [];
  for (let y = yStart; y <= yEnd; y++) {
    const yc = y + 0.5;
    xs.length = 0;
    for (let i = 0, j = n - 1; i < n; j = i++) {
      const y1 = pts[j * 2 + 1];
      const y2 = pts[i * 2 + 1];
      // Half-open test, so a vertex sitting on the scanline is counted once.
      if ((y1 <= yc && y2 > yc) || (y2 <= yc && y1 > yc)) {
        const t = (yc - y1) / (y2 - y1);
        xs.push(pts[j * 2] + t * (pts[i * 2] - pts[j * 2]));
      }
    }
    if (xs.length < 2) continue;
    xs.sort((a, b) => a - b);
    const row = y * w;
    for (let k = 0; k + 1 < xs.length; k += 2) {
      const x0 = Math.max(0, Math.ceil(xs[k] - 0.5));
      const x1 = Math.min(w - 1, Math.floor(xs[k + 1] - 0.5));
      for (let x = x0; x <= x1; x++) mask[row + x] = 255;
    }
  }
  return mask;
}

/** cv2.bitwise_and(dst, src) in place. */
export function andInto(dst, src) {
  for (let i = 0; i < dst.length; i++) if (!src[i]) dst[i] = 0;
  return dst;
}

/** cv2.bitwise_or(dst, src) in place. */
export function orInto(dst, src) {
  for (let i = 0; i < dst.length; i++) if (src[i]) dst[i] = 255;
  return dst;
}

/** cv2.bitwise_and(dst, cv2.bitwise_not(src)) in place — punches holes. */
export function andNotInto(dst, src) {
  for (let i = 0; i < dst.length; i++) if (src[i]) dst[i] = 0;
  return dst;
}

/** Zero every row above `y` — the Python `cheek[0:brow_y, :] = 0`. */
export function zeroAbove(mask, w, h, y) {
  const cut = Math.max(0, Math.min(h, y));
  mask.fill(0, 0, cut * w);
  return mask;
}

/** Keep only one side of a vertical line; `side` is 'left' or 'right'. */
export function keepHalf(mask, w, h, x, side) {
  const cx = Math.max(0, Math.min(w, Math.round(x)));
  for (let y = 0; y < h; y++) {
    const row = y * w;
    if (side === 'right') mask.fill(0, row, row + cx);
    else mask.fill(0, row + cx, row + w);
  }
  return mask;
}

/** Fill a rectangular band with 255 (the forehead search band). */
export function fillRect(mask, w, h, x0, y0, x1, y1) {
  const xa = Math.max(0, Math.min(w, Math.round(x0)));
  const xb = Math.max(0, Math.min(w, Math.round(x1)));
  const ya = Math.max(0, Math.min(h, Math.round(y0)));
  const yb = Math.max(0, Math.min(h, Math.round(y1)));
  for (let y = ya; y < yb; y++) mask.fill(255, y * w + xa, y * w + xb);
  return mask;
}

// --- Scratch pools ---------------------------------------------------------
// Everything below runs 30 times a second on two different frame sizes. These
// buffers are internal and never escape, so they are cached per size rather
// than reallocated: at 512x288 the labelling pass alone was throwing away
// 1.2 MB per call, and a phone's GC notices that even when a laptop does not.

const u8Pool = new Map();
const labelPool = new Map();

function borrowU8(n, slot = 0) {
  // `slot` keeps simultaneously-live buffers apart — openSquare holds two at
  // once, and handing it the same array twice would corrupt the filter.
  const key = `${n}:${slot}`;
  let buf = u8Pool.get(key);
  if (!buf) {
    buf = new Uint8Array(n);
    u8Pool.set(key, buf);
  }
  return buf;
}

function borrowLabels(n) {
  let pair = labelPool.get(n);
  if (!pair) {
    pair = { labels: new Int32Array(n), stack: new Int32Array(n) };
    labelPool.set(n, pair);
  } else {
    pair.labels.fill(0);
  }
  return pair;
}

// --- Morphology ------------------------------------------------------------

const discCache = new Map();

function discOffsets(r) {
  let d = discCache.get(r);
  if (d) return d;
  const out = [];
  for (let dy = -r; dy <= r; dy++) {
    for (let dx = -r; dx <= r; dx++) {
      if (dx * dx + dy * dy <= r * r) out.push(dx, dy);
    }
  }
  d = Int32Array.from(out);
  discCache.set(r, d);
  return d;
}

/**
 * cv2.dilate with an ELLIPSE structuring element of radius r.
 *
 * Implemented by stamping a disc at every set pixel. That is exact rather than
 * a square approximation, and cheap here because the only thing we dilate is
 * the eye/lip ring mask, which is sparse.
 */
export function dilateDisc(mask, w, h, r) {
  if (r <= 0) return mask;
  const off = discOffsets(r);
  const src = borrowU8(w * h, 0);
  src.set(mask);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      if (!src[y * w + x]) continue;
      for (let k = 0; k < off.length; k += 2) {
        const nx = x + off[k];
        const ny = y + off[k + 1];
        if (nx < 0 || nx >= w || ny < 0 || ny >= h) continue;
        mask[ny * w + nx] = 255;
      }
    }
  }
  return mask;
}

// Separable min/max filters over a (2r+1) square. Used only for the skin-mask
// smoothing pass, where the mask is dense and stamping would be wasteful.
function rankPass(src, dst, w, h, r, wantMax) {
  const tmp = borrowU8(w * h, 2);
  for (let y = 0; y < h; y++) {
    const row = y * w;
    for (let x = 0; x < w; x++) {
      let v = wantMax ? 0 : 255;
      const x0 = Math.max(0, x - r);
      const x1 = Math.min(w - 1, x + r);
      for (let i = x0; i <= x1; i++) {
        const s = src[row + i];
        if (wantMax ? s > v : s < v) v = s;
      }
      tmp[row + x] = v;
    }
  }
  for (let y = 0; y < h; y++) {
    const y0 = Math.max(0, y - r);
    const y1 = Math.min(h - 1, y + r);
    for (let x = 0; x < w; x++) {
      let v = wantMax ? 0 : 255;
      for (let j = y0; j <= y1; j++) {
        const s = tmp[j * w + x];
        if (wantMax ? s > v : s < v) v = s;
      }
      dst[y * w + x] = v;
    }
  }
  return dst;
}

/**
 * cv2.morphologyEx(MORPH_OPEN) — erode then dilate, knocking the speckle off
 * the segmenter's jittery edge. Uses a square element rather than the Python
 * 5x5 ellipse; at this radius that is a couple of corner pixels on a mask that
 * is about to be intersected with a polygon anyway.
 */
export function openSquare(mask, w, h, r) {
  if (r <= 0) return mask;
  const a = borrowU8(w * h, 1);
  rankPass(mask, a, w, h, r, false);   // erode
  rankPass(a, mask, w, h, r, true);    // dilate
  return mask;
}

/**
 * cv2.connectedComponentsWithStats followed by "keep the largest blob".
 * Iterative 8-connected flood fill — recursion would blow the JS stack on a
 * full-resolution face. Returns the surviving blob's area.
 */
export function keepLargestComponent(mask, w, h) {
  const n = w * h;
  const { labels, stack } = borrowLabels(n);   // 0 = unvisited
  let current = 0;
  let bestLabel = 0;
  let bestArea = 0;

  for (let seed = 0; seed < n; seed++) {
    if (!mask[seed] || labels[seed]) continue;
    current++;
    let area = 0;
    let sp = 0;
    stack[sp++] = seed;
    labels[seed] = current;

    while (sp > 0) {
      const p = stack[--sp];
      area++;
      const px = p % w;
      const py = (p / w) | 0;
      const x0 = px > 0 ? px - 1 : 0;
      const x1 = px < w - 1 ? px + 1 : w - 1;
      const y0 = py > 0 ? py - 1 : 0;
      const y1 = py < h - 1 ? py + 1 : h - 1;
      for (let y = y0; y <= y1; y++) {
        for (let x = x0; x <= x1; x++) {
          const q = y * w + x;
          if (mask[q] && !labels[q]) {
            labels[q] = current;
            stack[sp++] = q;
          }
        }
      }
    }
    if (area > bestArea) {
      bestArea = area;
      bestLabel = current;
    }
  }

  if (bestLabel === 0) return 0;
  for (let i = 0; i < n; i++) if (labels[i] !== bestLabel) mask[i] = 0;
  return bestArea;
}

/** cv2.countNonZero. */
export function countNonZero(mask) {
  let c = 0;
  for (let i = 0; i < mask.length; i++) if (mask[i]) c++;
  return c;
}

/**
 * Turn the segmenter's class-index map into a binary mask at the target size —
 * cv2.resize(..., INTER_NEAREST) and the `== FACE_SKIN_CLASS` test folded into
 * one pass. Straight copy when the sizes already agree.
 */
export function classMapToBinary(src, sw, sh, dst, dw, dh, classValue) {
  if (sw === dw && sh === dh) {
    for (let i = 0; i < dst.length; i++) dst[i] = src[i] === classValue ? 255 : 0;
    return dst;
  }
  const xr = sw / dw;
  const yr = sh / dh;
  for (let y = 0; y < dh; y++) {
    const sy = Math.min(sh - 1, (y * yr) | 0);
    const srow = sy * sw;
    const drow = y * dw;
    for (let x = 0; x < dw; x++) {
      const sx = Math.min(sw - 1, (x * xr) | 0);
      dst[drow + x] = src[srow + sx] === classValue ? 255 : 0;
    }
  }
  return dst;
}

/**
 * Paint a mask into RGBA pixels for display: translucent inside, solid along
 * the border. This replaces findContours + drawContours — we never needed the
 * contour vertices themselves, only the picture of them.
 */
export function maskToImageData(mask, w, h, rgb, target = null, fillAlpha = 56, edgeAlpha = 235) {
  const img = target || new ImageData(w, h);
  const px = img.data;
  if (target) px.fill(0);
  const [r, g, b] = rgb;
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const i = y * w + x;
      if (!mask[i]) continue;
      const edge =
        x === 0 || y === 0 || x === w - 1 || y === h - 1 ||
        !mask[i - 1] || !mask[i + 1] || !mask[i - w] || !mask[i + w];
      const o = i * 4;
      px[o] = r;
      px[o + 1] = g;
      px[o + 2] = b;
      px[o + 3] = edge ? edgeAlpha : fillAlpha;
    }
  }
  return img;
}

/** Mask -> opaque black/white ImageData, for the saved mask PNG. */
export function maskToBinaryImageData(mask, w, h) {
  const img = new ImageData(w, h);
  const px = img.data;
  for (let i = 0; i < mask.length; i++) {
    const v = mask[i] ? 255 : 0;
    const o = i * 4;
    px[o] = v;
    px[o + 1] = v;
    px[o + 2] = v;
    px[o + 3] = 255;
  }
  return img;
}

/**
 * Variance of the Laplacian — the standard cheap focus measure.
 *
 * A sharp image has strong second derivatives at edges, so their variance is
 * high; blur flattens them and the variance collapses. It costs one pass over
 * the pixels and needs no model.
 *
 * Read it as a relative number, not an absolute one: it scales with contrast
 * and resolution, so a threshold tuned on one device and framing does not
 * transfer. Comparing candidate frames from the same moment is what it is
 * genuinely reliable for.
 */
export function laplacianVariance(gray, w, h) {
  if (w < 3 || h < 3) return 0;
  let sum = 0;
  let sumSq = 0;
  let n = 0;
  for (let y = 1; y < h - 1; y++) {
    const row = y * w;
    for (let x = 1; x < w - 1; x++) {
      const i = row + x;
      const lap = 4 * gray[i] - gray[i - 1] - gray[i + 1] - gray[i - w] - gray[i + w];
      sum += lap;
      sumSq += lap * lap;
      n++;
    }
  }
  if (!n) return 0;
  const mean = sum / n;
  return Math.max(0, sumSq / n - mean * mean);
}
