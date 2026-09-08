/**
 * regions.js — a direct port of build_region_mask() and its helpers from
 * capture_prototype.py. Read it side by side with the Python; the structure is
 * intentionally identical.
 *
 *   FRONT (GROUP_3) = centre strip UNION the skin-mask forehead (to the hairline)
 *   CHEEK (GROUP_1/2) = face oval on one side of the nose, MINUS the centre
 *                       strip, cut off at the eyebrow line so it holds no forehead
 *   all regions      = MINUS dilated eye and lip rings
 */

import * as M from './mask.js';
import * as L from './landmarks.js';
import {
  HOLE_DILATE_PX, INTERSECT_SKIN_ALL_REGIONS, FOREHEAD_BAND_FRAC, scaleRadius,
} from './config.js';

/** Landmark ids -> flat pixel coordinates [x0,y0,x1,y1,...]. */
export function pointsFor(landmarks, w, h, ids) {
  const out = new Float64Array(ids.length * 2);
  for (let i = 0; i < ids.length; i++) {
    const lm = landmarks[ids[i]];
    out[i * 2] = lm.x * w;
    out[i * 2 + 1] = lm.y * h;
  }
  return out;
}

/** The eyebrow line in pixels — the cut that strips forehead off the cheeks. */
export function browY(landmarks, h) {
  let min = Infinity;
  for (const id of L.BROW_IDS) {
    const y = landmarks[id].y * h;
    if (y < min) min = y;
  }
  return Math.round(min);   // highest brow point (smallest y)
}

// Three scratch masks, reused across calls. buildRegionMask is on the hot path
// and its intermediates never escape it; allocating them per frame was ~2 MB of
// garbage every frame, which a phone notices even though a laptop does not.
const scratchPool = new Map();

function scratch(w, h) {
  const key = `${w}x${h}`;
  let s = scratchPool.get(key);
  if (!s) {
    s = { oval: M.createMask(w, h), front: M.createMask(w, h), aux: M.createMask(w, h) };
    scratchPool.set(key, s);
  }
  return s;
}

/** Outer silhouette. Fills `out` if given, otherwise allocates. */
export function faceOvalMask(landmarks, w, h, out) {
  const m = out || M.createMask(w, h);
  if (out) out.fill(0);
  M.fillPoly(m, w, h, pointsFor(landmarks, w, h, L.FACE_OVAL));
  return m;
}

/** Filled centre strip: nose / mouth / chin, topped at brow level. */
export function frontStripMask(landmarks, w, h, oval, out) {
  const m = out || M.createMask(w, h);
  if (out) out.fill(0);
  M.fillPoly(m, w, h, pointsFor(landmarks, w, h, L.FRONT_STRIP));
  M.andInto(m, oval || faceOvalMask(landmarks, w, h));
  return m;
}

/**
 * The FRONT forehead, taken from the skin mask rather than the mesh:
 *   - skin pixels ABOVE the eyebrow line (so, up to the real hairline),
 *   - between the left and right temple x (ears and side hair drop out),
 *   - largest connected blob (a raised hand cannot leak in).
 * Leaves `out` empty when no skin mask is available — the mesh-only fallback,
 * same as Python.
 */
export function frontForeheadMask(landmarks, w, h, skinMask, out) {
  const m = out || M.createMask(w, h);
  if (out) out.fill(0);
  if (!skinMask) return m;

  const cut = Math.max(1, browY(landmarks, h));
  const lx = landmarks[L.LEFT_TEMPLE].x * w;
  const rx = landmarks[L.RIGHT_TEMPLE].x * w;

  M.fillRect(m, w, h, Math.min(lx, rx), 0, Math.max(lx, rx), cut);
  M.andInto(m, skinMask);
  M.keepLargestComponent(m, w, h);
  return m;
}

/** Dilated eye + lip rings, punched out of every region. */
export function holesMask(landmarks, w, h, out) {
  const holes = out || M.createMask(w, h);
  if (out) out.fill(0);
  for (const ring of L.HOLE_RINGS) {
    M.fillPoly(holes, w, h, pointsFor(landmarks, w, h, ring));
  }
  M.dilateDisc(holes, w, h, scaleRadius(HOLE_DILATE_PX, w, h));
  return holes;
}

/**
 * Single-channel mask, 255 inside the region for `group`.
 *
 * One deviation from the Python, and it is deliberate. Python ran the
 * segmenter only for the front shot, so a cheek's pixel count was pure
 * geometry — hair, beard and shadow inside the face oval all counted as
 * "skin". Here every region is intersected with the real face-skin mask when
 * one is supplied, so all three counts mean the same thing and the acne ratio
 * stays comparable across regions. Set INTERSECT_SKIN_ALL_REGIONS = false in
 * config.js to reproduce the original numbers exactly.
 *
 * Note the ordering: the largest-component filter runs on the *geometric*
 * mask, before the skin intersection. If the segmenter momentarily cuts the
 * strip in two (a hard shadow beside the nose, say), we lose a few pixels
 * rather than half the region.
 */
export function buildRegionMask(landmarks, w, h, group, skinMask, stats = null) {
  const s = scratch(w, h);
  const mask = M.createMask(w, h);       // the only buffer that escapes

  const oval = faceOvalMask(landmarks, w, h, s.oval);
  const front = frontStripMask(landmarks, w, h, oval, s.front);

  if (group === 'GROUP_3') {
    mask.set(front);
    M.orInto(mask, frontForeheadMask(landmarks, w, h, skinMask, s.aux));
  } else {
    // cheek = face half, minus the front strip, minus the forehead
    mask.set(oval);
    M.keepHalf(mask, w, h, landmarks[L.NOSE_TIP].x * w, group === 'GROUP_1' ? 'right' : 'left');
    M.andNotInto(mask, front);
    M.zeroAbove(mask, w, h, browY(landmarks, h));
  }

  M.andNotInto(mask, holesMask(landmarks, w, h, s.aux));
  M.keepLargestComponent(mask, w, h);

  // Area before the skin intersection, so the caller can tell how much of the
  // region the segmenter refused to call skin. That difference is an
  // obstruction: glasses, a hand, hair swept across the face.
  if (stats) stats.geometricPx = M.countNonZero(mask);

  if (skinMask && INTERSECT_SKIN_ALL_REGIONS) M.andInto(mask, skinMask);

  if (stats) {
    stats.skinPx = M.countNonZero(mask);
    stats.coverage = stats.geometricPx > 0 ? stats.skinPx / stats.geometricPx : 0;
  }
  return mask;
}

/**
 * How much of the forehead is bare skin rather than hair.
 *
 * Measured on a band immediately above the brow line, narrowed to the middle
 * 60% of the temple span so the hairline at the sides does not count against a
 * perfectly clear forehead. Returns 1 when there is no skin mask to judge
 * from — an unknown must not block the shot.
 */
export function foreheadCoverage(landmarks, w, h, skinMask) {
  if (!skinMask) return 1;

  const brow = browY(landmarks, h);
  const faceH = Math.abs(landmarks[L.CHIN].y - landmarks[L.FOREHEAD_CENTER].y) * h;
  const top = Math.max(0, Math.round(brow - faceH * FOREHEAD_BAND_FRAC));
  const bottom = Math.min(h, brow);
  if (bottom <= top) return 1;

  const lx = landmarks[L.LEFT_TEMPLE].x * w;
  const rx = landmarks[L.RIGHT_TEMPLE].x * w;
  const cx = (lx + rx) / 2;
  const half = (Math.abs(rx - lx) / 2) * 0.6;
  const x0 = Math.max(0, Math.round(cx - half));
  const x1 = Math.min(w, Math.round(cx + half));
  if (x1 <= x0) return 1;

  let skin = 0;
  let total = 0;
  for (let y = top; y < bottom; y++) {
    const row = y * w;
    for (let x = x0; x < x1; x++) {
      total++;
      if (skinMask[row + x]) skin++;
    }
  }
  return total ? skin / total : 1;
}

/**
 * Number of face-skin pixels inside the region.
 *
 * The raw count scales with distance to the camera, so it is not comparable
 * across people or sessions on its own. It is the denominator of the
 * acne_pixels / skin_pixels ratio that cancels that scale out later.
 */
export function countSkinPixels(mask) {
  return M.countNonZero(mask);
}

/** Tight bounding box over every landmark, in pixels. */
export function faceBounds(landmarks, w, h) {
  let x0 = Infinity;
  let y0 = Infinity;
  let x1 = -Infinity;
  let y1 = -Infinity;
  for (const lm of landmarks) {
    const x = lm.x * w;
    const y = lm.y * h;
    if (x < x0) x0 = x;
    if (x > x1) x1 = x;
    if (y < y0) y0 = y;
    if (y > y1) y1 = y;
  }
  return { x0, y0, x1, y1, width: x1 - x0, height: y1 - y0 };
}

/** Grow a box by `frac` of its own size and clamp to the frame. */
export function padBounds(b, w, h, frac) {
  const pw = b.width * frac;
  const ph = b.height * frac;
  return {
    x0: Math.max(0, Math.round(b.x0 - pw)),
    y0: Math.max(0, Math.round(b.y0 - ph)),
    x1: Math.min(w, Math.round(b.x1 + pw)),
    y1: Math.min(h, Math.round(b.y1 + ph)),
  };
}
