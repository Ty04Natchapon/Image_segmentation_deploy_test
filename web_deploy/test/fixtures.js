/**
 * A synthetic 478-point face, good enough to exercise the region geometry
 * without a camera. Landmarks are laid out from the same ordering the real
 * arrays carry — the oval walks the silhouette clockwise from the forehead,
 * the front strip walks down its right edge and back up its left — so the
 * polygons close correctly even though the anatomy is a cartoon.
 */

import * as L from '../src/landmarks.js';

export const CX = 0.5;
export const CY = 0.5;
export const RX = 0.22;
export const RY = 0.30;
export const STRIP_HALF = 0.055;
export const BROW_Y = CY - 0.13;

export function makeFace() {
  const lm = Array.from({ length: 478 }, () => ({ x: CX, y: CY, z: 0 }));
  const set = (i, x, y) => { lm[i] = { x, y, z: 0 }; };

  // Front strip: right edge top -> bottom, then left edge bottom -> top.
  const stripTop = CY - 0.16;
  const stripBottom = CY + 0.26;
  L.FRONT_RIGHT_EDGE.forEach((id, i) => {
    const t = i / (L.FRONT_RIGHT_EDGE.length - 1);
    set(id, CX + STRIP_HALF, stripTop + t * (stripBottom - stripTop));
  });
  L.FRONT_LEFT_EDGE.forEach((id, i) => {
    const t = i / (L.FRONT_LEFT_EDGE.length - 1);
    set(id, CX - STRIP_HALF, stripBottom - t * (stripBottom - stripTop));
  });

  // Face oval, walked clockwise from the top. Written after the strip so the
  // shared vertices (10 = forehead centre, 152 = chin) land on the silhouette.
  L.FACE_OVAL.forEach((id, i) => {
    const a = -Math.PI / 2 + (i / L.FACE_OVAL.length) * Math.PI * 2;
    set(id, CX + RX * Math.cos(a), CY + RY * Math.sin(a));
  });

  // Exclusion rings.
  const ring = (ids, ox, oy, r) => ids.forEach((id, i) => {
    const a = (i / ids.length) * Math.PI * 2;
    set(id, ox + r * Math.cos(a), oy + r * 0.6 * Math.sin(a));
  });
  ring(L.RIGHT_EYE_RING, CX - 0.10, CY - 0.06, 0.045);
  ring(L.LEFT_EYE_RING, CX + 0.10, CY - 0.06, 0.045);
  ring(L.LIPS_RING, CX, CY + 0.13, 0.05);

  // Key points last, so nothing above overwrites them.
  set(L.NOSE_TIP, CX, CY + 0.02);
  set(L.LEFT_TEMPLE, CX - RX, CY);
  set(L.RIGHT_TEMPLE, CX + RX, CY);
  set(L.CHIN, CX, CY + RY);
  for (const id of L.BROW_IDS) set(id, lm[id].x, BROW_Y);

  return lm;
}

/** A face-skin mask: the whole oval, plus a forehead band up near the hairline. */
export function makeSkinMask(w, h, M) {
  const skin = M.createMask(w, h);
  const pts = [];
  for (let i = 0; i < 64; i++) {
    const a = (i / 64) * Math.PI * 2;
    pts.push((CX + RX * 1.05 * Math.cos(a)) * w, (CY + RY * 1.05 * Math.sin(a)) * h);
  }
  M.fillPoly(skin, w, h, pts);
  M.fillRect(skin, w, h, (CX - RX) * w, 0.08 * h, (CX + RX) * w, BROW_Y * h);
  return skin;
}
