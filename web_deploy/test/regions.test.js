import test from 'node:test';
import assert from 'node:assert/strict';
import * as M from '../src/mask.js';
import * as R from '../src/regions.js';
import { makeFace, makeSkinMask, CX, CY, BROW_Y, STRIP_HALF } from './fixtures.js';

const W = 256, H = 256;
const face = makeFace();
const at = (m, x, y, w = W) => m[Math.round(y * H) * w + Math.round(x * w)];

function countAbove(mask, w, h, yCut) {
  let n = 0;
  for (let i = 0; i < yCut * w; i++) if (mask[i]) n++;
  return n;
}

test('every group produces a non-empty region', () => {
  for (const g of ['GROUP_1', 'GROUP_2', 'GROUP_3']) {
    const m = R.buildRegionMask(face, W, H, g, null);
    assert.ok(R.countSkinPixels(m) > 500, `${g} should not be empty`);
  }
});

test('cheek regions contain no forehead at all', () => {
  const brow = R.browY(face, H);
  for (const g of ['GROUP_1', 'GROUP_2']) {
    const m = R.buildRegionMask(face, W, H, g, null);
    assert.equal(countAbove(m, W, H, brow), 0, `${g} leaked above the brow line`);
  }
});

test('cheeks sit on their own side of the nose and do not overlap', () => {
  const noseX = Math.round(face[1].x * W);
  const right = R.buildRegionMask(face, W, H, 'GROUP_1', null);
  const left = R.buildRegionMask(face, W, H, 'GROUP_2', null);

  for (let y = 0; y < H; y++) {
    for (let x = 0; x < noseX; x++) {
      assert.equal(right[y * W + x], 0, 'GROUP_1 crossed to the left of the nose');
    }
    for (let x = noseX; x < W; x++) {
      assert.equal(left[y * W + x], 0, 'GROUP_2 crossed to the right of the nose');
    }
  }
});

test('the front strip and the cheeks are disjoint', () => {
  const front = R.buildRegionMask(face, W, H, 'GROUP_3', null);
  for (const g of ['GROUP_1', 'GROUP_2']) {
    const cheek = R.buildRegionMask(face, W, H, g, null);
    const both = cheek.slice();
    M.andInto(both, front);
    assert.equal(M.countNonZero(both), 0, `${g} overlaps the front strip`);
  }
});

test('eyes and mouth are punched out of the regions that contain them', () => {
  const front = R.buildRegionMask(face, W, H, 'GROUP_3', null);
  assert.equal(at(front, CX, CY + 0.13), 0, 'mouth still inside the front strip');

  const left = R.buildRegionMask(face, W, H, 'GROUP_2', null);
  assert.equal(at(left, CX - 0.10, CY - 0.06), 0, 'eye still inside the cheek');
});

test('the skin mask extends the front forehead to the hairline', () => {
  const brow = R.browY(face, H);
  const skin = makeSkinMask(W, H, M);

  const meshOnly = R.buildRegionMask(face, W, H, 'GROUP_3', null);
  const withSkin = R.buildRegionMask(face, W, H, 'GROUP_3', skin);

  const gained = countAbove(withSkin, W, H, brow) - countAbove(meshOnly, W, H, brow);
  assert.ok(gained > 1000, `forehead barely grew (${gained} px)`);

  // and it reaches wider than the strip, which is the whole point
  assert.equal(at(withSkin, CX + STRIP_HALF * 2.2, BROW_Y - 0.06), 255);
});

test('region area scales with resolution, so the radii scale too', () => {
  // A pixel count that did not track resolution would mean the live preview and
  // the full-resolution capture disagreed about the same face.
  const small = R.countSkinPixels(R.buildRegionMask(face, 256, 256, 'GROUP_3', null));
  const large = R.countSkinPixels(R.buildRegionMask(face, 512, 512, 'GROUP_3', null));
  const ratio = large / small;
  assert.ok(ratio > 3.8 && ratio < 4.2, `expected ~4x area, got ${ratio.toFixed(3)}x`);
});

test('faceBounds and padBounds stay inside the frame', () => {
  const b = R.faceBounds(face, W, H);
  assert.ok(Math.abs(b.width - 2 * 0.22 * W) < 2);
  const p = R.padBounds(b, W, H, 0.25);
  assert.ok(p.x0 >= 0 && p.y0 >= 0 && p.x1 <= W && p.y1 <= H);
  assert.ok(p.x1 - p.x0 > b.width);
});
