import test from 'node:test';
import assert from 'node:assert/strict';
import * as M from '../src/mask.js';

const W = 40, H = 40;
const at = (m, x, y) => m[y * W + x];
const rect = (x0, y0, x1, y1) => [x0, y0, x1, y0, x1, y1, x0, y1];

test('fillPoly covers exactly the pixel centres inside a rectangle', () => {
  const m = M.createMask(W, H);
  M.fillPoly(m, W, H, rect(10, 10, 20, 20));
  assert.equal(M.countNonZero(m), 100);      // pixel centres 10.5..19.5 in both axes
  assert.equal(at(m, 10, 10), 255);
  assert.equal(at(m, 19, 19), 255);
  assert.equal(at(m, 20, 20), 0);
  assert.equal(at(m, 9, 15), 0);
});

test('fillPoly handles a non-convex polygon (even-odd spans)', () => {
  // A "C": a 20x20 block with a notch bitten out of its right-middle.
  const m = M.createMask(W, H);
  M.fillPoly(m, W, H, [
    5, 5, 25, 5, 25, 11, 15, 11, 15, 19, 25, 19, 25, 25, 5, 25,
  ]);
  assert.equal(at(m, 20, 8), 255, 'top arm filled');
  assert.equal(at(m, 20, 22), 255, 'bottom arm filled');
  assert.equal(at(m, 20, 15), 0, 'notch left empty');
  assert.equal(at(m, 10, 15), 255, 'spine filled');
});

test('boolean combinators match their cv2 counterparts', () => {
  const a = M.createMask(W, H);
  const b = M.createMask(W, H);
  M.fillPoly(a, W, H, rect(5, 5, 25, 25));
  M.fillPoly(b, W, H, rect(15, 15, 35, 35));

  const and = a.slice(); M.andInto(and, b);
  assert.equal(M.countNonZero(and), 100);

  const or = a.slice(); M.orInto(or, b);
  assert.equal(M.countNonZero(or), 400 + 400 - 100);

  const sub = a.slice(); M.andNotInto(sub, b);
  assert.equal(M.countNonZero(sub), 400 - 100);
  assert.equal(at(sub, 20, 20), 0);
  assert.equal(at(sub, 10, 10), 255);
});

test('zeroAbove and keepHalf cut where they say they do', () => {
  const m = M.createMask(W, H);
  m.fill(255);
  M.zeroAbove(m, W, H, 10);
  assert.equal(at(m, 5, 9), 0);
  assert.equal(at(m, 5, 10), 255);

  const r = M.createMask(W, H); r.fill(255);
  M.keepHalf(r, W, H, 20, 'right');
  assert.equal(at(r, 19, 5), 0);
  assert.equal(at(r, 20, 5), 255);

  const l = M.createMask(W, H); l.fill(255);
  M.keepHalf(l, W, H, 20, 'left');
  assert.equal(at(l, 19, 5), 255);
  assert.equal(at(l, 20, 5), 0);
});

test('dilateDisc grows by a true disc, not a square', () => {
  const m = M.createMask(W, H);
  m[20 * W + 20] = 255;
  M.dilateDisc(m, W, H, 3);
  assert.equal(at(m, 23, 20), 255, 'reaches r along the axis');
  assert.equal(at(m, 23, 23), 0, 'does not reach the square corner');
  assert.equal(M.countNonZero(m), 29);      // |{(dx,dy) : dx^2+dy^2 <= 9}|
});

test('openSquare removes speckle but keeps a solid blob', () => {
  const m = M.createMask(W, H);
  M.fillPoly(m, W, H, rect(10, 10, 25, 25));
  m[3 * W + 3] = 255;                        // isolated speckle
  M.openSquare(m, W, H, 2);
  assert.equal(at(m, 3, 3), 0, 'speckle erased');
  assert.equal(at(m, 17, 17), 255, 'blob interior survives');
});

test('keepLargestComponent drops every blob but the biggest', () => {
  const m = M.createMask(W, H);
  M.fillPoly(m, W, H, rect(2, 2, 8, 8));       // 36 px
  M.fillPoly(m, W, H, rect(20, 20, 35, 35));   // 225 px
  const area = M.keepLargestComponent(m, W, H);
  assert.equal(area, 225);
  assert.equal(M.countNonZero(m), 225);
  assert.equal(at(m, 5, 5), 0);
  assert.equal(at(m, 25, 25), 255);
});

test('keepLargestComponent joins blobs that touch only diagonally', () => {
  // 8-connectivity, matching cv2.connectedComponentsWithStats(connectivity=8).
  const m = M.createMask(W, H);
  m[10 * W + 10] = 255;
  m[11 * W + 11] = 255;
  assert.equal(M.keepLargestComponent(m, W, H), 2);
});

test('classMapToBinary selects the class and nearest-resizes', () => {
  const src = Uint8Array.from([0, 3,      // 2x2 class map, row 0
                               3, 1]);   //                row 1
  const dst = M.createMask(4, 4);
  M.classMapToBinary(src, 2, 2, dst, 4, 4, 3);
  // Each source pixel expands to a 2x2 block; only class 3 survives.
  assert.deepEqual(Array.from(dst.subarray(0, 4)), [0, 0, 255, 255], 'row 0');
  assert.deepEqual(Array.from(dst.subarray(4, 8)), [0, 0, 255, 255], 'row 1');
  assert.deepEqual(Array.from(dst.subarray(8, 12)), [255, 255, 0, 0], 'row 2');
  assert.deepEqual(Array.from(dst.subarray(12, 16)), [255, 255, 0, 0], 'row 3');
  assert.equal(M.countNonZero(dst), 8);

  // Same size in and out: no resampling, just the class test.
  const same = M.createMask(2, 2);
  M.classMapToBinary(src, 2, 2, same, 2, 2, 3);
  assert.deepEqual(Array.from(same), [0, 255, 255, 0]);
});

test('scratch pooling does not leak state between calls', () => {
  // Two different-sized calls interleaved must not corrupt each other.
  const a = M.createMask(W, H);
  M.fillPoly(a, W, H, rect(5, 5, 15, 15));
  M.keepLargestComponent(a, W, H);
  const first = M.countNonZero(a);

  const b = M.createMask(10, 10);
  M.fillPoly(b, 10, 10, [1, 1, 4, 1, 4, 4, 1, 4]);
  M.keepLargestComponent(b, 10, 10);

  const c = M.createMask(W, H);
  M.fillPoly(c, W, H, rect(5, 5, 15, 15));
  M.keepLargestComponent(c, W, H);
  assert.equal(M.countNonZero(c), first);
});
