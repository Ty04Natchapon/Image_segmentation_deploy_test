/**
 * The occlusion gate.
 *
 * Neither version had one, so a fringe over the forehead or glasses across a
 * cheek was captured happily — the mask just shrank around the obstruction and
 * the pixel count quietly dropped, which is worse than refusing the shot,
 * because the bad data looks exactly like good data downstream.
 */

import test from 'node:test';
import assert from 'node:assert/strict';
import * as M from '../src/mask.js';
import * as R from '../src/regions.js';
import { checkOcclusion } from '../src/pose.js';
import { MIN_SKIN_COVERAGE, MIN_FOREHEAD_COVERAGE } from '../src/config.js';
import { makeFace, makeSkinMask, CX, RX, BROW_Y } from './fixtures.js';

const W = 256, H = 256;
const face = makeFace();

/** The clear-forehead skin mask, with a fringe painted over the brow band. */
function withFringe(w, h) {
  const skin = makeSkinMask(w, h, M);
  const fringe = M.createMask(w, h);
  M.fillRect(fringe, w, h,
    (CX - RX) * w, (BROW_Y - 0.16) * h,
    (CX + RX) * w, (BROW_Y + 0.01) * h);
  M.andNotInto(skin, fringe);     // hair is not face-skin
  return skin;
}

test('a clear forehead scores high, a fringe scores low', () => {
  const clear = R.foreheadCoverage(face, W, H, makeSkinMask(W, H, M));
  const covered = R.foreheadCoverage(face, W, H, withFringe(W, H));

  assert.ok(clear > 0.9, `clear forehead should be near 1, got ${clear.toFixed(2)}`);
  assert.ok(covered < MIN_FOREHEAD_COVERAGE,
    `fringe should fall below the threshold, got ${covered.toFixed(2)}`);
});

test('no skin mask means no verdict, never a block', () => {
  // The segmenter fails occasionally. An unknown must not stop the shot.
  assert.equal(R.foreheadCoverage(face, W, H, null), 1);
  assert.equal(checkOcclusion('GROUP_3', 1, 1).ok, true);
});

test('buildRegionMask reports how much of the region is really skin', () => {
  const stats = {};
  R.buildRegionMask(face, W, H, 'GROUP_2', makeSkinMask(W, H, M), stats);

  assert.ok(stats.geometricPx > 0, 'geometric area should be measured');
  assert.ok(stats.skinPx <= stats.geometricPx, 'skin area cannot exceed geometry');
  assert.ok(stats.coverage > 0.9,
    `an unobstructed cheek should be almost all skin, got ${stats.coverage.toFixed(2)}`);
});

test('something covering a cheek drops coverage below the threshold', () => {
  // A band across the cheek, standing in for glasses or a hand.
  const skin = makeSkinMask(W, H, M);
  const bar = M.createMask(W, H);
  M.fillRect(bar, W, H, 0, 0.42 * H, W, 0.60 * H);
  M.andNotInto(skin, bar);

  const stats = {};
  R.buildRegionMask(face, W, H, 'GROUP_2', skin, stats);
  assert.ok(stats.coverage < MIN_SKIN_COVERAGE,
    `obstructed cheek should fail the gate, got ${stats.coverage.toFixed(2)}`);
});

test('the gate names what to move, and only checks the forehead up front', () => {
  assert.match(checkOcclusion('GROUP_3', 0.5, 1).msg, /hair or glasses/);
  assert.match(checkOcclusion('GROUP_3', 1, 0.2).msg, /forehead/);

  // A cheek shot does not care about the forehead: it is cut off at the brow.
  assert.equal(checkOcclusion('GROUP_1', 1, 0.0).ok, true);
});
