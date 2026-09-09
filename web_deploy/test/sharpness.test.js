/**
 * Blur detection.
 *
 * A manual shutter is the intuitive fix for blurry auto-captures and the wrong
 * one — tapping the screen shakes the phone at the exact moment of capture.
 * Measuring the frame is the fix, and it improves auto-capture rather than
 * replacing it.
 */

import test from 'node:test';
import assert from 'node:assert/strict';
import { laplacianVariance } from '../src/mask.js';
import { checkSharpness, PeakTracker } from '../src/pose.js';
import { MIN_SHARPNESS, PEAK_SAMPLE_INTERVAL_MS } from '../src/config.js';

const W = 64, H = 64;

/** Hard-edged checkerboard: maximum high-frequency content. */
function sharpImage() {
  const g = new Uint8Array(W * H);
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      g[y * W + x] = ((x >> 2) + (y >> 2)) % 2 ? 230 : 25;
    }
  }
  return g;
}

/** The same image after a 3x3 box blur, once per pass. */
function blur(src, passes = 1) {
  let cur = src;
  for (let p = 0; p < passes; p++) {
    const out = new Uint8Array(W * H);
    for (let y = 1; y < H - 1; y++) {
      for (let x = 1; x < W - 1; x++) {
        let sum = 0;
        for (let dy = -1; dy <= 1; dy++) {
          for (let dx = -1; dx <= 1; dx++) sum += cur[(y + dy) * W + x + dx];
        }
        out[y * W + x] = (sum / 9) | 0;
      }
    }
    cur = out;
  }
  return cur;
}

test('blur collapses the Laplacian variance', () => {
  const sharp = laplacianVariance(sharpImage(), W, H);
  const soft = laplacianVariance(blur(sharpImage(), 1), W, H);
  const softer = laplacianVariance(blur(sharpImage(), 3), W, H);

  assert.ok(sharp > soft, `sharp ${sharp.toFixed(0)} should beat blurred ${soft.toFixed(0)}`);
  assert.ok(soft > softer, 'more blur should score lower still');
  assert.ok(softer < sharp / 10, 'heavy blur should be an order of magnitude lower');
});

test('a flat image has no detail to measure', () => {
  const flat = new Uint8Array(W * H).fill(128);
  assert.equal(laplacianVariance(flat, W, H), 0);
});

test('an image too small to sample scores zero rather than crashing', () => {
  assert.equal(laplacianVariance(new Uint8Array(4), 2, 2), 0);
});

test('an unmeasured frame is not treated as a blurred one', () => {
  // 0 means "the face box was too small to sample". Refusing on that would
  // block shots for a reason the user cannot act on.
  assert.equal(checkSharpness(0).ok, true);
  assert.equal(checkSharpness(MIN_SHARPNESS - 1).ok, false);
  assert.equal(checkSharpness(MIN_SHARPNESS).ok, true);
  assert.match(checkSharpness(1).msg, /steadier/);
});

test('the tracker keeps the sharpest of the window, not the middle', () => {
  const t = new PeakTracker();
  const dt = PEAK_SAMPLE_INTERVAL_MS;
  // A front peak: the middle sample is closest to ratio 1.0, so it fires.
  // The last sample is the sharpest, so it is the one kept.
  t.push(0, 0.9, 'GROUP_3', { sharpness: 40, name: 'a' });
  t.push(dt, 1.0, 'GROUP_3', { sharpness: 50, name: 'b' });
  const winner = t.push(dt * 2, 0.9, 'GROUP_3', { sharpness: 90, name: 'c' });

  assert.ok(winner, 'the peak should still fire');
  assert.equal(winner.name, 'c', 'the sharpest frame should win');
});

test('a blurred sample discards the window instead of being captured', () => {
  const t = new PeakTracker();
  const dt = PEAK_SAMPLE_INTERVAL_MS;
  t.push(0, 0.9, 'GROUP_3', { sharpness: 50, name: 'a' });
  t.push(dt, 1.0, 'GROUP_3', { sharpness: 50, name: 'b' });
  const winner = t.push(dt * 2, 0.9, 'GROUP_3', { sharpness: 1, name: 'blurred' });

  assert.equal(winner, null, 'a window containing a blurred frame must not fire');
});

test('unmeasured sharpness still allows a capture', () => {
  const t = new PeakTracker();
  const dt = PEAK_SAMPLE_INTERVAL_MS;
  t.push(0, 0.9, 'GROUP_3', { sharpness: 0, name: 'a' });
  t.push(dt, 1.0, 'GROUP_3', { sharpness: 0, name: 'b' });
  const winner = t.push(dt * 2, 0.9, 'GROUP_3', { sharpness: 0, name: 'c' });

  assert.ok(winner, 'no measurement must not mean no capture');
});
