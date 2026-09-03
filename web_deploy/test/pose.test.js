import test from 'node:test';
import assert from 'node:assert/strict';
import {
  symmetryRatio, classifyRatio, checkDistance, checkLighting, checkPose,
  nextTargetHint, PeakTracker,
} from '../src/pose.js';
import { makeFace } from './fixtures.js';

const W = 256;

function faceWithNoseAt(x) {
  const lm = makeFace();
  lm[1] = { x, y: 0.52, z: 0 };
  return lm;
}

const solid = (v, n = 4) => {
  const px = new Uint8ClampedArray(n * n * 4);
  for (let i = 0; i < px.length; i += 4) {
    px[i] = px[i + 1] = px[i + 2] = v;
    px[i + 3] = 255;
  }
  return px;
};
const fullBox = (n = 4) => ({ x0: 0, y0: 0, x1: n, y1: n });

test('symmetry ratio is 1.0 head-on and swings with the turn', () => {
  assert.ok(Math.abs(symmetryRatio(faceWithNoseAt(0.5), W) - 1) < 1e-9);
  assert.ok(symmetryRatio(faceWithNoseAt(0.36), W) < 1, 'nose near the left temple');
  assert.ok(symmetryRatio(faceWithNoseAt(0.64), W) > 1, 'nose near the right temple');
});

test('classifyRatio splits on the same thresholds as the Python', () => {
  assert.equal(classifyRatio(1.0), 'GROUP_3');
  assert.equal(classifyRatio(1.51), 'GROUP_2');
  assert.equal(classifyRatio(1.50), 'GROUP_3', 'boundary is exclusive');
  assert.equal(classifyRatio(0.59), 'GROUP_1');
  assert.equal(classifyRatio(0.60), 'GROUP_3', 'boundary is exclusive');
});

test('distance gate measures against the shorter frame edge', () => {
  // Same face, landscape and portrait: the verdict must not change.
  const b = { width: 0.40 * 480 };
  assert.equal(checkDistance(b, 640, 480).ok, true);
  assert.equal(checkDistance(b, 480, 640).ok, true);

  assert.equal(checkDistance({ width: 0.10 * 480 }, 640, 480).msg, 'Move closer');
  assert.equal(checkDistance({ width: 0.80 * 480 }, 640, 480).msg, 'Move back');
});

test('lighting gate flags dark and blown-out frames', () => {
  assert.equal(checkLighting(solid(128), 4, fullBox()).ok, true);
  assert.equal(checkLighting(solid(20), 4, fullBox()).ok, false);
  assert.equal(checkLighting(solid(250), 4, fullBox()).ok, false);
  assert.ok(Math.abs(checkLighting(solid(128), 4, fullBox()).brightness - 128) < 1);
});

test('pose gate corrects only the front shot, and names the direction', () => {
  assert.equal(checkPose('GROUP_3', 1.0).ok, true);
  assert.equal(checkPose('GROUP_3', 1.30).ok, false);
  assert.match(checkPose('GROUP_3', 1.30).msg, /turn slightly right/);
  assert.match(checkPose('GROUP_3', 0.70).msg, /turn slightly left/);

  // Inherited from the Python: classifyRatio already used these thresholds to
  // pick the group, so a cheek shot passes by construction.
  assert.equal(checkPose('GROUP_1', 0.3).ok, true);
  assert.equal(checkPose('GROUP_2', 1.9).ok, true);
});

test('nextTargetHint walks the user through the missing angles', () => {
  const none = { GROUP_1: 0, GROUP_2: 0, GROUP_3: 0 };
  assert.match(nextTargetHint(none), /straight at the camera/);
  assert.match(nextTargetHint({ ...none, GROUP_3: 1 }), /LEFT cheek/);
  assert.match(nextTargetHint({ GROUP_1: 0, GROUP_2: 1, GROUP_3: 1 }), /RIGHT cheek/);
  assert.match(nextTargetHint({ GROUP_1: 1, GROUP_2: 1, GROUP_3: 1 }), /All three/);
});

test('the peak detector samples on a clock, not on frames', () => {
  const t = new PeakTracker();
  assert.equal(t.shouldSample(0), true);
  t.push(0, 1.0, 'GROUP_3', 'a');
  assert.equal(t.shouldSample(10), false, '10ms later is the same pose');
  assert.equal(t.push(10, 1.0, 'GROUP_3', 'ignored'), null);
  assert.equal(t.shouldSample(60), true);
});

test('the peak is the middle of three, per group', () => {
  const front = new PeakTracker();
  assert.equal(front.push(0, 0.9, 'GROUP_3', 'a'), null);
  assert.equal(front.push(60, 1.0, 'GROUP_3', 'b'), null);
  assert.equal(front.push(120, 0.9, 'GROUP_3', 'c'), 'b', 'closest to 1.0 wins');

  const left = new PeakTracker();
  left.push(0, 1.6, 'GROUP_2', 'a');
  left.push(60, 1.9, 'GROUP_2', 'b');
  assert.equal(left.push(120, 1.7, 'GROUP_2', 'c'), 'b', 'local max wins');

  const right = new PeakTracker();
  right.push(0, 0.5, 'GROUP_1', 'a');
  right.push(60, 0.3, 'GROUP_1', 'b');
  assert.equal(right.push(120, 0.4, 'GROUP_1', 'c'), 'b', 'local min wins');
});

test('a still-turning head and a group change both fail to peak', () => {
  const turning = new PeakTracker();
  turning.push(0, 1.0, 'GROUP_3', 'a');
  turning.push(60, 1.1, 'GROUP_3', 'b');
  assert.equal(turning.push(120, 1.2, 'GROUP_3', 'c'), null, 'still moving away');

  const mixed = new PeakTracker();
  mixed.push(0, 1.4, 'GROUP_3', 'a');
  mixed.push(60, 1.6, 'GROUP_2', 'b');
  assert.equal(mixed.push(120, 1.5, 'GROUP_2', 'c'), null, 'group changed mid-window');
});
