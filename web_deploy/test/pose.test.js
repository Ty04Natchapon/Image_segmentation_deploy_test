import test from 'node:test';
import assert from 'node:assert/strict';
import {
  symmetryRatio, classifyRatio, checkDistance, checkLighting,
  checkTargetPose, targetInstruction, stepLabel, SEQUENCE, PeakTracker,
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

test('the sequence asks for front, then left cheek, then right', () => {
  assert.deepEqual(SEQUENCE, ['GROUP_3', 'GROUP_2', 'GROUP_1']);
  assert.match(targetInstruction('GROUP_3'), /straight at the camera/);
  assert.match(targetInstruction('GROUP_2'), /LEFT cheek/);
  assert.match(targetInstruction('GROUP_1'), /RIGHT cheek/);
});

test('a pose that is not the one asked for is refused, and says what to do', () => {
  // This is the check the Python could not make. There, the detected pose WAS
  // the target by construction, so "turn your head" was unreachable.
  const wrong = checkTargetPose('GROUP_2', 'GROUP_3', 1.0);
  assert.equal(wrong.ok, false);
  assert.match(wrong.msg, /LEFT cheek/);

  const alsoWrong = checkTargetPose('GROUP_1', 'GROUP_2', 1.8);
  assert.equal(alsoWrong.ok, false);
  assert.match(alsoWrong.msg, /RIGHT cheek/);
});

test('matching the target passes, and the front keeps its tighter band', () => {
  assert.equal(checkTargetPose('GROUP_2', 'GROUP_2', 1.8).ok, true);
  assert.equal(checkTargetPose('GROUP_1', 'GROUP_1', 0.4).ok, true);

  assert.equal(checkTargetPose('GROUP_3', 'GROUP_3', 1.0).ok, true);
  // classifyRatio calls 0.6..1.5 "front", which is far looser than a portrait
  // needs, so the front target re-checks at +/-0.15.
  assert.equal(checkTargetPose('GROUP_3', 'GROUP_3', 1.30).ok, false);
  assert.match(checkTargetPose('GROUP_3', 'GROUP_3', 1.30).msg, /squarely/);
});

test('step labels count from one and end with a finished state', () => {
  assert.match(stepLabel(0), /^Step 1 of 3 — Front$/);
  assert.match(stepLabel(1), /^Step 2 of 3 — Left cheek$/);
  assert.match(stepLabel(2), /^Step 3 of 3 — Right cheek$/);
  assert.match(stepLabel(3), /All three captured/);
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
