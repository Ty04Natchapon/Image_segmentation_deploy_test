/**
 * pose.js — pose classification, the three quality gates, and the peak
 * detector. Ported from capture_prototype.py.
 */

import * as L from './landmarks.js';
import {
  RATIO_LEFT_CHEEK_THRESHOLD,
  RATIO_RIGHT_CHEEK_THRESHOLD,
  FRONT_SYMMETRY_TOLERANCE,
  FACE_WIDTH_MIN_FRAC,
  FACE_WIDTH_MAX_FRAC,
  BRIGHTNESS_MIN,
  BRIGHTNESS_MAX,
  MIN_SKIN_COVERAGE,
  MIN_FOREHEAD_COVERAGE,
  PEAK_SAMPLE_INTERVAL_MS,
  PEAK_WINDOW,
  GROUPS,
  GROUP_ORDER,
} from './config.js';

/** dist(nose, left temple) / dist(nose, right temple). 1.0 means facing front. */
export function symmetryRatio(landmarks, w) {
  const noseX = landmarks[L.NOSE_TIP].x * w;
  const leftX = landmarks[L.LEFT_TEMPLE].x * w;
  const rightX = landmarks[L.RIGHT_TEMPLE].x * w;
  const distLeft = Math.abs(noseX - leftX);
  const distRight = Math.abs(noseX - rightX) || 1e-6;
  return distLeft / distRight;
}

export function classifyRatio(ratio) {
  if (ratio > RATIO_LEFT_CHEEK_THRESHOLD) return 'GROUP_2';
  if (ratio < RATIO_RIGHT_CHEEK_THRESHOLD) return 'GROUP_1';
  return 'GROUP_3';
}

/**
 * Distance gate.
 *
 * Python compared the *padded* bbox against frame WIDTH, which falls apart in
 * portrait: the same face reads as a far smaller fraction of a tall frame. We
 * use the unpadded landmark span over the SHORTER frame edge, which behaves
 * the same in either orientation. See config.js for the converted thresholds.
 */
export function checkDistance(bounds, w, h) {
  const frac = bounds.width / Math.min(w, h);
  if (frac < FACE_WIDTH_MIN_FRAC) return { ok: false, msg: 'Move closer', frac };
  if (frac > FACE_WIDTH_MAX_FRAC) return { ok: false, msg: 'Move back', frac };
  return { ok: true, msg: 'Distance OK', frac };
}

/**
 * Lighting gate — mean luma inside the padded face box.
 *
 * Worth knowing on mobile: a phone ISP auto-exposes hard, so this gate passes
 * frames shot in genuinely bad light far more often than a webcam would. It
 * catches gross failures, not subtle ones.
 */
export function checkLighting(rgba, w, box) {
  let sum = 0;
  let n = 0;
  for (let y = box.y0; y < box.y1; y += 2) {
    let o = (y * w + box.x0) * 4;
    for (let x = box.x0; x < box.x1; x += 2, o += 8) {
      sum += 0.299 * rgba[o] + 0.587 * rgba[o + 1] + 0.114 * rgba[o + 2];
      n++;
    }
  }
  if (!n) return { ok: true, msg: 'Light ?', brightness: 0 };
  const brightness = sum / n;
  if (brightness < BRIGHTNESS_MIN) {
    return { ok: false, msg: 'Too dark — add light', brightness };
  }
  if (brightness > BRIGHTNESS_MAX) {
    return { ok: false, msg: 'Too bright — reduce light', brightness };
  }
  return { ok: true, msg: 'Light OK', brightness };
}

/**
 * Pose gate.
 *
 * Note the asymmetry, which is inherited from the Python and is not a bug:
 * classifyRatio() already used the 1.5 / 0.6 thresholds to *pick* the group,
 * so a cheek shot is passing by construction and can only ever read "hold
 * still". Only the front has a genuinely tighter gate — the 0.6..1.5
 * classification band versus the +/-0.15 capture band — which is why
 * "face forward" is the one correction the user can actually receive.
 */
export function checkPose(group, ratio) {
  if (group === 'GROUP_2') return { ok: true, msg: 'Hold still — left cheek' };
  if (group === 'GROUP_1') return { ok: true, msg: 'Hold still — right cheek' };
  if (Math.abs(ratio - 1.0) <= FRONT_SYMMETRY_TOLERANCE) {
    return { ok: true, msg: 'Hold still — front' };
  }
  return {
    ok: false,
    msg: ratio > 1.0 ? 'Face forward (turn slightly right)'
                     : 'Face forward (turn slightly left)',
  };
}

/**
 * Is anything covering the skin we are about to measure?
 *
 * `coverage` is how much of the region the segmenter calls face-skin; glasses
 * frames, a hand or hair across a cheek all drag it down. `forehead` is the
 * same idea for the band above the brow, which the front shot needs
 * separately — its region is mostly the centre strip, so a fringe barely moves
 * the overall figure while ruining the forehead entirely.
 */
export function checkOcclusion(group, coverage, forehead) {
  if (coverage < MIN_SKIN_COVERAGE) {
    return { ok: false, msg: 'Move hair or glasses off your face', coverage };
  }
  if (group === 'GROUP_3' && forehead < MIN_FOREHEAD_COVERAGE) {
    return { ok: false, msg: 'Move hair off your forehead', coverage };
  }
  return { ok: true, msg: 'Skin clear', coverage };
}

/** What to ask the user for next, given what has already been captured. */
export function nextTargetHint(counts) {
  const missing = GROUP_ORDER.find((g) => !counts[g]);
  if (!missing) return 'All three angles captured';
  if (missing === 'GROUP_3') return 'Look straight at the camera';
  if (missing === 'GROUP_2') return 'Turn your head to show your LEFT cheek';
  return 'Turn your head to show your RIGHT cheek';
}

function isPeak(prev2, prev1, curr, group) {
  if (group === 'GROUP_2') return prev1 > prev2 && prev1 >= curr;
  if (group === 'GROUP_1') return prev1 < prev2 && prev1 <= curr;
  return Math.abs(prev1 - 1.0) < Math.abs(prev2 - 1.0) &&
         Math.abs(prev1 - 1.0) <= Math.abs(curr - 1.0);
}

/**
 * Fires when the middle of three consecutive samples is a local extreme of the
 * symmetry ratio — the instant the head stopped turning.
 *
 * Python kept a 3-FRAME window, so its real duration was hostage to the frame
 * rate: 100ms at 30fps, 200ms at 15fps, and the peak got mushier the slower
 * the device. Sampling on a fixed clock instead keeps the window at ~180ms on
 * any handset.
 */
export class PeakTracker {
  constructor() {
    this.samples = [];
    this.lastSampleAt = -Infinity;
  }

  reset() {
    this.samples.length = 0;
  }

  /**
   * Whether push() would actually take a sample right now. The caller checks
   * this before snapshotting a full-resolution frame, so we only pay for the
   * copy on frames that can win.
   */
  shouldSample(now) {
    return now - this.lastSampleAt >= PEAK_SAMPLE_INTERVAL_MS;
  }

  /** Returns the winning sample's payload, or null. */
  push(now, ratio, group, payload) {
    if (now - this.lastSampleAt < PEAK_SAMPLE_INTERVAL_MS) return null;
    this.lastSampleAt = now;

    this.samples.push({ ratio, group, payload });
    if (this.samples.length > PEAK_WINDOW) this.samples.shift();
    if (this.samples.length < PEAK_WINDOW) return null;

    const [a, b, c] = this.samples;
    if (a.group !== b.group || b.group !== c.group) return null;
    if (!isPeak(a.ratio, b.ratio, c.ratio, b.group)) return null;

    this.reset();
    return b.payload;
  }
}

export const groupLabel = (g) => GROUPS[g].label;
export const groupDir = (g) => GROUPS[g].dir;
