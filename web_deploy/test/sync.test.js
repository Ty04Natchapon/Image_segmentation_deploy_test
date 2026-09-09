/**
 * The upload contract, locked down.
 *
 * These assertions exist for a hand-off reason rather than a correctness one:
 * another team builds the endpoint that receives this payload, so a silent
 * change to the field names would break them with no signal on our side. If a
 * test here fails, that is the reminder to tell them before shipping it —
 * and to update tools/mock_server.py to match.
 */

import test from 'node:test';
import assert from 'node:assert/strict';
import { buildFormData } from '../src/sync.js';

function makeRecord(overrides = {}) {
  return {
    id: 1,
    captureId: '7f0b2c4e-0000-4000-8000-000000000001',
    sessionId: 'session-abc',
    group: 'GROUP_3',
    ts: Date.UTC(2026, 8, 7, 10, 30, 0),
    ratio: 1.00213,
    brightness: 142.34,
    skinPx: 128437,
    facePx: 430,
    sharpness: 87,
    width: 1280,
    height: 720,
    camera: { width: 1280, height: 720, frameRate: 30 },
    fillLight: false,
    clean: new Blob([new Uint8Array([1, 2, 3])], { type: 'image/jpeg' }),
    mask: new Blob([new Uint8Array([4, 5])], { type: 'image/png' }),
    ...overrides,
  };
}

test('the payload carries every field the analysis server is promised', () => {
  const fd = buildFormData(makeRecord());
  const expected = [
    'capture_id', 'session_id', 'group', 'region', 'captured_at',
    'skin_px', 'face_px', 'sharpness', 'width', 'height', 'ratio', 'brightness',
    'fill_light', 'camera', 'app_version', 'image', 'mask',
  ];
  for (const field of expected) {
    assert.ok(fd.has(field), `contract field "${field}" is missing`);
  }
  assert.equal([...fd.keys()].length, expected.length, 'unexpected extra fields');
});

test('identity fields are verbatim, so retries are idempotent', () => {
  const rec = makeRecord();
  const a = buildFormData(rec);
  const b = buildFormData(rec);
  assert.equal(a.get('capture_id'), rec.captureId);
  assert.equal(b.get('capture_id'), rec.captureId, 'must not change between attempts');
  assert.equal(a.get('session_id'), 'session-abc');
});

test('group maps to a readable region name', () => {
  const name = (g) => buildFormData(makeRecord({ group: g })).get('region');
  assert.equal(name('GROUP_1'), 'right_cheek');
  assert.equal(name('GROUP_2'), 'left_cheek');
  assert.equal(name('GROUP_3'), 'front');
});

test('timestamps go out as ISO 8601 UTC, not a raw epoch', () => {
  const at = buildFormData(makeRecord()).get('captured_at');
  assert.equal(at, '2026-09-07T10:30:00.000Z');
});

test('camera settings are serialised as JSON the server can parse', () => {
  const raw = buildFormData(makeRecord()).get('camera');
  assert.deepEqual(JSON.parse(raw), { width: 1280, height: 720, frameRate: 30 });

  // A device that reports nothing must still produce valid JSON, not "undefined".
  const empty = buildFormData(makeRecord({ camera: undefined })).get('camera');
  assert.deepEqual(JSON.parse(empty), {});
});

test('image and mask are files, and the mask filename is distinguishable', () => {
  const fd = buildFormData(makeRecord());
  const image = fd.get('image');
  const mask = fd.get('mask');

  assert.ok(image instanceof Blob, 'image must be sent as a file part');
  assert.ok(mask instanceof Blob, 'mask must be sent as a file part');
  assert.match(fd.get('image').name ?? '', /Group_3_Front_.*\.jpg$/);
  assert.match(fd.get('mask').name ?? '', /Group_3_Front_.*_mask\.png$/);
});

test('the filename encodes group, time and pixel count, as the Python did', () => {
  const name = buildFormData(makeRecord()).get('image').name;
  assert.match(name, /^Group_3_Front_\d+_px128437\.jpg$/);
});

test('numeric fields are strings with fixed precision, not floats', () => {
  const fd = buildFormData(makeRecord());
  assert.equal(fd.get('ratio'), '1.0021');
  assert.equal(fd.get('brightness'), '142.3');
  assert.equal(fd.get('skin_px'), '128437');
  // Face width in pixels: divided by 140 it gives px/mm, the cheapest filter
  // the analysis side has for rejecting captures too coarse to learn from.
  assert.equal(fd.get('face_px'), '430');
  assert.equal(fd.get('sharpness'), '87');
  assert.equal(fd.get('fill_light'), 'false');
});
