/**
 * The Blob <-> ArrayBuffer round trip that captures are stored through.
 *
 * WebKit's IndexedDB aborts with a null error when asked to store a Blob,
 * which is what made captures vanish with the message "capture failed - null".
 * Images go in as ArrayBuffers and come back out as Blobs; if that conversion
 * is wrong, every saved capture is silently corrupt.
 */

import test from 'node:test';
import assert from 'node:assert/strict';
import { toStorable, fromStored } from '../src/storage.js';

const bytes = (n, seed = 0) =>
  new Uint8Array(Array.from({ length: n }, (_, i) => (i * 7 + seed) % 256));

function record() {
  return {
    group: 'GROUP_3',
    ts: 1757000000000,
    skinPx: 128437,
    clean: new Blob([bytes(64, 1)], { type: 'image/jpeg' }),
    mask: new Blob([bytes(32, 2)], { type: 'image/png' }),
    preview: new Blob([bytes(16, 3)], { type: 'image/jpeg' }),
  };
}

test('images are unwrapped to ArrayBuffers before storage', async () => {
  const stored = await toStorable(record());
  for (const field of ['clean', 'mask', 'preview']) {
    assert.ok(stored[field] instanceof ArrayBuffer, `${field} should be an ArrayBuffer`);
    assert.ok(!(stored[field] instanceof Blob), `${field} must not stay a Blob`);
  }
  assert.equal(stored.cleanType, 'image/jpeg');
  assert.equal(stored.maskType, 'image/png');
});

test('the round trip preserves bytes and MIME type exactly', async () => {
  const original = record();
  const restored = fromStored(await toStorable(original));

  for (const field of ['clean', 'mask', 'preview']) {
    assert.ok(restored[field] instanceof Blob, `${field} should come back a Blob`);
    const before = new Uint8Array(await original[field].arrayBuffer());
    const after = new Uint8Array(await restored[field].arrayBuffer());
    assert.deepEqual(Array.from(after), Array.from(before), `${field} bytes changed`);
    assert.equal(restored[field].type, original[field].type, `${field} type changed`);
  }
});

test('non-image fields pass through untouched', async () => {
  const restored = fromStored(await toStorable(record()));
  assert.equal(restored.group, 'GROUP_3');
  assert.equal(restored.ts, 1757000000000);
  assert.equal(restored.skinPx, 128437);
});

test('a record already holding Blobs is left alone by fromStored', async () => {
  // Rows written by an older build still have real Blobs in them.
  const legacy = record();
  const restored = fromStored(legacy);
  assert.equal(restored.clean, legacy.clean, 'must not rewrap an existing Blob');
});

test('a record with no images does not invent any', async () => {
  const stored = await toStorable({ group: 'GROUP_1', ts: 1 });
  const restored = fromStored(stored);
  assert.equal(restored.clean, undefined);
  assert.equal(restored.mask, undefined);
});
