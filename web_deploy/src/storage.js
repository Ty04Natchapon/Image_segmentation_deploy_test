/**
 * storage.js — captures live in IndexedDB on the device.
 *
 * The Python wrote three folders of JPEGs. A browser has no folders, and more
 * to the point face images should not leave the handset unless the user says
 * so, so each capture is kept locally as three blobs and exported on demand.
 *
 *   clean.jpg    the frame with NOTHING drawn on it
 *   mask.png     the region mask, white = in-region skin
 *   preview.jpg  clean + overlay, for the gallery thumbnail only
 *
 * Python saved the *annotated* frame — the one with 3px contour lines painted
 * over the skin at the region boundary — which quietly corrupts any later
 * per-pixel analysis. Keeping the overlay in a separate file fixes that.
 */

import { groupDir } from './pose.js';

const DB_NAME = 'skin-capture';
const DB_VERSION = 1;
const STORE = 'captures';

let dbPromise = null;

function openDb() {
  if (dbPromise) return dbPromise;
  dbPromise = new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, DB_VERSION);
    req.onupgradeneeded = () => {
      const db = req.result;
      if (!db.objectStoreNames.contains(STORE)) {
        const store = db.createObjectStore(STORE, { keyPath: 'id', autoIncrement: true });
        store.createIndex('ts', 'ts');
        store.createIndex('group', 'group');
      }
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
  return dbPromise;
}

/**
 * IndexedDB reports failures badly: `transaction.error` is frequently null on
 * an abort, so passing it straight to the caller produces the useless message
 * "null". Always hand back a real Error, and prefer the request-level error,
 * which names the actual cause where the transaction-level one does not.
 */
function dbError(source, what) {
  const e = source && source.error;
  if (e instanceof Error) return e;
  if (e) return new Error(`IndexedDB ${what}: ${e.name || e}`);
  return new Error(`IndexedDB ${what} with no error detail`);
}

function tx(mode, fn) {
  return openDb().then((db) => new Promise((resolve, reject) => {
    const t = db.transaction(STORE, mode);
    const store = t.objectStore(STORE);
    let result;
    try {
      result = fn(store);
    } catch (err) {
      reject(err);
      return;
    }
    if (result && result.__req) {
      result.__req.onerror = () => reject(dbError(result.__req, 'request failed'));
    }
    t.oncomplete = () => resolve(result && result.__req ? result.__req.result : result);
    t.onerror = () => reject(dbError(t, 'transaction failed'));
    t.onabort = () => reject(dbError(t, 'transaction aborted'));
  }));
}

const wrap = (req) => ({ __req: req });

/** Mirrors the Python filename: Group_3_Front_1738500000_px128437 */
export function captureBasename(rec) {
  return `${groupDir(rec.group)}_${Math.floor(rec.ts / 1000)}_px${rec.skinPx}`;
}

// Images are stored as ArrayBuffers, not Blobs.
//
// WebKit's IndexedDB has long-standing trouble writing Blobs: the transaction
// aborts with a null error, which is exactly the "capture failed - null" this
// worked around. ArrayBuffers are structured-cloned reliably everywhere, so we
// unwrap on the way in and rebuild the Blob on the way out. Callers never see
// the difference.
const BLOB_FIELDS = ['clean', 'mask', 'preview'];

// Exported for test/storage.test.js only: this round trip is what a capture
// lives or dies by, and it was untested when it broke.
export async function toStorable(rec) {
  const out = { ...rec };
  for (const field of BLOB_FIELDS) {
    const value = rec[field];
    if (value instanceof Blob) {
      out[field] = await value.arrayBuffer();
      out[`${field}Type`] = value.type || '';
    }
  }
  return out;
}

export function fromStored(rec) {
  const out = { ...rec };
  for (const field of BLOB_FIELDS) {
    const value = rec[field];
    if (value && !(value instanceof Blob)) {
      out[field] = new Blob([value], { type: rec[`${field}Type`] || '' });
    }
  }
  return out;
}

export async function saveCapture(rec) {
  // Converted before the transaction opens: an IndexedDB transaction closes
  // itself the moment the task queue drains, so awaiting inside one kills it.
  const storable = await toStorable(rec);
  return tx('readwrite', (store) => wrap(store.add(storable)));
}

export function listCaptures() {
  return tx('readonly', (store) => wrap(store.getAll()))
    .then((rows) => rows.map(fromStored).sort((a, b) => b.ts - a.ts));
}

export function deleteCapture(id) {
  return tx('readwrite', (store) => wrap(store.delete(id)));
}

export function clearAll() {
  return tx('readwrite', (store) => wrap(store.clear()));
}

/**
 * Hand the capture to the OS.
 *
 * On mobile the share sheet is the reliable path — iOS Safari treats an
 * <a download> pointing at a blob URL as "open in a new tab", which is not
 * what anyone wants. Desktop falls back to real downloads.
 */
export async function exportCapture(rec) {
  const base = captureBasename(rec);
  const files = [
    new File([rec.clean], `${base}.jpg`, { type: 'image/jpeg' }),
    new File([rec.mask], `${base}_mask.png`, { type: 'image/png' }),
  ];

  if (navigator.canShare && navigator.canShare({ files })) {
    try {
      await navigator.share({ files, title: base });
      return 'shared';
    } catch (err) {
      if (err && err.name === 'AbortError') return 'cancelled';
      // fall through to download
    }
  }

  for (const file of files) {
    const url = URL.createObjectURL(file);
    const a = document.createElement('a');
    a.href = url;
    a.download = file.name;
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 10_000);
  }
  return 'downloaded';
}

// --- upload bookkeeping ----------------------------------------------------
// The network half lives in sync.js; this is just the durable record of what
// has and has not made it to the server. Keeping the two apart is what lets a
// failed upload be a retry rather than a lost capture: the image is already
// safely on disk before anyone tries to send it.

/** Captures not yet accepted by the server, oldest first (upload in order). */
export function listPending() {
  return tx('readonly', (store) => wrap(store.getAll()))
    .then((rows) => rows.filter((r) => !r.uploaded).map(fromStored)
      .sort((a, b) => a.ts - b.ts));
}

function patch(id, fields) {
  return tx('readwrite', (store) => {
    const req = store.get(id);
    req.onsuccess = () => {
      const rec = req.result;
      if (rec) store.put({ ...rec, ...fields });
    };
    return null;
  });
}

export function markUploaded(id) {
  return patch(id, { uploaded: true, uploadError: null, uploadedAt: Date.now() });
}

export function markUploadFailed(id, attempts, message) {
  return patch(id, { uploaded: false, uploadAttempts: attempts, uploadError: message });
}
