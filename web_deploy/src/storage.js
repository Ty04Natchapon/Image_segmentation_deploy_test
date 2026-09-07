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
    t.oncomplete = () => resolve(result && result.__req ? result.__req.result : result);
    t.onerror = () => reject(t.error);
    t.onabort = () => reject(t.error);
  }));
}

const wrap = (req) => ({ __req: req });

/** Mirrors the Python filename: Group_3_Front_1738500000_px128437 */
export function captureBasename(rec) {
  return `${groupDir(rec.group)}_${Math.floor(rec.ts / 1000)}_px${rec.skinPx}`;
}

export function saveCapture(rec) {
  return tx('readwrite', (store) => wrap(store.add(rec)));
}

export function listCaptures() {
  return tx('readonly', (store) => wrap(store.getAll()))
    .then((rows) => rows.sort((a, b) => b.ts - a.ts));
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
    .then((rows) => rows.filter((r) => !r.uploaded).sort((a, b) => a.ts - b.ts));
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
