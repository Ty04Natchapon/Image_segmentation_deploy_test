/**
 * sync.js — handing captures to the analysis server.
 *
 * This is the boundary of this app's responsibility. Everything upstream of
 * here (guiding the shot, segmenting the region, deciding the frame is good)
 * happens on the device; everything downstream of the POST belongs to whoever
 * owns the analysis algorithm.
 *
 * buildFormData() below IS the interface contract. If the receiving team wants
 * a different shape, this one function is the only thing that changes — which
 * is the point of keeping it isolated. The contract is written up in README
 * under "Sending captures to the analysis server"; keep the two in step.
 *
 * The design assumption throughout: a phone on mobile data will fail
 * mid-upload, and that must never cost a capture. Images are durable in
 * IndexedDB before any upload is attempted, so a failure is only ever a retry.
 */

import {
  UPLOAD_ENDPOINT, UPLOAD_MAX_ATTEMPTS, UPLOAD_TIMEOUT_MS, UPLOAD_HEADERS, APP_VERSION,
} from './config.js';
import * as store from './storage.js';

const REGION_NAMES = {
  GROUP_1: 'right_cheek',
  GROUP_2: 'left_cheek',
  GROUP_3: 'front',
};

/**
 * The wire format. One multipart POST per capture.
 *
 * Two things the receiving team needs to know, and which are easy to get
 * wrong: `image` is the CLEAN frame with nothing drawn on it, and `mask` is a
 * separate 8-bit PNG at exactly the same dimensions, white where the region
 * is. They are aligned pixel for pixel, so the analysis can be confined to the
 * region with a straight boolean AND — no resampling, no coordinate transform.
 */
export function buildFormData(rec) {
  const base = store.captureBasename(rec);
  const body = new FormData();

  // Identity. capture_id is stable across retries, so the server can make
  // ingestion idempotent instead of collecting duplicates from flaky networks.
  body.append('capture_id', rec.captureId);
  body.append('session_id', rec.sessionId);

  // Which of the three angles this is.
  body.append('group', rec.group);
  body.append('region', REGION_NAMES[rec.group] || 'unknown');

  // Measurements taken at capture time.
  body.append('captured_at', new Date(rec.ts).toISOString());
  body.append('skin_px', String(rec.skinPx));
  // Face width in pixels. Divide by 140 for px/mm — the figure that decides
  // whether a lesion is resolvable, and the cheapest quality filter you have.
  body.append('face_px', String(rec.facePx || 0));
  // Variance of the Laplacian over the face box. Device-relative, so use it
  // to rank captures of the same person on the same phone, not as an absolute.
  body.append('sharpness', String(rec.sharpness || 0));
  // 'auto' (peak detector) or 'manual' (user shutter). Kept so the two can be
  // compared on collected data instead of on impressions.
  body.append('mode', rec.mode || 'auto');
  body.append('width', String(rec.width));
  body.append('height', String(rec.height));
  body.append('ratio', rec.ratio.toFixed(4));
  body.append('brightness', rec.brightness.toFixed(1));

  // Acquisition conditions — there is no raw path on a phone browser, so the
  // next best thing is recording what the camera was doing.
  body.append('fill_light', String(Boolean(rec.fillLight)));
  body.append('camera', JSON.stringify(rec.camera || {}));
  body.append('app_version', APP_VERSION);

  body.append('image', rec.clean, `${base}.jpg`);
  body.append('mask', rec.mask, `${base}_mask.png`);
  return body;
}

/** POST one capture. Resolves on 2xx, throws otherwise. */
export async function uploadOne(rec, endpoint = UPLOAD_ENDPOINT) {
  if (!endpoint) throw new Error('No upload endpoint configured');

  // Without a timeout a stalled mobile connection hangs the queue forever.
  const abort = new AbortController();
  const timer = setTimeout(() => abort.abort(), UPLOAD_TIMEOUT_MS);
  try {
    const res = await fetch(endpoint, {
      method: 'POST',
      body: buildFormData(rec),
      // Never set Content-Type here: the browser has to add the multipart
      // boundary itself, and overriding it produces a body the server cannot
      // parse.
      headers: UPLOAD_HEADERS,
      signal: abort.signal,
    });
    if (!res.ok) throw new Error(`Server returned ${res.status} ${res.statusText}`);
    return res;
  } finally {
    clearTimeout(timer);
  }
}

let running = false;

/**
 * Drain the pending queue, oldest first.
 *
 * Serial on purpose: three concurrent multi-megabyte uploads over one phone
 * connection is slower than three in a row, and it makes failures much harder
 * to attribute. Safe to call often — overlapping calls collapse into one.
 */
export async function syncPending({ onProgress } = {}) {
  if (running || !UPLOAD_ENDPOINT) return { sent: 0, failed: 0, skipped: !UPLOAD_ENDPOINT };
  running = true;

  let sent = 0;
  let failed = 0;
  try {
    const pending = await store.listPending();
    for (const rec of pending) {
      const attempts = (rec.uploadAttempts || 0) + 1;
      if (attempts > UPLOAD_MAX_ATTEMPTS) {
        // Stop retrying, but keep the capture — it can still be exported by
        // hand, and giving up on the network is not the same as losing data.
        failed++;
        continue;
      }
      if (onProgress) onProgress({ rec, attempts, remaining: pending.length - sent - failed });

      try {
        await uploadOne(rec);
        await store.markUploaded(rec.id);
        sent++;
      } catch (err) {
        const message = err && err.name === 'AbortError'
          ? `Timed out after ${UPLOAD_TIMEOUT_MS / 1000}s`
          : String((err && err.message) || err);
        await store.markUploadFailed(rec.id, attempts, message);
        failed++;
        console.warn(`[sync] ${rec.id} attempt ${attempts} failed: ${message}`);
      }
    }
  } finally {
    running = false;
  }
  return { sent, failed };
}

export async function pendingCount() {
  if (!UPLOAD_ENDPOINT) return 0;
  return (await store.listPending()).length;
}

/** Retry automatically when the phone comes back online. */
export function watchConnectivity(onSynced) {
  if (!UPLOAD_ENDPOINT) return;
  window.addEventListener('online', async () => {
    const result = await syncPending();
    if (result.sent && onSynced) onSynced(result);
  });
}
