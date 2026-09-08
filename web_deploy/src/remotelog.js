/**
 * remotelog.js — forwards the phone's errors to the dev server's terminal.
 *
 * An iPhone has no readable console without a tethered Mac, which makes any
 * bug that only reproduces on the handset painful to chase: you can see that
 * something failed, but not what. When an upload endpoint is configured this
 * POSTs errors to it, and tools/mock_server.py prints them.
 *
 * Diagnostics only. It does nothing unless an endpoint is set, and it never
 * sends image data — only messages and stack traces.
 */

import { UPLOAD_ENDPOINT } from './config.js';

let target = null;
let sending = false;
const queue = [];

/** Resolve /__log against the configured endpoint, whatever shape it is. */
function resolveTarget() {
  if (!UPLOAD_ENDPOINT) return null;
  try {
    return new URL('__log', new URL(UPLOAD_ENDPOINT, location.href)).href;
  } catch {
    return null;
  }
}

async function drain() {
  if (sending || !queue.length || !target) return;
  sending = true;
  try {
    while (queue.length) {
      const entry = queue.shift();
      // keepalive so a log sent during a page hide still gets out.
      await fetch(target, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(entry),
        keepalive: true,
      }).catch(() => {});
    }
  } finally {
    sending = false;
  }
}

export function remoteLog(message, detail = '', level = 'error') {
  if (!target) return;
  // Bounded: a bug that fires every frame must not become a network flood.
  if (queue.length > 20) return;
  queue.push({ level, message: String(message), detail: String(detail) });
  drain();
}

/**
 * Catch what nothing else does — a throw outside a try, or a promise that
 * rejects with no handler. These are precisely the failures that leave no
 * trace on a phone.
 */
export function installRemoteLogging() {
  target = resolveTarget();
  if (!target) return false;

  window.addEventListener('error', (e) => {
    remoteLog(e.message || 'window error', `${e.filename || ''}:${e.lineno || 0}`);
  });
  window.addEventListener('unhandledrejection', (e) => {
    const r = e.reason;
    remoteLog(`unhandled rejection: ${(r && r.message) || r}`, (r && r.stack) || '');
  });

  remoteLog(
    `connected — ${navigator.userAgent}`,
    `screen ${screen.width}x${screen.height} dpr ${devicePixelRatio}`,
    'info',
  );
  return true;
}
