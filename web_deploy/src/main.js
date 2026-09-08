/**
 * main.js — the app loop, the browser counterpart of main() in
 * capture_prototype.py.
 *
 * Per processed frame:
 *   1. mirror the camera frame into a downscaled working canvas
 *   2. FaceLandmarker -> 478 landmarks
 *   3. symmetry ratio -> GROUP_1 / GROUP_2 / GROUP_3
 *   4. segmenter (every Nth frame) -> face-skin mask
 *   5. build the region mask, count its pixels
 *   6. distance / lighting / pose gates
 *   7. feed the peak detector; on a peak, capture at FULL resolution
 */

import * as CFG from './config.js';
import * as R from './regions.js';
import * as M from './mask.js';
import {
  symmetryRatio, classifyRatio, checkDistance, checkLighting, checkPose,
  nextTargetHint, PeakTracker, groupLabel,
} from './pose.js';
import { loadVision, detectLandmarks, writeSkinMask } from './vision.js';
import * as store from './storage.js';
import * as sync from './sync.js';

const $ = (id) => document.getElementById(id);
const els = {
  view: $('view'), hint: $('hint'), gates: $('gates'), flash: $('flash'), debug: $('debug'),
  gateDistance: $('gateDistance'), gateLight: $('gateLight'), gatePose: $('gatePose'),
  startOverlay: $('startOverlay'), startBtn: $('startBtn'), startMsg: $('startMsg'),
  flipBtn: $('flipBtn'), galleryBtn: $('galleryBtn'), galleryCount: $('galleryCount'),
  lightBtn: $('lightBtn'), fillLight: $('fillLight'), syncBtn: $('syncBtn'),
  hold: $('hold'), holdBar: $('holdBar'), serverLink: $('serverLink'),
  errbar: $('errbar'),
  gallery: $('gallery'), galleryGrid: $('galleryGrid'), galleryEmpty: $('galleryEmpty'),
  closeGalleryBtn: $('closeGalleryBtn'), clearBtn: $('clearBtn'), progress: $('progress'),
};

const DEBUG = new URLSearchParams(location.search).has('debug');

const state = {
  vision: null,
  stream: null,
  video: null,
  facingMode: 'user',
  raf: 0,
  frameIndex: 0,
  lastTimestamp: -1,
  proc: null,          // { canvas, ctx, w, h }
  overlay: null,       // { canvas, ctx }
  lumaCanvas: null,
  skinBuffer: null,
  skinValid: false,
  ring: [],            // full-resolution snapshots for the peak window
  ringIndex: 0,
  tracker: new PeakTracker(),
  cooldownUntil: 0,
  flashUntil: 0,
  counts: { GROUP_1: 0, GROUP_2: 0, GROUP_3: 0 },
  lastPx: { GROUP_1: null, GROUP_2: null, GROUP_3: null },
  capturing: false,
  cameraCaps: {},
  cameraSettings: {},
  cameraControls: '',
  lastError: '',
  tries: 0,
  saved: 0,
  failed: 0,
  fps: 0,
  lastFrameAt: 0,
};

// --- small canvas helpers --------------------------------------------------

function makeCanvas(w, h) {
  const c = document.createElement('canvas');
  c.width = w;
  c.height = h;
  return c;
}

/**
 * The Python did `cv2.flip(frame, 1)` before anything else, and every
 * downstream decision — including which cheek counts as "left" — is defined in
 * that mirrored space. So we mirror the pixels here too. CSS-mirroring the
 * preview instead would look identical but feed the model an unflipped frame,
 * silently swapping GROUP_1 and GROUP_2.
 */
function drawMirrored(ctx, source, w, h) {
  ctx.save();
  ctx.setTransform(-1, 0, 0, 1, w, 0);
  ctx.drawImage(source, 0, 0, w, h);
  ctx.restore();
}

function procSize(vw, vh) {
  const scale = Math.min(1, CFG.PROC_LONG_EDGE / Math.max(vw, vh));
  return { w: Math.max(2, Math.round(vw * scale)), h: Math.max(2, Math.round(vh * scale)) };
}

/**
 * Mean luma inside the face box, measured by letting the GPU downscale that
 * box into a 32x32 tile and reading back 4 KB.
 *
 * The obvious alternative — getImageData over the whole working frame — copies
 * about 600 KB per frame and forces a pipeline stall each time. That is free
 * on a laptop and distinctly not free on a phone.
 */
function meanLuma(source, box) {
  const w = Math.max(1, box.x1 - box.x0);
  const h = Math.max(1, box.y1 - box.y0);
  const c = state.lumaCanvas;
  const ctx = c.getContext('2d', { willReadFrequently: true });
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = 'high';
  ctx.clearRect(0, 0, c.width, c.height);
  ctx.drawImage(source, box.x0, box.y0, w, h, 0, 0, c.width, c.height);
  return ctx.getImageData(0, 0, c.width, c.height).data;
}

// --- camera ----------------------------------------------------------------

function allocateBuffers(vw, vh) {
  const p = procSize(vw, vh);
  const canvas = makeCanvas(p.w, p.h);
  state.proc = {
    canvas,
    // The landmarker reads this canvas every frame; the hint keeps Chrome from
    // shuttling it to the GPU and back.
    ctx: canvas.getContext('2d', { willReadFrequently: true }),
    w: p.w,
    h: p.h,
  };

  const overlay = makeCanvas(p.w, p.h);
  state.overlay = {
    canvas: overlay,
    ctx: overlay.getContext('2d'),
    image: new ImageData(p.w, p.h),   // reused every frame, not reallocated
  };

  state.skinBuffer = M.createMask(p.w, p.h);
  state.skinValid = false;
  state.lumaCanvas = makeCanvas(32, 32);

  // Three capture-resolution snapshots, cycled. The peak detector picks the
  // MIDDLE of three samples, so by the time we know which frame won, the
  // camera has moved on ~120ms — the winning pixels have to already be held.
  // Capped rather than native: four canvases at the sensor's full 1080p is
  // memory an iPhone will not reliably give us.
  const capScale = Math.min(1, CFG.CAPTURE_MAX_LONG_EDGE / Math.max(vw, vh));
  const cw = Math.max(2, Math.round(vw * capScale));
  const ch = Math.max(2, Math.round(vh * capScale));
  state.captureSize = { w: cw, h: ch };
  state.ring = [0, 1, 2].map(() => makeCanvas(cw, ch));
  state.ringIndex = 0;

  els.view.width = cw;
  els.view.height = ch;
  state.viewCtx = els.view.getContext('2d');
}

/**
 * Record what this camera can actually do.
 *
 * Worth being blunt about the result on iOS: Safari exposes no exposure, ISO,
 * white-balance or sharpness control, and no ImageCapture API at all, so there
 * is no way to ask it for a less-processed frame. Chrome on Android often does
 * expose some of these. Rather than guess per device, log the real capability
 * set and keep the settings with every capture — so when a batch of images
 * comes out over-smoothed, you can see what the camera was doing at the time.
 */
function probeCamera(track) {
  if (!track) return;
  try {
    state.cameraCaps = track.getCapabilities ? track.getCapabilities() : {};
  } catch {
    state.cameraCaps = {};
  }
  try {
    state.cameraSettings = track.getSettings ? track.getSettings() : {};
  } catch {
    state.cameraSettings = {};
  }

  const controls = ['exposureMode', 'exposureTime', 'iso', 'whiteBalanceMode',
    'colorTemperature', 'focusMode', 'sharpness', 'contrast', 'saturation']
    .filter((k) => k in state.cameraCaps);
  // Surfaced in the debug panel, not just the console: on an iPhone the
  // console needs a tethered Mac, and this is exactly the answer you want on
  // the phone in your hand.
  state.cameraControls = controls.length ? controls.join(',') : 'none';

  console.log('[camera] settings', state.cameraSettings);
  console.log('[camera] capabilities', state.cameraCaps);
  console.log(controls.length
    ? `[camera] image controls available: ${controls.join(', ')}`
    : '[camera] no image-processing controls exposed (expected on iOS Safari) — '
      + 'light the subject well instead, and prefer colour over texture downstream');
}

async function startCamera() {
  if (!window.isSecureContext) {
    throw new Error('Camera access needs HTTPS. Open this page over https:// or on localhost.');
  }
  if (!navigator.mediaDevices?.getUserMedia) {
    throw new Error('This browser does not expose a camera API.');
  }

  stopStream();
  state.stream = await navigator.mediaDevices.getUserMedia({
    audio: false,
    video: {
      facingMode: state.facingMode,
      width: { ideal: CFG.REQUESTED_WIDTH },
      height: { ideal: CFG.REQUESTED_HEIGHT },
    },
  });

  probeCamera(state.stream.getVideoTracks()[0]);

  const video = state.video || document.createElement('video');
  // iOS Safari refuses to play an inline stream without all three of these,
  // and takes the video fullscreen instead.
  video.playsInline = true;
  video.muted = true;
  video.autoplay = true;
  video.setAttribute('playsinline', '');
  video.srcObject = state.stream;
  state.video = video;

  await new Promise((resolve, reject) => {
    video.onloadedmetadata = resolve;
    video.onerror = () => reject(new Error('The camera stream failed to load.'));
  });
  await video.play();

  allocateBuffers(video.videoWidth, video.videoHeight);
}

function stopStream() {
  if (state.stream) {
    for (const track of state.stream.getTracks()) track.stop();
    state.stream = null;
  }
}

function describeCameraError(err) {
  switch (err?.name) {
    case 'NotAllowedError':
    case 'SecurityError':
      return 'Camera permission was denied. Allow it in your browser settings, then reload.';
    case 'NotFoundError':
    case 'OverconstrainedError':
      return 'No suitable camera was found on this device.';
    case 'NotReadableError':
      return 'The camera is busy — close any other app or tab using it, then try again.';
    default:
      return err?.message || 'Could not start the camera.';
  }
}

// --- UI helpers ------------------------------------------------------------

/**
 * Gates have three states, not two. "No face yet" is not the same as "you are
 * standing too close", and showing both in warning amber trains people to
 * ignore the colour.
 */
function setGate(el, state, text) {
  el.classList.toggle('ok', state === 'ok');
  el.classList.toggle('bad', state === 'bad');
  el.classList.toggle('idle', state === 'idle');
  el.querySelector('.txt').textContent = text;
}

function renderProgress() {
  els.progress.innerHTML = CFG.GROUP_ORDER.map((g) => {
    const n = state.counts[g];
    const px = state.lastPx[g];
    return `<div class="chip${n ? ' done' : ''}">
      <div class="name">${groupLabel(g)}</div>
      <div class="val">${n}</div>
      <div class="px">${px == null ? '—' : `${px.toLocaleString()} px`}</div>
    </div>`;
  }).join('');
}

// --- the frame loop --------------------------------------------------------

/** MediaPipe requires strictly increasing timestamps in VIDEO mode. */
function nextTimestamp() {
  const t = Math.max(state.lastTimestamp + 1, Math.round(performance.now()));
  state.lastTimestamp = t;
  return t;
}

function loop() {
  state.raf = requestAnimationFrame(loop);

  const video = state.video;
  if (!video || video.readyState < 2 || !state.proc) return;

  const now = performance.now();
  if (state.lastFrameAt) {
    state.fps += (1000 / Math.max(1, now - state.lastFrameAt) - state.fps) * 0.1;
  }
  state.lastFrameAt = now;

  const { canvas: procCanvas, ctx: procCtx, w: pw, h: ph } = state.proc;
  const viewCtx = state.viewCtx;

  drawMirrored(procCtx, video, pw, ph);
  drawMirrored(viewCtx, video, els.view.width, els.view.height);

  const ts = nextTimestamp();
  const landmarks = detectLandmarks(state.vision.landmarker, procCanvas, ts);
  state.frameIndex++;

  if (!landmarks) {
    state.tracker.reset();
    state.skinValid = false;
    els.hint.textContent = 'No face detected';
    setGate(els.gateDistance, 'idle', 'Distance');
    setGate(els.gateLight, 'idle', 'Lighting');
    setGate(els.gatePose, 'idle', 'Pose');
    els.hold.hidden = true;
    els.hint.classList.remove('ready');
    if (DEBUG) els.debug.textContent = `fps    ${state.fps.toFixed(0)}\nno face`;
    drawFlash(now);
    return;
  }

  const ratio = symmetryRatio(landmarks, pw);
  const group = classifyRatio(ratio);

  // The segmenter is the expensive half of the pipeline, and its GPU->CPU mask
  // readback is the expensive half of that. The face barely moves in 33ms, so
  // we refresh it every Nth frame and reuse the last mask in between.
  if (state.frameIndex % CFG.SEG_EVERY_N_FRAMES === 0) {
    state.skinValid = writeSkinMask(
      state.vision.segmenterVideo, procCanvas, ts, pw, ph, state.skinBuffer,
    );
  }
  const skin = state.skinValid ? state.skinBuffer : null;

  // Built ONCE per frame. The Python rebuilt this same mask two or three times
  // per frame (contours, then the live count, then again on save) — invisible
  // on a desktop, not on a handset.
  const mask = R.buildRegionMask(landmarks, pw, ph, group, skin);
  const skinPx = R.countSkinPixels(mask);

  // Gates BEFORE the overlay is drawn, because the outline's colour is the
  // verdict and it cannot be coloured before the verdict exists.
  const bounds = R.faceBounds(landmarks, pw, ph);
  const lightBox = R.padBounds(bounds, pw, ph, CFG.BBOX_PADDING);
  const distance = checkDistance(bounds, pw, ph);
  const light = checkLighting(meanLuma(procCanvas, lightBox), 32, { x0: 0, y0: 0, x1: 32, y1: 32 });
  const pose = checkPose(group, ratio);
  const cooling = now < state.cooldownUntil;

  // Deliberately excludes the cooldown: your position is still good in the
  // three seconds after a shot, and flashing the outline back to red would
  // read as "you did something wrong" when you did not.
  const positionOk = distance.ok && light.ok && pose.ok;

  state.overlay.ctx.putImageData(
    M.maskToImageData(mask, pw, ph,
      positionOk ? CFG.OVERLAY_READY : CFG.OVERLAY_WAIT, state.overlay.image), 0, 0);
  viewCtx.drawImage(state.overlay.canvas, 0, 0, els.view.width, els.view.height);

  setGate(els.gateDistance, distance.ok ? 'ok' : 'bad', distance.msg);
  setGate(els.gateLight, light.ok ? 'ok' : 'bad', light.msg);
  setGate(els.gatePose, pose.ok ? 'ok' : 'bad', pose.msg);

  // One instruction at a time, in the order the user can act on them: get
  // close enough, get lit, then get the angle. Stacking all three corrections
  // at once just means none of them gets read.
  let hint;
  if (!distance.ok) hint = distance.msg;
  else if (!light.ok) hint = light.msg;
  else if (!pose.ok) hint = pose.msg;
  // "What next" belongs in the cooldown, not while you are being asked to hold
  // — telling someone to turn their head under a green outline and a filling
  // progress bar is two contradictory instructions at once.
  else if (cooling) hint = nextTargetHint(state.counts);
  else hint = 'Hold still';
  els.hint.textContent = hint;
  els.hint.classList.toggle('ready', positionOk);

  // A filling bar while the shot is being taken. Without it "hold still" asks
  // the user to freeze for an unknown length of time, which nobody does well.
  const held = state.tracker.samples.length;
  els.hold.hidden = !(positionOk && !cooling);
  els.holdBar.style.width = `${Math.min(100, (held / CFG.PEAK_WINDOW) * 100)}%`;

  if (DEBUG) {
    els.debug.textContent =
      `fps     ${state.fps.toFixed(0)}\n` +
      `cam     ${state.cameraSettings.width || '?'}x${state.cameraSettings.height || '?'}\n` +
      `proc    ${pw}x${ph}\n` +
      `ratio   ${ratio.toFixed(3)}  ${group}\n` +
      `face    ${distance.frac.toFixed(3)} of short edge\n` +
      `bright  ${light.brightness.toFixed(0)}\n` +
      `skin px ${skinPx.toLocaleString()}\n` +
      `seg     ${state.skinValid ? 'on' : 'OFF'}\n` +
      `ctrl    ${state.cameraControls || '?'}\n` +
      `shot    ${state.captureSize ? state.captureSize.w + 'x' + state.captureSize.h : '?'}` +
      `  try ${state.tries} ok ${state.saved} err ${state.failed}` +
      (state.lastError ? `\nerr     ${state.lastError}` : '');
  }

  const gatesOpen = distance.ok && light.ok && pose.ok && !cooling && !state.capturing;
  if (!gatesOpen) {
    state.tracker.reset();
  } else if (state.tracker.shouldSample(now)) {
    // Snapshot the full-resolution frame before sampling: if this one turns
    // out to be the peak, the camera will have moved on by the time we know.
    const frame = state.ring[state.ringIndex];
    state.ringIndex = (state.ringIndex + 1) % state.ring.length;
    drawMirrored(frame.getContext('2d'), video, frame.width, frame.height);

    const winner = state.tracker.push(now, ratio, group, {
      frame,
      group,
      ratio,
      brightness: light.brightness,
      landmarks: landmarks.map((p) => ({ x: p.x, y: p.y })),
    });
    if (winner) capture(winner, now);
  }

  drawFlash(now);
}

function drawFlash(now) {
  els.flash.hidden = now >= state.flashUntil;
}

/**
 * A capture error has to survive the frame loop.
 *
 * The hint is rewritten every frame, so anything put there lasts about 16ms —
 * long enough to be technically displayed and far too short to be read. This
 * banner stays up until the next successful capture, or until it is tapped.
 */
function showError(message) {
  state.lastError = message;
  els.errbar.textContent = `Capture failed — ${message}  (tap to dismiss)`;
  els.errbar.hidden = false;
}

function clearError() {
  state.lastError = '';
  els.errbar.hidden = true;
}

// --- capture ---------------------------------------------------------------

const toBlob = (canvas, type, quality) => new Promise((resolve, reject) => {
  canvas.toBlob((b) => (b ? resolve(b) : reject(new Error('toBlob returned null'))), type, quality);
});

async function makePreview(frame, mask, w, h, group) {
  const scale = Math.min(1, 480 / Math.max(w, h));
  const pw = Math.round(w * scale);
  const ph = Math.round(h * scale);
  const overlay = makeCanvas(w, h);
  // A saved capture passed every gate by definition, so its overlay is green.
  overlay.getContext('2d').putImageData(
    M.maskToImageData(mask, w, h, CFG.OVERLAY_READY), 0, 0);

  const out = makeCanvas(pw, ph);
  const ctx = out.getContext('2d');
  ctx.drawImage(frame, 0, 0, pw, ph);
  ctx.drawImage(overlay, 0, 0, pw, ph);
  return toBlob(out, 'image/jpeg', 0.8);
}

/**
 * Capture at FULL resolution.
 *
 * The live loop works on a downscaled frame for speed, but landmarks are
 * normalised 0..1, so the exact same region mask can be rebuilt at the
 * camera's native resolution for free — no upscaling of a low-res mask. The
 * only extra cost is one segmenter pass, which happens once per capture rather
 * than once per frame.
 */
async function capture(payload, now) {
  state.capturing = true;
  state.tries++;
  state.cooldownUntil = now + CFG.COOLDOWN_MS;
  state.flashUntil = now + CFG.CAPTURE_TEXT_MS;
  state.tracker.reset();

  try {
    const frame = payload.frame;
    const w = frame.width;
    const h = frame.height;

    const skin = M.createMask(w, h);
    const hasSkin = writeSkinMask(state.vision.segmenterImage, frame, null, w, h, skin);
    const mask = R.buildRegionMask(payload.landmarks, w, h, payload.group, hasSkin ? skin : null);
    const skinPx = R.countSkinPixels(mask);

    const maskCanvas = makeCanvas(w, h);
    maskCanvas.getContext('2d').putImageData(M.maskToBinaryImageData(mask, w, h), 0, 0);

    // Three files, deliberately. Python baked the contour lines into the saved
    // JPEG, which paints over the very skin pixels a later acne pass needs to
    // measure. The clean frame and the mask stay separate; the overlay is a
    // thumbnail only.
    // One at a time, not Promise.all: iOS Safari has a long history of
    // returning null from concurrent toBlob calls on large canvases, and our
    // toBlob helper turns a null into a thrown capture.
    const clean = await toBlob(frame, 'image/jpeg', CFG.JPEG_QUALITY);
    const maskBlob = await toBlob(maskCanvas, 'image/png');
    const preview = await makePreview(frame, mask, w, h, payload.group);

    const rec = {
      // Generated here, not by the server, so a retry re-sends the SAME id and
      // ingestion can be made idempotent on the far side.
      captureId: (crypto.randomUUID && crypto.randomUUID()) || `${Date.now()}-${Math.random()}`,
      sessionId: CFG.SESSION_ID,
      group: payload.group,
      ts: Date.now(),
      ratio: payload.ratio,
      brightness: payload.brightness,
      skinPx,
      width: w,
      height: h,
      // What the camera was doing when this was taken. There is no raw path on
      // a phone browser, so the next best thing is recording the conditions.
      camera: { ...state.cameraSettings },
      fillLight: document.body.classList.contains('lit'),
      clean,
      mask: maskBlob,
      preview,
      uploaded: false,
      uploadAttempts: 0,
      uploadError: null,
    };
    // Stored BEFORE any upload is attempted. The capture is safe on the device
    // from this line onwards, so a failed hand-off is only ever a retry.
    await store.saveCapture(rec);

    state.counts[payload.group]++;
    state.lastPx[payload.group] = skinPx;
    renderProgress();
    await refreshGalleryCount();

    console.log(`[saved] ${store.captureBasename(rec)} ` +
      `(ratio=${rec.ratio.toFixed(2)}, brightness=${rec.brightness.toFixed(0)}, skin_px=${skinPx})`);

    state.saved++;
    clearError();

    if (CFG.UPLOAD_ENDPOINT && CFG.UPLOAD_AUTO) runSync();
  } catch (err) {
    // Shown on screen, not just logged: reading a phone's console needs a
    // tethered Mac, and this is the message that explains an empty gallery.
    console.error('[error] capture failed:', err);
    state.failed++;
    // Into its own banner, NOT the hint: the frame loop rewrites the hint
    // about thirty times a second, so an error put there is gone before
    // anyone can read it — which is exactly how this stayed invisible.
    showError(`${err && err.name ? err.name + ': ' : ''}${(err && err.message) || err}`);
  } finally {
    state.capturing = false;
  }
}

// --- handing captures to the analysis server -------------------------------

/**
 * Drain the upload queue and reflect it in the button.
 *
 * Fire-and-forget from the capture path: uploading a couple of megabytes over
 * mobile data takes seconds, and the viewfinder must keep running throughout.
 */
async function runSync() {
  if (!CFG.UPLOAD_ENDPOINT) return;
  els.syncBtn.disabled = true;
  try {
    const result = await sync.syncPending({
      onProgress: ({ remaining }) => { els.syncBtn.textContent = `Sending ${remaining}…`; },
    });
    if (result.failed) {
      console.warn(`[sync] ${result.sent} sent, ${result.failed} still queued`);
    }
  } catch (err) {
    console.error('[sync] queue failed:', err);
  } finally {
    els.syncBtn.disabled = false;
    await refreshSyncBadge();
  }
}

async function refreshSyncBadge() {
  if (!CFG.UPLOAD_ENDPOINT) return;
  const n = await sync.pendingCount();
  els.syncBtn.textContent = n ? `Send ${n}` : 'All sent';
  els.syncBtn.classList.toggle('pending', n > 0);
}

/** A corner badge on the thumbnail: readable at a glance across a grid. */
function uploadBadge(rec) {
  if (!CFG.UPLOAD_ENDPOINT) return '';
  if (rec.uploaded) return '<span class="badge sent">Sent</span>';
  if (rec.uploadAttempts >= CFG.UPLOAD_MAX_ATTEMPTS) {
    return '<span class="badge failed">Failed</span>';
  }
  return '<span class="badge queued">Queued</span>';
}

/** The detail line under it — why something is queued, when that matters. */
function uploadStatus(rec) {
  if (!CFG.UPLOAD_ENDPOINT) return '';
  if (rec.uploaded) return '<span class="ok-txt">On the server</span>';
  if (rec.uploadAttempts >= CFG.UPLOAD_MAX_ATTEMPTS) {
    return `<span class="bad-txt">Gave up after ${rec.uploadAttempts}: ${rec.uploadError || 'upload failed'}</span>`;
  }
  if (rec.uploadError) return `<span class="warn-txt">Retrying: ${rec.uploadError}</span>`;
  return '<span class="warn-txt">Waiting to send</span>';
}

// --- gallery ---------------------------------------------------------------

let galleryUrls = [];

function releaseGalleryUrls() {
  for (const url of galleryUrls) URL.revokeObjectURL(url);
  galleryUrls = [];
}

async function refreshGalleryCount() {
  const rows = await store.listCaptures();
  els.galleryCount.textContent = String(rows.length);
  return rows;
}

async function openGallery() {
  const rows = await refreshGalleryCount();

  // A link straight to the server's own view. The gallery can only report what
  // this device believes it sent; that page shows what actually landed. When
  // the two disagree, the disagreement is the bug.
  if (CFG.UPLOAD_ENDPOINT) {
    els.serverLink.href = new URL('__received', new URL(CFG.UPLOAD_ENDPOINT, location.href)).href;
    els.serverLink.hidden = false;
  }
  releaseGalleryUrls();
  els.galleryEmpty.hidden = rows.length > 0;
  els.galleryGrid.innerHTML = '';

  for (const rec of rows) {
    const url = URL.createObjectURL(rec.preview);
    galleryUrls.push(url);

    const card = document.createElement('div');
    card.className = 'card';
    card.innerHTML = `
      <div class="thumb"><img src="${url}" alt="${groupLabel(rec.group)} capture">${uploadBadge(rec)}</div>
      <div class="meta">
        <b>${groupLabel(rec.group)}</b>
        <span>${rec.skinPx.toLocaleString()} skin px</span>
        <span>${new Date(rec.ts).toLocaleString()}</span>
        <span>${uploadStatus(rec)}</span>
      </div>
      <div class="row">
        <button class="ghost" data-act="export">Export</button>
        <button class="ghost danger" data-act="delete">Delete</button>
      </div>`;

    card.querySelector('[data-act="export"]').addEventListener('click', async (e) => {
      const btn = e.currentTarget;
      btn.disabled = true;
      try {
        await store.exportCapture(rec);
      } catch (err) {
        console.error('[error] export failed:', err);
      } finally {
        btn.disabled = false;
      }
    });

    card.querySelector('[data-act="delete"]').addEventListener('click', async () => {
      await store.deleteCapture(rec.id);
      state.counts[rec.group] = Math.max(0, state.counts[rec.group] - 1);
      renderProgress();
      await openGallery();
    });

    els.galleryGrid.appendChild(card);
  }

  els.gallery.hidden = false;
}

function closeGallery() {
  els.gallery.hidden = true;
  releaseGalleryUrls();
}

// --- boot ------------------------------------------------------------------

// Kick the model downloads off immediately. They need no permission, and by
// the time the user has read the intro and tapped Start, ~10 MB of WASM and
// weights are usually already in the HTTP cache.
const visionPromise = loadVision({
  onProgress: (m) => { els.startMsg.textContent = m; },
});
visionPromise.catch((err) => console.error('[error] model load failed:', err));

async function restoreCounts() {
  const rows = await store.listCaptures();
  for (const rec of rows) {
    state.counts[rec.group]++;
    if (state.lastPx[rec.group] == null) state.lastPx[rec.group] = rec.skinPx;
  }
  els.galleryCount.textContent = String(rows.length);
  renderProgress();
}

async function showFlipIfMultipleCameras() {
  try {
    const devices = await navigator.mediaDevices.enumerateDevices();
    els.flipBtn.hidden = devices.filter((d) => d.kind === 'videoinput').length < 2;
  } catch {
    els.flipBtn.hidden = true;
  }
}

async function begin() {
  els.startBtn.disabled = true;
  els.startMsg.classList.remove('error');
  try {
    // Camera first, while the tap is still "fresh". Awaiting the model load
    // before this would spend the user-gesture context and make iOS Safari far
    // more likely to refuse the stream.
    els.startMsg.textContent = 'Requesting camera…';
    await startCamera();

    els.startMsg.textContent = 'Loading models…';
    state.vision = await visionPromise;

    els.startOverlay.hidden = true;
    els.gates.hidden = false;
    els.progress.hidden = false;
    els.galleryBtn.hidden = false;
    els.lightBtn.hidden = false;
    els.syncBtn.hidden = !CFG.UPLOAD_ENDPOINT;
    els.debug.hidden = !DEBUG;
    await showFlipIfMultipleCameras();

    cancelAnimationFrame(state.raf);
    loop();
  } catch (err) {
    console.error('[error]', err);
    els.startMsg.textContent = describeCameraError(err);
    els.startMsg.classList.add('error');
    els.startBtn.disabled = false;
  }
}

els.startBtn.addEventListener('click', begin);
els.lightBtn.addEventListener('click', () => {
  const on = document.body.classList.toggle('lit');
  els.fillLight.hidden = !on;
  els.lightBtn.textContent = on ? 'Light off' : 'Fill light';
});

els.errbar.addEventListener('click', clearError);
els.syncBtn.addEventListener('click', runSync);
els.galleryBtn.addEventListener('click', openGallery);
els.closeGalleryBtn.addEventListener('click', closeGallery);

els.clearBtn.addEventListener('click', async () => {
  if (!confirm('Delete every capture on this device? This cannot be undone.')) return;
  await store.clearAll();
  state.counts = { GROUP_1: 0, GROUP_2: 0, GROUP_3: 0 };
  state.lastPx = { GROUP_1: null, GROUP_2: null, GROUP_3: null };
  renderProgress();
  await openGallery();
});

els.flipBtn.addEventListener('click', async () => {
  els.flipBtn.disabled = true;
  cancelAnimationFrame(state.raf);
  state.facingMode = state.facingMode === 'user' ? 'environment' : 'user';
  try {
    await startCamera();
  } catch (err) {
    console.error('[error] camera switch failed:', err);
    state.facingMode = state.facingMode === 'user' ? 'environment' : 'user';
    await startCamera();
  } finally {
    state.tracker.reset();
    state.lastFrameAt = 0;
    els.flipBtn.disabled = false;
    loop();
  }
});

// Backgrounding the tab should not keep the camera light on or burn battery.
document.addEventListener('visibilitychange', () => {
  if (document.hidden) {
    cancelAnimationFrame(state.raf);
    state.raf = 0;
    state.tracker.reset();
  } else if (state.vision && state.stream && !state.raf) {
    state.lastFrameAt = 0;
    loop();
  }
});

window.addEventListener('pagehide', stopStream);

renderProgress();
// Retry the queue whenever the phone regains signal — a capture taken in a
// basement should still reach the server on the walk out.
sync.watchConnectivity(() => refreshGalleryCount());
refreshSyncBadge().catch(() => {});
// If IndexedDB is unavailable — a private tab on iOS is the usual reason —
// every capture would appear to succeed and silently vanish. Better to say so
// on load than to let someone shoot a whole session into nothing.
restoreCounts().catch((err) => {
  console.error('[error] could not read past captures:', err);
  showError(`storage unavailable (${(err && err.message) || err}). `
    + 'Captures cannot be saved — a private browsing tab is the usual cause.');
});
