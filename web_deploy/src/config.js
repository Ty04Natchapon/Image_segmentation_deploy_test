/**
 * config.js — every tunable constant, ported from capture_prototype.py.
 *
 * Values marked [RETUNE] were tuned against a landscape laptop webcam and do
 * NOT transfer to a phone front camera unchanged. See README "Tuning".
 */

// --- MediaPipe assets ------------------------------------------------------
export const TASKS_VISION_VERSION = '1.0.1';
export const WASM_ROOT =
  `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${TASKS_VISION_VERSION}/wasm`;
export const FACE_LANDMARKER_MODEL =
  'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task';
export const SEGMENTER_MODEL =
  'https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite';

// Selfie Multiclass classes: 0 bg, 1 hair, 2 body-skin, 3 face-skin, 4 clothes, 5 other
export const FACE_SKIN_CLASS = 3;

// --- Camera / resolution ---------------------------------------------------
// Requested capture resolution. The saved images use whatever the camera
// actually gives us; PROC_LONG_EDGE only controls the live inference copy.
//
// Ask high on purpose. A phone ISP applies noise reduction that smooths exactly
// the fine skin detail an acne pass needs, and there is no web API to turn it
// off — so the only counter is to start with more real pixels. 1080p is the
// practical ceiling: at 4K the full-resolution mask build crosses a second.
export const REQUESTED_WIDTH = 1920;
export const REQUESTED_HEIGHT = 1080;

// ...but cap what we actually hold and process. iOS Safari is far stricter
// about canvas memory than a desktop: the capture path keeps three
// full-resolution snapshots plus the view canvas, and at 1080p that is ~33 MB
// of backing store before a capture even starts, on top of a mask build over
// two million pixels. Capping the long edge keeps the ceiling predictable.
// Landmarks are normalised, so the region mask still rebuilds exactly at
// whatever size this yields.
export const CAPTURE_MAX_LONG_EDGE = 1280;

// Live loop runs landmarks + masks on a downscaled copy (long edge, px).
// On capture we rebuild the mask at full resolution — landmarks are
// normalised 0..1, so they scale for free.
export const PROC_LONG_EDGE = 512;

// Run the segmenter every Nth processed frame and reuse the last mask in
// between. The face barely moves in 33ms; this is the main mobile perf lever.
export const SEG_EVERY_N_FRAMES = 2;

// --- Pose thresholds (symmetry ratio = dist_left / dist_right) -------------
export const RATIO_LEFT_CHEEK_THRESHOLD = 1.5;   // ratio > this -> GROUP_2
export const RATIO_RIGHT_CHEEK_THRESHOLD = 0.6;  // ratio < this -> GROUP_1
export const FRONT_SYMMETRY_TOLERANCE = 0.15;    // |ratio - 1| <= this for front

// --- Distance thresholds [RETUNE] -----------------------------------------
// Python measured the *padded* bbox against frame WIDTH. That breaks in
// portrait: a phone frame is tall, so the same face reads as a much smaller
// fraction. We measure the *unpadded* landmark span against the SHORTER frame
// edge, which is orientation-independent. Values below are the Python
// thresholds algebraically converted for a 640x480 sensor — treat them as a
// starting point and re-measure on a real handset (open with ?debug=1).
export const FACE_WIDTH_MIN_FRAC = 0.20;  // smaller -> "MOVE CLOSER"
export const FACE_WIDTH_MAX_FRAC = 0.58;  // larger  -> "MOVE BACK"

// --- Lighting thresholds ---------------------------------------------------
export const BRIGHTNESS_MIN = 70;
export const BRIGHTNESS_MAX = 205;
export const BBOX_PADDING = 0.25;   // padding used for the lighting ROI only

// --- Capture timing --------------------------------------------------------
export const COOLDOWN_MS = 3000;
export const CAPTURE_TEXT_MS = 1000;

// The Python peak detector used a 3-FRAME window, so its real duration was at
// the mercy of the frame rate (100ms at 30fps, 200ms at 15fps). We sample on a
// fixed clock instead, so the window is ~180ms on any device.
export const PEAK_SAMPLE_INTERVAL_MS = 60;
export const PEAK_WINDOW = 3;

// --- Region building -------------------------------------------------------
// These radii are in PIXELS, and Python only ever ran at one resolution, so it
// could hard-code them. We work at two (a downscaled live frame and the
// full-resolution capture), and a 6px hole dilation means something different
// at 288px tall than at 720px. Both are therefore expressed relative to the
// resolution they were tuned at and scaled per frame — otherwise the live
// pixel count and the saved pixel count would disagree.
export const REFERENCE_SHORT_EDGE = 480;   // the webcam the Python was tuned on
export const HOLE_DILATE_PX = 6;           // eye/mouth exclusion dilation
export const SKIN_OPEN_PX = 2;             // morphological open on the skin mask

/** Scale a radius tuned at REFERENCE_SHORT_EDGE to the current frame size. */
export function scaleRadius(px, w, h) {
  return Math.max(1, Math.round(px * Math.min(w, h) / REFERENCE_SHORT_EDGE));
}

// Python only ran the segmenter for the FRONT shot, so a cheek's "skin pixel"
// count was really just geometric area — it included hair, beard and shadow.
// With this on, every region is intersected with the real face-skin mask, so
// all three counts mean the same thing and the acne ratio stays comparable.
export const INTERSECT_SKIN_ALL_REGIONS = true;

// --- Output ----------------------------------------------------------------
export const JPEG_QUALITY = 0.95;

// --- Hand-off to the analysis server ---------------------------------------
// Where captures are POSTed for the downstream algorithm. Null disables
// uploading entirely and the app stays fully on-device, which is the right
// default until there is a server to receive them and consent to send them.
//
// Override without editing this file by opening the page with ?api=<url> —
// useful for pointing a test handset at a laptop running tools/mock_server.py.
// Guarded so this module can also be imported outside a browser — the contract
// test in test/sync.test.js runs under Node, where `location` does not exist.
export const UPLOAD_ENDPOINT = new URLSearchParams(
  typeof location !== 'undefined' ? location.search : '',
).get('api') || null;

// Upload as soon as a capture is taken. With this off, captures queue locally
// and go up only when the user taps Sync.
export const UPLOAD_AUTO = true;

// A phone on mobile data WILL fail mid-upload. Captures are already durable in
// IndexedDB, so a failure is a retry, never a lost capture.
export const UPLOAD_MAX_ATTEMPTS = 5;
export const UPLOAD_TIMEOUT_MS = 30000;

// Sent with every capture so the server can group the three angles taken in
// one sitting. Regenerated on each page load.
export const SESSION_ID = (crypto.randomUUID && crypto.randomUUID())
  || String(Date.now()) + Math.random().toString(16).slice(2);

export const APP_VERSION = '1.0.0';

export const GROUPS = {
  GROUP_1: { dir: 'Group_1_Right_Cheek', label: 'Right cheek', color: [255, 64, 64] },
  GROUP_2: { dir: 'Group_2_Left_Cheek',  label: 'Left cheek',  color: [255, 64, 64] },
  GROUP_3: { dir: 'Group_3_Front',       label: 'Front',       color: [64, 224, 96] },
};
export const GROUP_ORDER = ['GROUP_3', 'GROUP_2', 'GROUP_1'];
