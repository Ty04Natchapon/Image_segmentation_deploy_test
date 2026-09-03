# Skin Capture — web build

A browser port of `../original code/capture_prototype.py`. It runs the same
pipeline — MediaPipe Face Mesh landmarks, Selfie Multiclass skin segmentation,
the same region polygons, the same three quality gates, the same peak-capture
rule — entirely on the user's device, so it works on a phone with no server and
no upload.

Static files only. No build step, no bundler, no npm install to deploy.

```
index.html            markup
styles.css            styling
manifest.webmanifest  PWA metadata (Add to Home Screen)
src/config.js         every tunable constant
src/landmarks.js      MediaPipe landmark indices, copied verbatim from the Python
src/mask.js           the binary-raster ops that replace OpenCV
src/regions.js        build_region_mask() and friends
src/pose.js           ratio, classification, gates, peak detector
src/vision.js         model loading + face-skin mask extraction
src/storage.js        IndexedDB, export, optional upload
test/                 27 Node tests over the pure logic
```

## Run it locally

```bash
npm run dev            # serves this folder on http://localhost:8000
npm test               # runs the geometry and gate tests
```

`http://localhost` counts as a secure origin, so the camera works there.

## Test it on a phone

`getUserMedia` is blocked on plain HTTP everywhere except localhost, so
`http://192.168.x.x:8000` from your phone **will not work** — the camera will
never prompt. Two ways round it:

```bash
npm run dev            # terminal 1
npm run tunnel         # terminal 2 -> gives you an https:// URL
```

or just deploy a preview (below) and open that. Deploying is usually faster.

## Deploy

Any static host with HTTPS. Pick one:

```bash
npx vercel             # reads vercel.json
npx netlify deploy     # reads netlify.toml
npx wrangler pages deploy .
```

GitHub Pages works too — push this folder and enable Pages; you lose only the
custom headers, which are hardening, not requirements.

**Do not enable cross-origin isolation** (`Cross-Origin-Embedder-Policy`). It
would block both the WASM on jsDelivr and the models on `storage.googleapis.com`,
neither of which sends a CORP header, and the single-threaded SIMD build here
does not need it. The supplied configs deliberately leave it off.

## Tuning on real hardware

Open the deployed page with `?debug=1` for a live readout of frame rate, the
symmetry ratio, face width as a fraction of the short edge, brightness, region
pixel count, and whether the segmenter is running.

**Start with the distance thresholds.** They are the one thing most likely to
look broken on first contact with a phone, and it is not the port's fault. The
Python compared the *padded* face box against frame WIDTH, tuned on a landscape
webcam. A phone front camera is narrower-FOV and portrait, so the same face
fills far more of the frame and reads as "move back" constantly. This build
measures the *unpadded* landmark span against the SHORTER frame edge, which is
orientation-independent, and `FACE_WIDTH_MIN_FRAC` / `FACE_WIDTH_MAX_FRAC` in
`src/config.js` are the Python values converted for a 640x480 sensor. Hold a
phone at the distance you actually want, read `face` off the debug panel, and
set the bounds around it.

Other constants worth a look, all in `src/config.js`:

| Constant | What it does |
| --- | --- |
| `PROC_LONG_EDGE` | live inference resolution; drop to 384 on a slow handset |
| `SEG_EVERY_N_FRAMES` | raise to 3 if the frame rate sags; the mask is reused in between |
| `BRIGHTNESS_MIN/MAX` | lighting gate — see the caveat below |
| `PEAK_SAMPLE_INTERVAL_MS` | how long the capture window spans (3 samples) |
| `INTERSECT_SKIN_ALL_REGIONS` | set false to reproduce the Python's cheek numbers exactly |

## What is identical to the Python, and what is not

**Identical.** Every landmark index (`FACE_OVAL`, `FRONT_RIGHT_EDGE`, the eye
and lip rings) is copied verbatim — the web `FaceLandmarker` returns the same
478-point topology as `FaceMesh(refine_landmarks=True)`, so the numbers carry
over unchanged. The symmetry ratio, the 1.5 / 0.6 / ±0.15 thresholds, the
region construction, the 3-second cooldown and the "middle of three is a local
extreme" capture rule all behave as before. It is the same segmentation model
file from the same bucket.

**Changed on purpose — four fixes:**

1. **Cheek pixel counts now mean the same thing as front ones.** The Python ran
   the segmenter only for the front shot, so a cheek's "skin pixel" count was
   really geometric area: hair, beard and shadow inside the face oval all
   counted. Every region is now intersected with the real face-skin mask, so the
   `acne / skin` ratio you are building toward stays comparable across regions.

2. **The saved image is clean.** The Python drew 3px contour lines onto the
   frame and saved *that*, painting over the very boundary skin a later
   per-pixel pass needs to measure. Each capture now stores three files: a clean
   JPEG, the region mask as a PNG, and a separate overlay thumbnail.

3. **The capture window is time-based.** A 3-*frame* window lasted 100ms at
   30fps and 200ms at 15fps, so the peak got mushier the slower the device.
   Sampling on a 60ms clock keeps it at ~180ms on any handset.

4. **Morphology radii scale with resolution.** The 6px hole dilation and 2px
   skin open were hard-coded for one webcam. This build works at two resolutions
   — a downscaled live frame and the full-resolution capture — so both radii
   scale, and the live count matches the saved count.

Two other things are worth knowing but were not "fixed", because they are
inherited behaviour rather than defects:

- **Cheek pose feedback cannot fire.** `classifyRatio` uses the same 1.5 / 0.6
  thresholds that `checkPose` then re-tests, so a cheek shot passes by
  construction and can only ever read "hold still". Only the front has a
  genuinely tighter gate. Rather than restructure the algorithm, the UI now
  tracks which of the three angles you still need and prompts for the next one.
- **The lighting gate is weaker on a phone.** Mobile auto-exposure normalises
  brightness hard, so frames shot in genuinely bad light still pass. It catches
  gross failures only. Auto white balance also shifts skin hue frame to frame,
  which will matter once you detect acne by colour — budget for a colour
  constancy step or a reference card.

## Captures and privacy

Nothing is uploaded. Captures live in IndexedDB on the device and are exported
only when the user taps Export, which goes through the OS share sheet on mobile
(iOS Safari treats `<a download>` on a blob URL as "open in a new tab", so the
share sheet is the reliable path) and falls back to real downloads on desktop.

For a project handling face images this is worth stating in your report: running
in the browser is not just a convenience, it means the data never leaves the
participant's phone unless they choose to hand it over.

### Sending captures to a server

`uploadCapture(rec, endpoint)` in `src/storage.js` posts a capture as multipart
form data (`image`, `mask`, plus `group`, `timestamp`, `skin_px`, `ratio`,
`brightness`, `width`, `height`). Nothing calls it — wire it up once you have
consent and somewhere to put the data.

### Vendoring the models

By default the two models stream from `storage.googleapis.com` and the WASM from
jsDelivr. To serve them yourself — for offline use, or if your marker wants the
build self-contained — drop the files in `models/` and repoint
`FACE_LANDMARKER_MODEL`, `SEGMENTER_MODEL` and `WASM_ROOT` in `src/config.js`.
Both deploy configs already cache `/models/*` immutably.

```bash
mkdir -p models
curl -o models/face_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
curl -o models/selfie_multiclass_256x256.tflite \
  https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite
```

## Known limits

- The full-resolution mask build on capture takes 50–150ms and will hitch one
  frame. It happens inside the 3-second cooldown, so it is not visible, but if
  you raise the capture resolution much further, move it to a Web Worker.
- The rear camera is mirrored like the front one, so the GROUP_1 / GROUP_2
  labels stay consistent. Saved rear-camera images are therefore flipped.
- Region masks are still purely geometric plus skin segmentation. Acne
  detection is not implemented here, in either version.
