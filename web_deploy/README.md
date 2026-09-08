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

## Image quality: can you get a raw, unprocessed frame?

**On iOS Safari, no.** There is no web API for it:

- The `ImageCapture` API (`takePhoto`, `getPhotoCapabilities`, photo settings) is
  not implemented in Safari at all.
- `MediaTrackConstraints` for `exposureMode`, `exposureTime`, `iso`,
  `whiteBalanceMode`, `colorTemperature` and `sharpness` are not exposed either.
  Open the app with `?debug=1` and read the `ctrl` line — on an iPhone it says
  `none`, which is the honest answer for that device. Chrome on Android usually
  lists several.

One useful clarification: the `getUserMedia` video stream is **not** the Camera
app's photo pipeline. It does not get Deep Fusion, Smart HDR or Photographic
Styles, so what you are capturing is already less processed than a normal
iPhone photo. What it does still get is auto-exposure, auto white balance, and
noise reduction plus sharpening — and noise reduction is the one that erases
acne detail.

### What actually helps, strongest first

1. **Light the subject.** Noise reduction strength tracks sensor gain. In bright
   even light the sensor runs at low ISO and the ISP barely smooths at all; in a
   dim room it smooths hard. This is the biggest lever by a wide margin and the
   only one fully under your control. Use a window or a lamp. The **Fill light**
   button is the fallback when neither is available: it floods the screen white
   around a shrunken viewfinder, so the phone itself becomes the lamp.

2. **Resolution.** The capture now requests 1080p rather than 720p. More pixels
   per millimetre of skin means a fixed-radius smoothing kernel removes
   proportionally less of what you care about.

3. **Detect by colour, not texture.** This is the one that matters most for the
   acne pass you have not written yet. Noise reduction attacks *luminance*
   detail aggressively but leaves *chroma* blobs largely intact — and acne is
   fundamentally a redness signal, not a texture signal. Work in CIELAB `a*`, or
   an erythema index such as `log(R) - log(G)`, rather than on grayscale
   gradients or high-pass texture. A texture-based detector spends its life
   fighting the ISP; a chroma-based one mostly does not notice it.

4. **Normalise per image.** Auto white balance drifts between shots, so absolute
   colour thresholds will not hold across a session. Cheap fix: take the median
   colour *inside the region mask* as that image's skin reference and measure
   redness relative to it. You already have the mask, so this costs almost
   nothing and makes the measurement self-calibrating.

Every capture now records `camera` (the live `MediaTrackSettings` — resolution,
frame rate, and whatever else the platform reports) and `fillLight`, so when a
batch comes out over-smoothed you can check what the camera was doing instead of
guessing.

### If you genuinely need raw

A native iOS app can do it — `AVCapturePhotoOutput` will hand you Bayer RAW /
DNG with the ISP bypassed. That is a different project from a web app, and for
a ratio-based measure it is almost certainly not worth it. An Android handset is
the cheaper middle ground: Chrome there exposes `ImageCapture` and several
`MediaTrackConstraints`, so you can at least pin exposure and white balance.

## Sending captures to the analysis server

This app's responsibility ends at the POST. It guides the shot, segments the
region, decides the frame is good, and hands the result over; everything after
that belongs to whoever owns the analysis algorithm.

**Give the receiving team this section.** It is the whole interface.

### The request

One `multipart/form-data` POST per capture, to whatever URL is configured as
`UPLOAD_ENDPOINT`.

| Field | Type | Meaning |
| --- | --- | --- |
| `capture_id` | string | UUID generated on the device. **Stable across retries** — use it to make ingestion idempotent |
| `session_id` | string | Groups the three angles taken in one sitting |
| `group` | string | `GROUP_1` / `GROUP_2` / `GROUP_3` |
| `region` | string | `right_cheek` / `left_cheek` / `front` — the same thing, readable |
| `captured_at` | string | ISO 8601 UTC |
| `skin_px` | integer | Pixels inside the region mask |
| `face_px` | integer | Face width in pixels. **Divide by 140 for px/mm** — the resolution filter |
| `width`, `height` | integer | Dimensions of **both** files |
| `ratio` | float | Pose symmetry: 1.0 is head-on |
| `brightness` | float | Mean luma of the face box, 0–255 |
| `fill_light` | bool | Whether the screen fill light was on |
| `camera` | JSON | `MediaTrackSettings` as reported by the device |
| `app_version` | string | This app's version |
| `image` | file | **The clean JPEG. Nothing is drawn on it** |
| `mask` | file | 8-bit PNG, white = region skin |

Two things that are easy to get wrong and matter downstream:

- **`image` has no annotations.** No contour lines, no overlay. The Python
  prototype baked green/red contours into its saved JPEGs, which painted over
  the very boundary pixels an analysis pass needs. That was fixed here
  deliberately — do not reintroduce it.
- **`mask` is pixel-aligned with `image`,** identical dimensions. Restricting
  analysis to the region is a straight boolean AND. No resampling, no
  coordinate transform, no scaling factor to agree on.

### The response

Any 2xx means accepted; the body is ignored. Anything else is treated as a
failure and retried up to `UPLOAD_MAX_ATTEMPTS` times.

**Please return 2xx for a `capture_id` already stored.** Phones lose signal
mid-upload, so retries of an already-received capture are normal, not an error.

**CORS is required.** The app is served from a different origin than the API,
so responses need `Access-Control-Allow-Origin`. Without it every upload looks
like a network failure to the browser, even when the server stored the file
perfectly.

### What happens when the network fails

Captures are written to IndexedDB **before** any upload is attempted, so a
failed hand-off is a retry, never a lost capture. The queue drains oldest
first, one at a time, and retries after each new capture, when the phone comes
back online, and when the user taps **Send**. The gallery shows per-capture
status, and the Send button carries the pending count.

### Testing without the real server

`tools/mock_server.py` implements exactly the contract above — no dependencies,
standard library only. It stores captures in the same folder layout the Python
prototype produced, so you can eyeball them:

```bash
python tools/mock_server.py            # listens on :8001, writes ./received
```

**You do not normally need to configure anything.** At boot the app asks its
own origin whether it accepts captures, by fetching `/__health`. The dev server
answers, so uploading turns itself on; a static host like GitHub Pages returns
404, so it stays off rather than failing every capture against a server that
was never there.

To override — a real API on another host, say — add `?api=` to the URL. It is
remembered for that origin, so it survives a home-screen launch where the
manifest drops the query string. `?api=off` turns uploading off and forgets it.

```
http://localhost:8000/?api=http://localhost:8001
```

From a phone, the API needs its own HTTPS tunnel, because an HTTPS page cannot
POST to a plain-HTTP address:

```bash
python tools/mock_server.py                                # terminal 1
npx --yes cloudflared tunnel --url http://localhost:8001   # terminal 2
```

```
https://<your-site>/?api=https://<tunnel>.trycloudflare.com
```

`test/sync.test.js` locks the field names down. If you change the contract in
`buildFormData`, those tests fail — that is the reminder to tell the receiving
team and update the mock server, rather than breaking them silently.

### Before you turn this on for real

`UPLOAD_ENDPOINT` defaults to `null`, so the app stays fully on-device until
someone deliberately configures it. Face images leaving a participant's phone
is the point at which most year-4 projects need consent forms and departmental
ethics sign-off. Worth settling that before wiring in a live endpoint.

### Testing the whole thing on a phone with one tunnel

`npm run dev` serves the app; `tools/mock_server.py` is the API. Running them
on separate ports means two tunnels and a CORS round trip, which is a lot of
moving parts to debug at once. `--serve` collapses them into one process:

```bash
npm run dev:api     # app AND API on :8000
npm run tunnel      # one HTTPS URL for both
```

Then open the printed tunnel URL on the phone with `?api=/` — a relative
endpoint, so uploads go back to the same origin the page came from:

```
https://<tunnel>.trycloudflare.com/?debug=1&api=/
```

Same origin means no CORS involved at all, and nothing needs deploying: this
serves your working tree, so an edit is live on refresh. Captures land in
`received/<session_id>/Group_3_Front/` next to the app.

Useful while iterating:

- `GET /__health` returns `{"status": "ok", "received": N}` — a quick check that
  captures are arriving without digging through folders.
- Responses are sent `Cache-Control: no-store`, so a stale `main.js` never
  survives a refresh.

### Checking what actually reached the server

The app's gallery can only tell you what this device *believes* it sent. The
server's own view is the other end of the wire:

```
https://<tunnel>/__received
```

A grid of everything on the server's disk — the clean image and its mask side
by side, with region, skin pixel count, dimensions, ratio, brightness, fill
light, and the capture and session ids. It polls `/__health` and reloads when
the count changes, so you can leave it open on a laptop while shooting on a
phone and watch captures land.

In the app, Gallery → **On server** opens the same page, and each thumbnail
carries a badge:

| Badge | Meaning |
| --- | --- |
| **Sent** | the server accepted it |
| **Queued** | still on the device, waiting to go |
| **Failed** | gave up after `UPLOAD_MAX_ATTEMPTS`; the detail line says why |

When the gallery says Sent and `/__received` does not show it, that gap is the
bug — and it is worth knowing which side to look at.

### Reading the phone's console from your laptop

An iPhone has no readable console without a tethered Mac, which makes any bug
that only reproduces on the handset hard to chase. With an endpoint configured,
`src/remotelog.js` forwards errors — including uncaught throws and unhandled
promise rejections — to `POST /__log`, and the dev server prints them:

```
[16:34:10] PHONE error: QuotaExceededError: storage full
           at capture (main.js:479)
```

Diagnostics only: it is inert without an endpoint, and never sends image data.

## Swapping the mock server for the real one

Nothing in the app is specific to `mock_server.py`. It POSTs multipart to a URL
and wants a 2xx back — any stack that honours the contract above works
unchanged. Point it at the real API by setting one constant in `src/config.js`:

```js
export const DEFAULT_UPLOAD_ENDPOINT = 'https://api.example.com/captures';
```

That takes precedence over the same-origin probe, so the deployed app posts to
the real API with no query string and nothing to configure per device. If the
API needs a key, add it to `UPLOAD_HEADERS` — but read the preflight note below
before you do.

### Four things that only break with a real server

The mock server sits on the same origin, unauthenticated, with no proxy in
front. A real one usually has none of those properties, and each difference has
a failure mode that looks like "uploads just don't work":

1. **HTTPS is mandatory.** The page is served over https, and a browser blocks
   a plain-http request from an https page as mixed content — with no error the
   app can catch. An `http://` endpoint fails 100% of the time, silently.

2. **CORS.** A different origin means the response needs
   `Access-Control-Allow-Origin`. Without it the browser discards the response
   and the upload looks like a network failure *even when the server stored the
   file perfectly* — so check the server's own logs before believing the app.

3. **Preflight, if you add headers.** A multipart POST with no custom headers
   is a "simple" request and goes straight out. Add `Authorization` — or
   anything else — and it becomes preflighted, so the server must also answer
   `OPTIONS` with the matching `Access-Control-Allow-Headers`. This is the
   usual reason a request that works in curl fails in the browser.

4. **Body size limits.** A capture is roughly 200 KB–1 MB across the two files.
   nginx defaults to 1 MB (`client_max_body_size`), and several frameworks
   default lower. The symptom is HTTP 413 on the larger captures only, which
   reads as intermittent.

### A reference implementation

The contract in FastAPI, matching `mock_server.py` behaviour:

```python
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"])

@app.post("/captures")
async def receive(
    capture_id: str = Form(...),
    session_id: str = Form(...),
    group: str = Form(...),
    region: str = Form(...),
    skin_px: int = Form(...),
    width: int = Form(...),
    height: int = Form(...),
    image: UploadFile = File(...),
    mask: UploadFile = File(...),
):
    if already_stored(capture_id):
        return {"status": "duplicate"}      # 2xx, so the client stops retrying
    store(capture_id, await image.read(), await mask.read())
    return {"status": "stored"}
```

The one non-obvious requirement: **return 2xx for a `capture_id` you already
hold.** Phones lose signal mid-upload, so retries of an already-received
capture are routine. Answering with an error makes the client retry until it
gives up, and the capture never leaves the device.

## Is the resolution enough to train on?

File size is the wrong question — it is an output, not an input. What decides
whether a lesion is learnable is **pixels per millimetre of skin**, and that
depends on the capture resolution and how much of the frame the face fills.

Open with `?debug=1` and read it live:

```
face    0.412 of short edge  3.0 px/mm
```

An adult face is about 140 mm wide, so px/mm is just `face_px / 140`. What the
current settings produce, on a 1080p sensor capped to a 1280 long edge:

| Face fills | face px | px/mm | a 3 mm papule spans |
| --- | --- | --- | --- |
| 0.20 (minimum gate) | 144 | 1.0 | **3 px** |
| 0.40 (typical) | 288 | 2.1 | 6 px |
| 0.58 (maximum gate) | 418 | 3.0 | 9 px |

**The bottom row of that table is the problem.** At the minimum the distance
gate currently allows, a papule is three pixels across — no model learns from
that, and no JPEG quality setting recovers it. The gate is inherited from the
Python, where it controlled framing, not data quality; it was never chosen with
a training set in mind.

Two levers, in order of effect:

1. **Raise `FACE_WIDTH_MIN_FRAC`** so under-resolved captures are refused
   rather than collected. Asking for 3 px/mm means a face filling about 0.40 of
   the short edge; the user simply holds the phone closer.
2. **Raise `CAPTURE_MAX_LONG_EDGE`** toward 1920. It was capped at 1280 to work
   around iOS canvas memory during a bug that has since been fixed, so this is
   worth re-testing on the target handset. At 1920 the same 0.40 framing gives
   3.1 px/mm and a 3 mm lesion spans 9 px.

`face_px` rides along with every capture, so the analysis side can filter on it
rather than discovering the problem after training.

### Two things that cost resolution invisibly

- **Noise reduction.** The phone ISP smooths fine luminance detail hardest in
  low light, which is exactly the signal a texture-based detector needs. Light
  the subject; see "Image quality" above.
- **JPEG chroma subsampling.** Browser JPEG encoders typically store colour at
  half resolution (4:2:0). If the acne pass keys on redness — which is the right
  choice, since chroma survives noise reduction better than texture — that
  halves the effective resolution of the signal it depends on. If it matters,
  encode the uploaded copy as PNG and accept the larger files; this is a
  research dataset, not a consumer app.

The definitive test is empirical and takes a minute: capture one image, open it
at 100%, and see whether *you* can pick out the lesions. If a person cannot, a
model will not.
