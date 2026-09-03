"""
capture_prototype.py

Capture webcam frames and auto-sort them into 3 folders based on head yaw
(left side / right side / front), using MediaPipe Face Mesh + OpenCV, with an
Image Segmenter (Selfie Multiclass, "face-skin" class) used to recover the
FULL forehead on the front shot.

This version also DISPLAYS and SAVES the number of face-skin pixels inside each
captured region (no ratio is computed yet -- acne is not detected here). The
count is simply the number of white pixels in the region mask (eyes and mouth
already punched out), so it represents skin only.

Region layout:
    - FRONT  (GROUP_3) -> center strip (zones 1,3,4,5) PLUS the entire forehead,
                          extended up to the real hairline using the skin mask
                          (Face Mesh alone clips the upper forehead).
    - LEFT/RIGHT cheek  -> cheek + jaw only, cut off at the eyebrow line so they
                           contain NO forehead at all.

Before saving, the frame must pass three quality gates:
    1. Distance   -- face not too far / not too close.
    2. Lighting   -- scene not too dark / not blown out.
    3. Pose       -- head turned to the required side / facing forward.

Install dependencies first:
    pip install opencv-python mediapipe numpy

The Selfie Multiclass model is downloaded automatically on first run and cached
next to this script as 'selfie_multiclass_256x256.tflite'.

Run:
    python capture_prototype.py

Press 'q' in the video window to quit.
"""

import os
import time
import urllib.request
from collections import deque

import cv2
import mediapipe as mp
import numpy as np

# MediaPipe Tasks API (for the Image Segmenter)
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
GROUP_1_DIR = "Group_1_Right_Cheek"   # left head turn -> right cheek exposed
GROUP_2_DIR = "Group_2_Left_Cheek"    # right head turn -> left cheek exposed
GROUP_3_DIR = "Group_3_Front"         # looking straight at camera

# --- Pose thresholds (symmetry ratio = dist_left / dist_right) -------------
RATIO_LEFT_CHEEK_THRESHOLD = 1.5    # ratio > this  -> Group 2 (left cheek)
RATIO_RIGHT_CHEEK_THRESHOLD = 0.6   # ratio < this  -> Group 1 (right cheek)
FRONT_SYMMETRY_TOLERANCE = 0.15     # |ratio - 1.0| must be <= this for front

# --- Distance thresholds ---------------------------------------------------
FACE_WIDTH_MIN_FRAC = 0.22   # smaller than this -> "too far"
FACE_WIDTH_MAX_FRAC = 0.65   # larger  than this -> "too close"

# --- Lighting thresholds ---------------------------------------------------
BRIGHTNESS_MIN = 70          # below -> "too dark"
BRIGHTNESS_MAX = 205         # above -> "too bright"

COOLDOWN_SECONDS = 3.0
CAPTURE_TEXT_DURATION = 1.0

# --- Selfie Multiclass segmenter -------------------------------------------
# The Selfie Multiclass 256x256 model labels each pixel with one of:
#   0 background, 1 hair, 2 body-skin, 3 face-skin, 4 clothes, 5 others.
SEG_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/image_segmenter/"
    "selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite"
)
SEG_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "selfie_multiclass_256x256.tflite"
)
FACE_SKIN_CLASS = 3          # category index for "face-skin"
# When True, the front region's forehead is extended up to the hairline using
# the skin mask. If the segmenter fails to load, we fall back to mesh-only.
USE_SKIN_FOREHEAD = True

# Optional: overlay the raw skin mask (semi-transparent blue) for tuning.
DEBUG_SHOW_SKIN_MASK = False

# --- Key landmark indices --------------------------------------------------
NOSE_TIP = 1
LEFT_TEMPLE = 234
RIGHT_TEMPLE = 454
FOREHEAD_CENTER = 10
CHIN = 152
MOUTH_LEFT = 61
MOUTH_RIGHT = 291

# Eyebrow tops -- used as the CUT LINE that removes the forehead from cheeks.
LEFT_BROW_TOP = 105
RIGHT_BROW_TOP = 334
BROW_INNER_L = 107
BROW_INNER_R = 336

# ---------------------------------------------------------------------------
# SUBUNIT POLYGONS
# ---------------------------------------------------------------------------
# FACE OVAL (outer silhouette) -- the tiling area for the LOWER face. Note the
# mesh oval clips the upper forehead; that's fine here because the cheeks are
# cut off at the brow anyway, and the front forehead is rebuilt from the skin
# mask.
FACE_OVAL = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
    397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
    172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109,
]

# FRONT strip (zones 1,3,4,5) center band. Same smooth paranasal edges as
# before. The forehead portion of "zone 1" is added separately from the skin
# mask (see _front_forehead_mask), so this strip only needs to cover from just
# above the brows down through nose / mouth / chin.
FRONT_RIGHT_EDGE = [
    FOREHEAD_CENTER, 337, 336,       # right of forehead center (zone 1)
    285, 417, 351, 419,              # beside nose bridge, top->down (zone 3)
    399, 420, 360, 344,              # beside nostril / ala (still paranasal)
    438, 327,                        # naso-labial -> beside right nostril base
    326, 423, 426,                   # into the mouth column (zone 4)
    436, 434, 432,                   # right edge of mouth column
    424, 418, 421, 200,              # right side of chin -> chin center (zone 5)
]
FRONT_LEFT_EDGE = [
    200, 201, 194, 204,              # chin center -> left side of chin (zone 5)
    216, 212, 206,                   # left edge of mouth column (zone 4)
    203, 98,                         # mouth column -> beside left nostril base
    97, 64,                          # naso-labial (mirror of 327/438-ish)
    48, 49, 131,                     # beside nostril / ala, bottom->up
    198, 174, 196, 122, 193,         # beside nose bridge, down->up (zone 3)
    55, 107, 108, FOREHEAD_CENTER,   # left of forehead center (zone 1)
]

# EYE / MOUTH exclusion rings (ordered, from MediaPipe FaceMesh).
LEFT_EYE_RING = [
    249, 390, 373, 374, 380, 381, 382, 362,
    398, 384, 385, 386, 387, 388, 466, 263,
]
RIGHT_EYE_RING = [
    7, 33, 246, 161, 160, 159, 158, 157,
    173, 133, 155, 154, 153, 145, 144, 163,
]
LIPS_RING = [
    0, 37, 39, 40, 185, 61, 146, 91, 181, 84,
    17, 314, 405, 321, 375, 291, 409, 270, 269, 267,
]

HOLE_DILATE_PX = 6


# ---------------------------------------------------------------------------
# SEGMENTER SETUP
# ---------------------------------------------------------------------------
def ensure_seg_model():
    """Download the Selfie Multiclass model once and cache it locally."""
    if os.path.exists(SEG_MODEL_PATH):
        return True
    try:
        print("[INFO] Downloading Selfie Multiclass model (first run)...")
        urllib.request.urlretrieve(SEG_MODEL_URL, SEG_MODEL_PATH)
        print(f"[INFO] Saved model to {SEG_MODEL_PATH}")
        return True
    except Exception as e:  # noqa
        print(f"[WARN] Could not download segmenter model: {e}")
        print("[WARN] Falling back to mesh-only forehead (no skin extension).")
        return False


def make_segmenter():
    """
    Build an ImageSegmenter in VIDEO mode returning a category mask. Returns
    None if the model isn't available (we then fall back to mesh-only).
    """
    if not (USE_SKIN_FOREHEAD and ensure_seg_model()):
        return None
    try:
        base = mp_python.BaseOptions(model_asset_path=SEG_MODEL_PATH)
        opts = mp_vision.ImageSegmenterOptions(
            base_options=base,
            running_mode=mp_vision.RunningMode.VIDEO,
            output_category_mask=True,
        )
        return mp_vision.ImageSegmenter.create_from_options(opts)
    except Exception as e:  # noqa
        print(f"[WARN] Could not create segmenter: {e}. Mesh-only fallback.")
        return None


def get_face_skin_mask(segmenter, frame_rgb, w, h, timestamp_ms):
    """
    Run the segmenter and return a binary uint8 mask (255 = face-skin) at the
    frame's resolution. Returns None on any failure so callers can fall back.
    """
    if segmenter is None:
        return None
    try:
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        result = segmenter.segment_for_video(mp_image, timestamp_ms)
        cat = result.category_mask.numpy_view()  # HxW uint8 of class indices
        skin = np.where(cat == FACE_SKIN_CLASS, 255, 0).astype(np.uint8)
        if skin.shape[:2] != (h, w):
            skin = cv2.resize(skin, (w, h), interpolation=cv2.INTER_NEAREST)
        # smooth the jittery mask edge a touch
        skin = cv2.morphologyEx(
            skin, cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        )
        return skin
    except Exception as e:  # noqa
        print(f"[WARN] Segmentation failed this frame: {e}")
        return None


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------
def create_output_dirs():
    for d in (GROUP_1_DIR, GROUP_2_DIR, GROUP_3_DIR):
        os.makedirs(d, exist_ok=True)
        print(f"[INFO] Folder ready: {d}")


def classify_ratio(ratio):
    if ratio > RATIO_LEFT_CHEEK_THRESHOLD:
        return "GROUP_2"
    elif ratio < RATIO_RIGHT_CHEEK_THRESHOLD:
        return "GROUP_1"
    else:
        return "GROUP_3"


def get_folder(group):
    return {
        "GROUP_1": GROUP_1_DIR,
        "GROUP_2": GROUP_2_DIR,
        "GROUP_3": GROUP_3_DIR,
    }[group]


def get_face_bounding_box(landmarks, w, h, padding=0.25):
    xs = [lm.x * w for lm in landmarks]
    ys = [lm.y * h for lm in landmarks]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    box_w = x_max - x_min
    box_h = y_max - y_min
    pad_w = box_w * padding
    pad_h = box_h * padding
    x_min = max(0, int(x_min - pad_w))
    y_min = max(0, int(y_min - pad_h))
    x_max = min(w, int(x_max + pad_w))
    y_max = min(h, int(y_max + pad_h))
    return x_min, y_min, x_max, y_max


def check_distance(bbox, w, h):
    x1, y1, x2, y2 = bbox
    face_w_frac = (x2 - x1) / float(w)
    if face_w_frac < FACE_WIDTH_MIN_FRAC:
        return "TOO_FAR", "MOVE CLOSER"
    if face_w_frac > FACE_WIDTH_MAX_FRAC:
        return "TOO_CLOSE", "MOVE BACK"
    return "OK", "Distance OK"


def check_lighting(frame, bbox):
    x1, y1, x2, y2 = bbox
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return "OK", "Light ?", 0.0
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    brightness = float(np.mean(gray))
    if brightness < BRIGHTNESS_MIN:
        return "TOO_DARK", "TOO DARK - add light", brightness
    if brightness > BRIGHTNESS_MAX:
        return "TOO_BRIGHT", "TOO BRIGHT - reduce light", brightness
    return "OK", "Light OK", brightness


def check_pose(group, ratio):
    if group == "GROUP_2":
        if ratio > RATIO_LEFT_CHEEK_THRESHOLD:
            return True, "Hold still - Left cheek"
        return False, "TURN HEAD to show LEFT cheek"
    if group == "GROUP_1":
        if ratio < RATIO_RIGHT_CHEEK_THRESHOLD:
            return True, "Hold still - Right cheek"
        return False, "TURN HEAD to show RIGHT cheek"
    if abs(ratio - 1.0) <= FRONT_SYMMETRY_TOLERANCE:
        return True, "Hold still - Front"
    if ratio > 1.0:
        return False, "FACE FORWARD (turn slightly right)"
    return False, "FACE FORWARD (turn slightly left)"


def _pts(landmarks, w, h, idx_list):
    return np.array(
        [[landmarks[i].x * w, landmarks[i].y * h] for i in idx_list],
        dtype=np.int32,
    )


def _brow_y(landmarks, h):
    """The y (pixels) of the eyebrow line -- the cut used to strip forehead."""
    ys = [landmarks[i].y * h for i in
          (LEFT_BROW_TOP, RIGHT_BROW_TOP, BROW_INNER_L, BROW_INNER_R)]
    return int(min(ys))  # highest brow point (smallest y) = cut line


def _face_oval_mask(landmarks, w, h):
    m = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(m, [_pts(landmarks, w, h, FACE_OVAL)], 255)
    return m


def _front_strip_mask(landmarks, w, h):
    """Filled mask of the FRONT center strip (nose/mouth/chin + brow-level top)."""
    m = np.zeros((h, w), dtype=np.uint8)
    poly = _pts(landmarks, w, h, FRONT_RIGHT_EDGE + FRONT_LEFT_EDGE)
    cv2.fillPoly(m, [poly], 255)
    return cv2.bitwise_and(m, _face_oval_mask(landmarks, w, h))


def _front_forehead_mask(landmarks, w, h, skin_mask):
    """
    Build the FOREHEAD fill for the FRONT region from the skin mask:
      - keep skin pixels ABOVE the eyebrow line (up to the hairline),
      - between the LEFT and RIGHT temple x (so ears/hair to the sides drop out),
      - largest connected blob (so a raised hand/arm can't leak in).
    Returns an empty mask if no skin mask is available.
    """
    out = np.zeros((h, w), dtype=np.uint8)
    if skin_mask is None:
        return out

    brow_y = _brow_y(landmarks, h)
    lx = int(landmarks[LEFT_TEMPLE].x * w)
    rx = int(landmarks[RIGHT_TEMPLE].x * w)
    x_lo, x_hi = min(lx, rx), max(lx, rx)

    band = np.zeros((h, w), dtype=np.uint8)
    band[0:max(1, brow_y), x_lo:x_hi] = 255  # above brows, between temples

    fh = cv2.bitwise_and(skin_mask, band)

    # keep only the largest blob (the forehead), drop stray skin patches
    n, lab, stats, _ = cv2.connectedComponentsWithStats(fh, connectivity=8)
    if n > 1:
        biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        fh = np.where(lab == biggest, 255, 0).astype(np.uint8)
    return fh


def _holes_mask(landmarks, w, h):
    holes = np.zeros((h, w), dtype=np.uint8)
    for ring in (LEFT_EYE_RING, RIGHT_EYE_RING, LIPS_RING):
        cv2.fillPoly(holes, [_pts(landmarks, w, h, ring)], 255)
    if HOLE_DILATE_PX > 0:
        k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (HOLE_DILATE_PX * 2 + 1, HOLE_DILATE_PX * 2 + 1)
        )
        holes = cv2.dilate(holes, k)
    return holes


def build_region_mask(landmarks, w, h, group, skin_mask=None):
    """
    Build a single-channel uint8 mask (255 = inside region) for `group`.

      - FRONT   = center strip UNION the skin-mask forehead (up to hairline).
      - RIGHT   = face oval, right of nose-tip x, MINUS front strip,
                  and CUT OFF above the eyebrow line (no forehead).
      - LEFT    = mirror of RIGHT.
    Eyes + mouth are punched out of all regions.
    """
    face = _face_oval_mask(landmarks, w, h)
    front = _front_strip_mask(landmarks, w, h)
    brow_y = _brow_y(landmarks, h)

    if group == "GROUP_3":
        forehead = _front_forehead_mask(landmarks, w, h, skin_mask)
        mask = cv2.bitwise_or(front, forehead)
    else:
        # cheek = face half, minus the front strip
        nose_x = int(landmarks[NOSE_TIP].x * w)
        half = np.zeros((h, w), dtype=np.uint8)
        if group == "GROUP_1":      # image-right cheek
            half[:, nose_x:] = 255
        else:                        # GROUP_2 image-left cheek
            half[:, :nose_x] = 255
        cheek = cv2.bitwise_and(face, half)
        cheek = cv2.bitwise_and(cheek, cv2.bitwise_not(front))
        # strip the forehead: zero everything above the eyebrow line
        cheek[0:max(0, brow_y), :] = 0
        mask = cheek

    # punch out eyes + mouth
    holes = _holes_mask(landmarks, w, h)
    mask = cv2.bitwise_and(mask, cv2.bitwise_not(holes))

    # keep only the largest connected blob
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n > 2:
        biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        mask = np.where(lab == biggest, 255, 0).astype(np.uint8)
    return mask


def count_skin_pixels(landmarks, w, h, group, skin_mask=None):
    """
    Return the number of face-skin pixels inside the region for `group`.
    This is simply the count of white (255) pixels in the region mask --
    eyes and mouth are already punched out, so it's skin only.

    NOTE: the count scales with distance to the camera (closer face = more
    pixels). It is NOT comparable across people or sessions on its own; it is
    meant here only to confirm we can extract per-region skin pixels. A ratio
    (acne_pixels / skin_pixels) is what cancels out scale later.
    """
    mask = build_region_mask(landmarks, w, h, group, skin_mask)
    return int(cv2.countNonZero(mask))


def get_region_contours(landmarks, w, h, group, skin_mask=None):
    mask = build_region_mask(landmarks, w, h, group, skin_mask)
    contours, _ = cv2.findContours(
        mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
    )
    return contours


def draw_region_overlay(frame, group, contours):
    annotated = frame.copy()
    color = (0, 255, 0) if group == "GROUP_3" else (0, 0, 255)
    cv2.drawContours(annotated, contours, -1, color, 3)
    return annotated


def is_peak(prev2, prev1, curr, group):
    if group == "GROUP_2":
        return prev1 > prev2 and prev1 >= curr
    elif group == "GROUP_1":
        return prev1 < prev2 and prev1 <= curr
    else:
        return abs(prev1 - 1.0) < abs(prev2 - 1.0) and abs(prev1 - 1.0) <= abs(curr - 1.0)


def put_status_line(frame, text, y, color):
    cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    create_output_dirs()

    segmenter = make_segmenter()
    if segmenter is not None:
        print("[INFO] Skin-mask forehead recovery ENABLED (front shot).")
    else:
        print("[INFO] Skin-mask disabled -- using mesh-only forehead.")

    mp_face_mesh = mp.solutions.face_mesh

    backend = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY
    cap = cv2.VideoCapture(0, backend)
    if not cap.isOpened():
        print("[ERROR] Could not open webcam at index 0. Trying index 1...")
        cap = cv2.VideoCapture(1, backend)
        if not cap.isOpened():
            print("[FATAL] No webcam could be opened. Please check that:")
            print("   - A webcam is physically connected")
            print("   - No other application (Zoom, Teams, etc.) is using it")
            print("   - This terminal / IDE has camera permission")
            return

    history = deque(maxlen=3)
    last_saved_time = 0.0
    capture_message_until = 0.0
    saved_counts = {"GROUP_1": 0, "GROUP_2": 0, "GROUP_3": 0}
    # remember the last saved skin-pixel count per group (for the end summary)
    saved_skin_px = {"GROUP_1": None, "GROUP_2": None, "GROUP_3": None}

    consecutive_failures = 0
    MAX_CONSECUTIVE_FAILURES = 60
    frame_idx = 0

    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as face_mesh:

        while True:
            ret, frame = cap.read()
            if not ret:
                consecutive_failures += 1
                print(f"[WARNING] Failed to grab frame ({consecutive_failures}/{MAX_CONSECUTIVE_FAILURES}). Retrying...")
                if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                    print("[FATAL] Webcam opened but never delivered a frame.")
                    break
                time.sleep(0.05)
                continue
            consecutive_failures = 0

            frame = cv2.flip(frame, 1)  # mirror view
            h, w, _ = frame.shape
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)

            display_frame = frame.copy()
            now = time.time()
            frame_idx += 1

            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark

                # --- pose ratio ---
                nose_x = landmarks[NOSE_TIP].x * w
                left_x = landmarks[LEFT_TEMPLE].x * w
                right_x = landmarks[RIGHT_TEMPLE].x * w
                dist_left = abs(nose_x - left_x)
                dist_right = abs(nose_x - right_x)
                if dist_right == 0:
                    dist_right = 1e-6
                ratio = dist_left / dist_right
                group = classify_ratio(ratio)

                # --- skin mask (only needed for the FRONT forehead) ---
                skin_mask = None
                if group == "GROUP_3" and segmenter is not None:
                    ts_ms = int(frame_idx * (1000.0 / 30.0))
                    skin_mask = get_face_skin_mask(segmenter, rgb, w, h, ts_ms)

                # --- optional debug overlay of the raw skin mask ---
                if DEBUG_SHOW_SKIN_MASK and skin_mask is not None:
                    blue = np.zeros_like(display_frame)
                    blue[skin_mask > 0] = (255, 0, 0)
                    display_frame = cv2.addWeighted(display_frame, 1.0, blue, 0.35, 0)

                # --- bounding box + region ---
                bbox = get_face_bounding_box(landmarks, w, h)
                bx1, by1, bx2, by2 = bbox
                cv2.rectangle(display_frame, (bx1, by1), (bx2, by2), (255, 200, 0), 1)

                region_contours = get_region_contours(landmarks, w, h, group, skin_mask)
                region_color = (0, 255, 0) if group == "GROUP_3" else (0, 0, 255)
                cv2.drawContours(display_frame, region_contours, -1, region_color, 2)

                # --- skin pixel count for this region (live) ---
                skin_px = count_skin_pixels(landmarks, w, h, group, skin_mask)

                # --- quality gates ---
                dist_status, dist_msg = check_distance(bbox, w, h)
                light_status, light_msg, brightness = check_lighting(frame, bbox)
                pose_ok, pose_msg = check_pose(group, ratio)

                distance_ok = dist_status == "OK"
                lighting_ok = light_status == "OK"

                cv2.putText(
                    display_frame, f"Ratio: {ratio:.2f} ({group})",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2,
                )

                # --- show skin pixel count on screen ---
                cv2.putText(
                    display_frame, f"Skin px: {skin_px:,}",
                    (10, h - 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2,
                )

                ok_color = (0, 200, 0)
                
                
                warn_color = (0, 165, 255)
                put_status_line(display_frame, dist_msg, 60,
                                ok_color if distance_ok else warn_color)
                put_status_line(display_frame, light_msg, 85,
                                ok_color if lighting_ok else warn_color)
                put_status_line(display_frame, pose_msg, 110,
                                ok_color if pose_ok else warn_color)

                in_cooldown = (now - last_saved_time) < COOLDOWN_SECONDS
                if in_cooldown:
                    remaining = COOLDOWN_SECONDS - (now - last_saved_time)
                    put_status_line(display_frame, f"Cooldown: {remaining:.1f}s",
                                    135, warn_color)

                all_gates_ok = distance_ok and lighting_ok and pose_ok and not in_cooldown

                if all_gates_ok:
                    history.append((ratio, frame.copy(), group, region_contours))

                    if len(history) == 3:
                        (r2, f2, g2, reg2), (r1, f1, g1, reg1), (r0, f0, g0, reg0) = (
                            history[0], history[1], history[2]
                        )
                        if g2 == g1 == g0 and is_peak(r2, r1, r0, g1):
                            folder = get_folder(g1)

                            # skin pixel count for the saved frame
                            saved_px = count_skin_pixels(
                                landmarks, w, h, g1, skin_mask
                            )

                            filename = os.path.join(
                                folder,
                                f"{folder}_{int(time.time())}_px{saved_px}.jpg",
                            )
                            annotated = draw_region_overlay(f1, g1, reg1)
                            cv2.imwrite(filename, annotated)

                            saved_counts[g1] += 1
                            saved_skin_px[g1] = saved_px
                            last_saved_time = now
                            capture_message_until = now + CAPTURE_TEXT_DURATION
                            print(f"[SAVED] {filename}  (ratio={r1:.2f}, "
                                  f"brightness={brightness:.0f}, "
                                  f"skin_px={saved_px})")
                            history.clear()
                else:
                    history.clear()
            else:
                cv2.putText(
                    display_frame, "No face detected", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2,
                )
                history.clear()

            if now < capture_message_until:
                cv2.putText(
                    display_frame, "IMAGE CAPTURED!",
                    (max(10, w // 2 - 180), h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3,
                )

            y0 = h - 90
            labels = [
                ("GROUP_1", "Right Cheek"),
                ("GROUP_2", "Left Cheek"),
                ("GROUP_3", "Front"),
            ]
            for i, (g, label) in enumerate(labels):
                px = saved_skin_px[g]
                px_txt = f" ({px:,} px)" if px is not None else ""
                cv2.putText(
                    display_frame, f"{label}: {saved_counts[g]}{px_txt}",
                    (10, y0 + i * 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                )

            cv2.imshow("Capture Prototype - Press 'q' to quit", display_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()
    if segmenter is not None:
        segmenter.close()

    print("[INFO] Session ended. Totals:", saved_counts)
    print("[INFO] Last saved skin-pixel count per region:")
    for g, label in (("GROUP_1", "Right Cheek"),
                     ("GROUP_2", "Left Cheek"),
                     ("GROUP_3", "Front")):
        px = saved_skin_px[g]
        px_txt = f"{px:,} px" if px is not None else "(none captured)"
        print(f"    {label:12s}: {px_txt}")


if __name__ == "__main__":
    main()