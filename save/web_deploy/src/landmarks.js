/**
 * landmarks.js — MediaPipe Face Mesh landmark indices, copied verbatim from
 * capture_prototype.py.
 *
 * FaceLandmarker (web) returns the same 478-point topology that Python's
 * FaceMesh(refine_landmarks=True) produces, so every index below carries over
 * unchanged. Do not renumber anything here.
 */

// --- Key landmarks ---------------------------------------------------------
export const NOSE_TIP = 1;
export const LEFT_TEMPLE = 234;
export const RIGHT_TEMPLE = 454;
export const FOREHEAD_CENTER = 10;
export const CHIN = 152;

// Eyebrow tops — the CUT LINE that removes the forehead from the cheek regions.
export const LEFT_BROW_TOP = 105;
export const RIGHT_BROW_TOP = 334;
export const BROW_INNER_L = 107;
export const BROW_INNER_R = 336;
export const BROW_IDS = [LEFT_BROW_TOP, RIGHT_BROW_TOP, BROW_INNER_L, BROW_INNER_R];

// --- Outer silhouette ------------------------------------------------------
// The mesh oval clips the upper forehead. That is fine: cheeks are cut at the
// brow anyway, and the front forehead is rebuilt from the skin mask.
export const FACE_OVAL = [
  10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
  397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
  172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109,
];

// --- FRONT centre strip (zones 1, 3, 4, 5) ---------------------------------
// Covers from just above the brows down through nose / mouth / chin. The
// forehead above the brow line is added separately from the skin mask.
export const FRONT_RIGHT_EDGE = [
  FOREHEAD_CENTER, 337, 336,       // right of forehead centre (zone 1)
  285, 417, 351, 419,              // beside nose bridge, top -> down (zone 3)
  399, 420, 360, 344,              // beside nostril / ala (still paranasal)
  438, 327,                        // naso-labial -> beside right nostril base
  326, 423, 426,                   // into the mouth column (zone 4)
  436, 434, 432,                   // right edge of mouth column
  424, 418, 421, 200,              // right side of chin -> chin centre (zone 5)
];
export const FRONT_LEFT_EDGE = [
  200, 201, 194, 204,              // chin centre -> left side of chin (zone 5)
  216, 212, 206,                   // left edge of mouth column (zone 4)
  203, 98,                         // mouth column -> beside left nostril base
  97, 64,                          // naso-labial (mirror of 327 / 438)
  48, 49, 131,                     // beside nostril / ala, bottom -> up
  198, 174, 196, 122, 193,         // beside nose bridge, down -> up (zone 3)
  55, 107, 108, FOREHEAD_CENTER,   // left of forehead centre (zone 1)
];
export const FRONT_STRIP = [...FRONT_RIGHT_EDGE, ...FRONT_LEFT_EDGE];

// --- Exclusion rings (eyes + lips are punched out of every region) ---------
export const LEFT_EYE_RING = [
  249, 390, 373, 374, 380, 381, 382, 362,
  398, 384, 385, 386, 387, 388, 466, 263,
];
export const RIGHT_EYE_RING = [
  7, 33, 246, 161, 160, 159, 158, 157,
  173, 133, 155, 154, 153, 145, 144, 163,
];
export const LIPS_RING = [
  0, 37, 39, 40, 185, 61, 146, 91, 181, 84,
  17, 314, 405, 321, 375, 291, 409, 270, 269, 267,
];
export const HOLE_RINGS = [LEFT_EYE_RING, RIGHT_EYE_RING, LIPS_RING];
