/**
 * vision.js — loads the two MediaPipe models and hands back a face-skin mask.
 *
 * These are the same two models the Python used: face_landmarker (which
 * produces the identical 478-point topology as FaceMesh with
 * refine_landmarks=True) and selfie_multiclass_256x256, downloaded from the
 * same Google bucket rather than cached next to a script.
 */

import {
  FilesetResolver,
  FaceLandmarker,
  ImageSegmenter,
} from 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.0.1/vision_bundle.mjs';

import {
  WASM_ROOT,
  FACE_LANDMARKER_MODEL,
  SEGMENTER_MODEL,
  FACE_SKIN_CLASS,
  SKIN_OPEN_PX,
  scaleRadius,
} from './config.js';
import { classMapToBinary, openSquare } from './mask.js';

/**
 * Try the GPU delegate first and fall back to CPU. Plenty of budget Android
 * handsets advertise WebGL but fail to build the GPU graph, and a hard failure
 * there would take the whole app down.
 */
async function createWithFallback(create, options, onFallback) {
  try {
    return await create({ ...options, baseOptions: { ...options.baseOptions, delegate: 'GPU' } });
  } catch (err) {
    if (onFallback) onFallback(err);
    return create({ ...options, baseOptions: { ...options.baseOptions, delegate: 'CPU' } });
  }
}

export async function loadVision({ onProgress } = {}) {
  const report = (m) => onProgress && onProgress(m);

  report('Loading runtime…');
  const fileset = await FilesetResolver.forVisionTasks(WASM_ROOT);

  report('Loading face landmarker…');
  const landmarker = await createWithFallback(
    (o) => FaceLandmarker.createFromOptions(fileset, o),
    {
      baseOptions: { modelAssetPath: FACE_LANDMARKER_MODEL },
      runningMode: 'VIDEO',
      numFaces: 1,
      minFaceDetectionConfidence: 0.5,
      minFacePresenceConfidence: 0.5,
      minTrackingConfidence: 0.5,
      outputFaceBlendshapes: false,
      outputFacialTransformationMatrixes: false,
    },
    () => report('GPU unavailable, falling back to CPU…'),
  );

  report('Loading skin segmenter…');
  const segmenterOptions = {
    baseOptions: { modelAssetPath: SEGMENTER_MODEL },
    outputCategoryMask: true,
    outputConfidenceMasks: false,
  };

  // Two instances on purpose. The live preview wants VIDEO mode, which carries
  // temporal state and expects monotonically rising timestamps; the capture
  // path runs one shot at full resolution and must not disturb that state.
  const segmenterVideo = await createWithFallback(
    (o) => ImageSegmenter.createFromOptions(fileset, o),
    { ...segmenterOptions, runningMode: 'VIDEO' },
  );
  const segmenterImage = await createWithFallback(
    (o) => ImageSegmenter.createFromOptions(fileset, o),
    { ...segmenterOptions, runningMode: 'IMAGE' },
  );

  return { landmarker, segmenterVideo, segmenterImage };
}

/** Landmarks for one face, or null. Timestamps must rise monotonically. */
export function detectLandmarks(landmarker, source, timestampMs) {
  const res = landmarker.detectForVideo(source, timestampMs);
  const faces = res && res.faceLandmarks;
  return faces && faces.length ? faces[0] : null;
}

/**
 * Run the segmenter and write a binary face-skin mask into `out` at w x h.
 * Returns true on success; on failure the caller falls back to a mesh-only
 * forehead, exactly as the Python did.
 *
 * The no-callback overload copies the mask data out of WASM for us, but we
 * still own the result and have to close it or the heap grows every frame.
 */
export function writeSkinMask(segmenter, source, timestampMs, w, h, out) {
  if (!segmenter) return false;
  let result = null;
  try {
    result = timestampMs === null
      ? segmenter.segment(source)
      : segmenter.segmentForVideo(source, timestampMs);

    const cm = result.categoryMask;
    if (!cm) return false;

    classMapToBinary(cm.getAsUint8Array(), cm.width, cm.height, out, w, h, FACE_SKIN_CLASS);
    openSquare(out, w, h, scaleRadius(SKIN_OPEN_PX, w, h));  // smooth the jittery edge
    return true;
  } catch (err) {
    console.warn('[warn] segmentation failed this frame:', err);
    return false;
  } finally {
    if (result) result.close();
  }
}

export function closeVision(v) {
  if (!v) return;
  v.landmarker?.close();
  v.segmenterVideo?.close();
  v.segmenterImage?.close();
}
