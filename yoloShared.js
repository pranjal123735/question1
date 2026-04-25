import * as tf from '@tensorflow/tfjs';

/**
 * donjuanpond/tfjs-yolov8n: NHWC [1, 640, 640, 3] float32 / 127.5.
 * Native uses smaller camera tensor + resize; web can feed any size from video.
 */
export const MODEL_INPUT = 640;
export const CAMERA_TENSOR = 416;

export const MIN_CONFIDENCE = 0.26;
export const MIN_DISPLAY_SCORE = 0.4;
export const DETECT_INTERVAL_MS = 580;
export const MAX_BOXES = 12;
export const DISPLAY_MAX_BOXES = 5;
export const MIN_BOX_AREA_RATIO = 0.0018;
export const MAX_BOX_AREA_RATIO = 0.92;
export const SAME_CLASS_IOU_SUPPRESS = 0.45;

export const YOLO_MODEL_URLS = [
  'https://raw.githubusercontent.com/donjuanpond/tfjs-yolov8n/main/static/model/model.json',
];

export const BOX_PALETTE = [
  '#3B82F6',
  '#22C55E',
  '#F97316',
  '#EAB308',
  '#A855F7',
  '#EC4899',
  '#06B6D4',
];

export const COCO_LABELS = [
  'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
  'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
  'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
  'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
  'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle',
  'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
  'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
  'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
  'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book',
  'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush',
];

/**
 * Extra per-class confidence floors for labels that often appear as false positives
 * on indoor textures, low light, or partial human face/skin regions.
 */
const CLASS_MIN_SCORE = {
  bird: 0.62,
  cat: 0.62,
  dog: 0.62,
  potted_plant: 0.65,
  plant: 0.65,
  toothbrush: 0.72,
  vase: 0.62,
};

function normalizeClassKey(label) {
  return String(label || '').trim().toLowerCase().replace(/\s+/g, '_');
}

export function bboxIouXYWH(a, b) {
  const [ax, ay, aw, ah] = a;
  const [bx, by, bw, bh] = b;
  const x1 = Math.max(ax, bx);
  const y1 = Math.max(ay, by);
  const x2 = Math.min(ax + aw, bx + bw);
  const y2 = Math.min(ay + ah, by + bh);
  const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  const union = aw * ah + bw * bh - inter;
  return union > 0 ? inter / union : 0;
}

export function refinePredictions(predictions) {
  if (!predictions?.length) {
    return [];
  }

  const area = MODEL_INPUT * MODEL_INPUT;
  const cleaned = predictions
    .filter((p) => {
      const classKey = normalizeClassKey(p.class);
      const classFloor = CLASS_MIN_SCORE[classKey] ?? MIN_DISPLAY_SCORE;
      return p.score >= classFloor;
    })
    .map((p) => {
      const [x, y, w, h] = p.bbox;
      const x1 = Math.max(0, Math.min(MODEL_INPUT, x));
      const y1 = Math.max(0, Math.min(MODEL_INPUT, y));
      const x2 = Math.max(0, Math.min(MODEL_INPUT, x + w));
      const y2 = Math.max(0, Math.min(MODEL_INPUT, y + h));
      return {
        ...p,
        bbox: [x1, y1, Math.max(0, x2 - x1), Math.max(0, y2 - y1)],
      };
    })
    .filter((p) => {
      const [, , w, h] = p.bbox;
      if (!Number.isFinite(p.score) || !Number.isFinite(w) || !Number.isFinite(h)) {
        return false;
      }
      const ratio = (w * h) / area;
      return ratio >= MIN_BOX_AREA_RATIO && ratio <= MAX_BOX_AREA_RATIO;
    })
    .sort((a, b) => b.score - a.score);

  const deduped = [];
  for (const p of cleaned) {
    const dupe = deduped.some(
      (d) => d.class === p.class && bboxIouXYWH(d.bbox, p.bbox) >= SAME_CLASS_IOU_SUPPRESS
    );
    if (!dupe) {
      deduped.push(p);
    }
    if (deduped.length >= DISPLAY_MAX_BOXES) {
      break;
    }
  }
  return deduped;
}

export function mapPredictionsToOverlay(predictions, previewW, previewH) {
  if (!previewW || !previewH || !predictions?.length) {
    return [];
  }
  const sx = previewW / MODEL_INPUT;
  const sy = previewH / MODEL_INPUT;
  return [...predictions]
    .sort((a, b) => b.score - a.score)
    .map((p, i) => {
      const [x, y, w, h] = p.bbox;
      return {
        key: `${p.class}-${i}-${Math.round(p.score * 1000)}`,
        label: `${p.class} ${(p.score * 100).toFixed(0)}%`,
        left: x * sx,
        top: y * sy,
        width: Math.max(w * sx, 2),
        height: Math.max(h * sy, 2),
        color: BOX_PALETTE[i % BOX_PALETTE.length],
      };
    });
}

export function prepareSpatialForYolo(imageTensor, rotate90) {
  return tf.tidy(() => {
    let x = imageTensor.toFloat();
    if (rotate90) {
      // tf.image.rotateWithOffset expects rank-4 on web backend.
      x = tf.image.rotateWithOffset(x.expandDims(0), Math.PI / 2, 0, 0, 0).squeeze([0]);
    }
    const h = x.shape[0];
    const w = x.shape[1];
    if (h !== MODEL_INPUT || w !== MODEL_INPUT) {
      x = tf.image.resizeBilinear(x, [MODEL_INPUT, MODEL_INPUT]);
    }
    return x;
  });
}

export async function runDonjuanpondYolo(model, imageTensor, rotate90) {
  const spatial = prepareSpatialForYolo(imageTensor, rotate90);
  const yoloInput = tf.tidy(() => spatial.div(127.5).expandDims(0));
  spatial.dispose();

  const raw = model.execute({ x: yoloInput });
  yoloInput.dispose();

  const res = Array.isArray(raw) ? raw[0] : raw;
  if (Array.isArray(raw)) {
    raw.forEach((t, i) => {
      if (i !== 0) {
        t.dispose();
      }
    });
  }

  const transRes = res.transpose([0, 2, 1]);
  res.dispose();

  const boxes = tf.tidy(() => {
    const w = transRes.slice([0, 0, 2], [-1, -1, 1]);
    const h = transRes.slice([0, 0, 3], [-1, -1, 1]);
    const cx = transRes.slice([0, 0, 0], [-1, -1, 1]);
    const cy = transRes.slice([0, 0, 1], [-1, -1, 1]);
    const x1 = tf.sub(cx, tf.div(w, 2));
    const y1 = tf.sub(cy, tf.div(h, 2));
    return tf.concat([x1, y1, tf.add(x1, w), tf.add(y1, h)], 2).squeeze();
  });

  const rawScores = transRes.slice([0, 0, 4], [-1, -1, 80]).squeeze([0]);
  transRes.dispose();

  const probs = tf.sigmoid(rawScores);
  rawScores.dispose();
  const scores = probs.max(1);
  const classes = probs.argMax(1);

  const yxBoxes = tf.tidy(() => {
    const x1 = boxes.slice([0, 0], [-1, 1]);
    const y1 = boxes.slice([0, 1], [-1, 1]);
    const x2 = boxes.slice([0, 2], [-1, 1]);
    const y2 = boxes.slice([0, 3], [-1, 1]);
    return tf.concat([y1, x1, y2, x2], 1);
  });

  const nmsIdx = await tf.image.nonMaxSuppressionAsync(
    yxBoxes,
    scores,
    MAX_BOXES,
    0.45,
    MIN_CONFIDENCE
  );
  yxBoxes.dispose();
  probs.dispose();

  const selectedBoxes = boxes.gather(nmsIdx, 0);
  const selectedScores = scores.gather(nmsIdx, 0);
  const selectedClasses = classes.gather(nmsIdx, 0);
  nmsIdx.dispose();
  boxes.dispose();
  scores.dispose();
  classes.dispose();

  const bd = selectedBoxes.dataSync();
  const sd = selectedScores.dataSync();
  const cd = selectedClasses.dataSync();
  selectedBoxes.dispose();
  selectedScores.dispose();
  selectedClasses.dispose();

  const predictions = [];
  const n = sd.length;
  for (let i = 0; i < n; i += 1) {
    const x1 = bd[i * 4];
    const y1 = bd[i * 4 + 1];
    const x2 = bd[i * 4 + 2];
    const y2 = bd[i * 4 + 3];
    const cls = Math.round(cd[i]);
    predictions.push({
      class: COCO_LABELS[cls] ?? `class_${cls}`,
      score: sd[i],
      bbox: [x1, y1, x2 - x1, y2 - y1],
    });
  }
  return refinePredictions(predictions);
}

export function estimateDistanceMeters(bboxWidth, bboxHeight, frameWidth, frameHeight) {
  const objectAreaRatio = (bboxWidth * bboxHeight) / (frameWidth * frameHeight);

  if (objectAreaRatio > 0.45) return 0.5;
  if (objectAreaRatio > 0.25) return 1.0;
  if (objectAreaRatio > 0.12) return 1.8;
  if (objectAreaRatio > 0.06) return 2.8;
  if (objectAreaRatio > 0.03) return 4.0;
  return 6.0;
}
