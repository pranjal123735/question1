from __future__ import annotations

import os
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None

try:
    from PIL import Image
except Exception:  # pragma: no cover
    Image = None

try:
    from ultralytics import YOLO
except Exception:  # pragma: no cover
    YOLO = None


VEHICLE_LABELS = {"car", "truck", "bus", "motorcycle", "bicycle"}
MOVING_SPEED_THRESHOLD_KMH = 1.8

# YOLO inference (tune for fewer false positives; raise conf if still noisy)
# Default small model; override with CAR_VISION_YOLO_MODEL if needed.
YOLO_MODEL_NAME = os.environ.get("CAR_VISION_YOLO_MODEL", "yolov8s.pt")
# Higher = fewer false boxes (better label accuracy). Lower via env only if you need more recall on tiny/distant objects.
YOLO_CONF = float(os.environ.get("CAR_VISION_YOLO_CONF", "0.38"))
YOLO_IOU = float(os.environ.get("CAR_VISION_YOLO_IOU", "0.5"))
YOLO_MAX_DET = int(os.environ.get("CAR_VISION_YOLO_MAX_DET", "60"))
YOLO_IMGSZ = int(os.environ.get("CAR_VISION_YOLO_IMGSZ", "640"))
# Slower (~2× infer) but can slightly improve robustness; use when you care more about detection than FPS.
YOLO_TTA = os.environ.get("CAR_VISION_YOLO_TTA", "").lower() in ("1", "true", "yes")

# Set CAR_VISION_DISABLE_PERSON_FP_FILTER=1 only if you want maximum recall (more wrong "person" boxes possible).
DISABLE_PERSON_FP_FILTER = os.environ.get("CAR_VISION_DISABLE_PERSON_FP_FILTER", "").lower() in (
    "1",
    "true",
    "yes",
)

# Monocular height: when bbox is tall vs frame, visible extent is often << nominal object height (e.g. face vs 1.7m person).
DISTANCE_CLOSE_FRAC = float(os.environ.get("CAR_VISION_DISTANCE_CLOSE_FRAC", "0.14"))
DISTANCE_MIN_M = float(os.environ.get("CAR_VISION_DISTANCE_MIN_M", "0.08"))
DISTANCE_MAX_M = float(os.environ.get("CAR_VISION_DISTANCE_MAX_M", "120.0"))

# Temporal smoothing: stable class over last N frames (reduces phone↔person flicker)
CLASS_HISTORY_LEN = int(os.environ.get("CAR_VISION_CLASS_HISTORY", "7"))
# Minimum frames with same winning class before switching label (hysteresis)
CLASS_SWITCH_AFTER = int(os.environ.get("CAR_VISION_CLASS_SWITCH_FRAMES", "3"))

# Person false-positive suppression (handheld phone confused as person)
PERSON_MIN_CONF_SMALL = float(os.environ.get("CAR_VISION_PERSON_MIN_CONF_SMALL", "0.62"))
PERSON_MAX_AREA_FRAC_WEAK = float(os.environ.get("CAR_VISION_PERSON_MAX_AREA_FRAC", "0.0045"))
PERSON_TALL_NARROW_AR = float(os.environ.get("CAR_VISION_PERSON_TALL_AR", "3.2"))
PERSON_TALL_NARROW_MAX_CONF = float(os.environ.get("CAR_VISION_PERSON_TALL_MAX_CONF", "0.72"))

# IoU: if person overlaps cell phone and phone scores reasonably, drop person
PHONE_PERSON_IOU = float(os.environ.get("CAR_VISION_PHONE_PERSON_IOU", "0.25"))
PHONE_PERSON_CONF_RATIO = float(os.environ.get("CAR_VISION_PHONE_CONF_RATIO", "0.85"))

API_VERSION = "0.4.0"

app = FastAPI(title="Car Vision Backend", version="0.3.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class DetectionOut(BaseModel):
    track_id: int
    label: str
    confidence: float
    bbox_xyxy: List[float]
    distance_m: float
    speed_kmh: float
    is_moving: bool
    ttc_s: float
    risk_percent: float
    track_age_s: float = 0.0


class FrameDiagnostics(BaseModel):
    """Cheap frame stats so the client can warn when ranging/detection is less reliable."""

    brightness_01: float
    texture_variance: float
    low_light: bool
    glare_risk: bool
    low_contrast: bool
    quality_hint: str


class TripSnapshot(BaseModel):
    """Phase 4: per-session counters (in-memory until POST /trip/reset)."""

    frames: int
    danger_frames: int
    caution_frames: int
    safe_frames: int
    near_miss_count: int
    trip_elapsed_s: float


class TripEventOut(BaseModel):
    ts_s: float
    severity: str
    label: str
    track_id: int
    distance_m: float
    risk_percent: float
    ttc_s: Optional[float] = None


class TripStatsOut(BaseModel):
    trip_started_s: float
    frames: int
    danger_frames: int
    caution_frames: int
    safe_frames: int
    near_miss_count: int
    trip_elapsed_s: float
    events: List[TripEventOut]


class AnalyzeResponse(BaseModel):
    frame_time_s: float
    detections: List[DetectionOut]
    trip: TripSnapshot
    api_version: str = API_VERSION
    frame_diagnostics: Optional[FrameDiagnostics] = None


class CalibrationConfig(BaseModel):
    focal_like: float
    meters_per_px: float
    default_object_height_m: float
    object_heights_m: Dict[str, float]


class CalibrationPatch(BaseModel):
    focal_like: Optional[float] = None
    meters_per_px: Optional[float] = None
    default_object_height_m: Optional[float] = None


@dataclass
class TrackState:
    cx: float
    cy: float
    t_s: float
    speed_mps: float


MODEL = None
MODEL_LOAD_ERROR: Optional[str] = None
if YOLO is not None:
    try:
        MODEL = YOLO(YOLO_MODEL_NAME)
    except Exception as e:
        MODEL = None
        MODEL_LOAD_ERROR = str(e)
else:
    MODEL_LOAD_ERROR = "ultralytics import failed"

TRACKS: Dict[int, TrackState] = {}
TRACK_FIRST_SEEN_S: Dict[int, float] = {}
NEXT_ID = 1
# Per track: deque of (raw_model_label, confidence) from YOLO — output label is smoothed, never random
TRACK_CLASS_HIST: Dict[int, Deque[Tuple[str, float]]] = defaultdict(
    lambda: deque(maxlen=CLASS_HISTORY_LEN)
)
# Last emitted stable label per track (hysteresis)
TRACK_STABLE_LABEL: Dict[int, str] = {}
# Phase 4: trip / near-miss session (process-local; reset via POST /trip/reset)
NEAR_MISS_DEBOUNCE_S = float(os.environ.get("CAR_VISION_NEAR_MISS_DEBOUNCE_S", "4.0"))
TRIP_EVENTS_MAX = int(os.environ.get("CAR_VISION_TRIP_EVENTS_MAX", "80"))

TRIP_STARTED_S: float = time.time()
TRIP_FRAMES: int = 0
TRIP_DANGER_FRAMES: int = 0
TRIP_CAUTION_FRAMES: int = 0
TRIP_SAFE_FRAMES: int = 0
TRIP_NEAR_MISS_COUNT: int = 0
TRIP_EVENTS: Deque[dict] = deque(maxlen=TRIP_EVENTS_MAX)
_LAST_NEAR_MISS_LOG: Dict[str, float] = {}

CALIBRATION = CalibrationConfig(
    focal_like=float(os.environ.get("CAR_VISION_DEFAULT_FOCAL_LIKE", "900.0")),
    meters_per_px=0.05,
    default_object_height_m=1.5,
    object_heights_m={
        "person": 1.7,
        "bicycle": 1.1,
        "motorcycle": 1.2,
        "car": 1.5,
        "truck": 3.0,
        "bus": 3.2,
    },
)


def estimate_distance_m(
    label: str,
    bbox_xyxy: Tuple[float, float, float, float],
    frame_w: float,
    frame_h: float,
) -> float:
    """
    Pinhole-ish ranging from bbox vertical span. When the box is large vs frame height,
    the visible real-world extent is usually much smaller than nominal class height
    (e.g. close-up face vs full 1.7m person), so we shrink effective height to avoid
    overstating distance.
    """
    _x1, y1, _x2, y2 = bbox_xyxy
    h = max(y2 - y1, 1.0)
    fh = max(frame_h, 1.0)
    obj_h_m = CALIBRATION.object_heights_m.get(label, CALIBRATION.default_object_height_m)
    # 1.0 when object is small on screen; <1 when it fills many rows (close / partial subject).
    shrink = min(1.0, (DISTANCE_CLOSE_FRAC * fh) / h)
    effective_h_m = obj_h_m * shrink
    distance = (effective_h_m * CALIBRATION.focal_like) / h
    return float(max(DISTANCE_MIN_M, min(distance, DISTANCE_MAX_M)))


def ttc_and_risk(distance_m: float, speed_kmh: float) -> Tuple[float, float]:
    if speed_kmh <= 0.0:
        return 999.0, 0.0
    speed_mps = max(0.0, speed_kmh / 3.6)
    rel_speed = max(0.1, speed_mps)
    ttc = distance_m / rel_speed
    if ttc < 1.5:
        risk = 96.0
    elif ttc < 3.0:
        risk = 72.0
    elif ttc < 5.0:
        risk = 42.0
    else:
        risk = 15.0
    return float(ttc), float(risk)


def threat_score_value(det: DetectionOut) -> float:
    """Same weighting as the web client `threatScore()` for consistent ranking."""
    distance_m = float(det.distance_m)
    speed_kmh = float(det.speed_kmh)
    risk = float(det.risk_percent)
    moving = bool(det.is_moving)
    df = max(0.0, min(1.0, (45.0 - distance_m) / 45.0))
    sf = max(0.0, min(1.0, speed_kmh / 70.0)) if moving else 0.0
    rf = max(0.0, min(1.0, risk / 100.0))
    return df * 0.45 + sf * 0.25 + rf * 0.3


def band_for_detection(det: DetectionOut) -> str:
    risk = float(det.risk_percent)
    ttc = float(det.ttc_s)
    if risk >= 75.0 or ttc < 1.8:
        return "DANGER"
    if risk >= 40.0 or ttc < 3.5:
        return "CAUTION"
    return "SAFE"


def pick_top_detection(detections: List[DetectionOut]) -> Optional[DetectionOut]:
    if not detections:
        return None
    pool = [d for d in detections if d.label in VEHICLE_LABELS and d.is_moving]
    if not pool:
        pool = list(detections)
    return max(pool, key=threat_score_value)


def trip_snapshot(now_s: float) -> TripSnapshot:
    elapsed = max(0.0, now_s - TRIP_STARTED_S)
    return TripSnapshot(
        frames=TRIP_FRAMES,
        danger_frames=TRIP_DANGER_FRAMES,
        caution_frames=TRIP_CAUTION_FRAMES,
        safe_frames=TRIP_SAFE_FRAMES,
        near_miss_count=TRIP_NEAR_MISS_COUNT,
        trip_elapsed_s=round(elapsed, 1),
    )


def update_trip_stats(detections: List[DetectionOut], now_s: float) -> None:
    global TRIP_FRAMES, TRIP_DANGER_FRAMES, TRIP_CAUTION_FRAMES, TRIP_SAFE_FRAMES, TRIP_NEAR_MISS_COUNT

    TRIP_FRAMES += 1
    top = pick_top_detection(detections)
    if top is None:
        TRIP_SAFE_FRAMES += 1
        return
    band = band_for_detection(top)
    if band == "DANGER":
        TRIP_DANGER_FRAMES += 1
    elif band == "CAUTION":
        TRIP_CAUTION_FRAMES += 1
    else:
        TRIP_SAFE_FRAMES += 1

    if top.label in VEHICLE_LABELS and top.is_moving and band in ("CAUTION", "DANGER"):
        key = f"{top.track_id}:{band}"
        last_t = _LAST_NEAR_MISS_LOG.get(key, 0.0)
        if now_s - last_t >= NEAR_MISS_DEBOUNCE_S:
            _LAST_NEAR_MISS_LOG[key] = now_s
            TRIP_NEAR_MISS_COUNT += 1
            ttc_val = float(top.ttc_s)
            TRIP_EVENTS.appendleft(
                {
                    "ts_s": now_s,
                    "severity": band,
                    "label": top.label,
                    "track_id": int(top.track_id),
                    "distance_m": round(float(top.distance_m), 2),
                    "risk_percent": round(float(top.risk_percent), 1),
                    "ttc_s": None if ttc_val >= 900.0 else round(ttc_val, 2),
                }
            )


def reset_trip_state() -> None:
    global TRIP_STARTED_S, TRIP_FRAMES, TRIP_DANGER_FRAMES, TRIP_CAUTION_FRAMES, TRIP_SAFE_FRAMES
    global TRIP_NEAR_MISS_COUNT, _LAST_NEAR_MISS_LOG

    TRIP_STARTED_S = time.time()
    TRIP_FRAMES = 0
    TRIP_DANGER_FRAMES = 0
    TRIP_CAUTION_FRAMES = 0
    TRIP_SAFE_FRAMES = 0
    TRIP_NEAR_MISS_COUNT = 0
    TRIP_EVENTS.clear()
    _LAST_NEAR_MISS_LOG.clear()


def centroid(bbox: Tuple[float, float, float, float]) -> Tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def compute_frame_diagnostics(frame: "np.ndarray") -> FrameDiagnostics:
    """Lightweight scene cues from the RGB frame (no extra model)."""
    gray = np.mean(frame.astype(np.float32), axis=2)
    step = max(1, int(min(gray.shape[0], gray.shape[1]) / 90))
    small = gray[::step, ::step]
    brightness_01 = float(np.clip(small.mean() / 255.0, 0.0, 1.0))
    texture_variance = float(np.var(small))
    bright_frac = float((small >= 235.0).mean())
    low_light = brightness_01 < 0.11
    glare_risk = bright_frac > 0.085
    low_contrast = texture_variance < 12.0
    hints: List[str] = []
    if low_light:
        hints.append("low_light")
    if glare_risk:
        hints.append("glare")
    if low_contrast:
        hints.append("low_contrast")
    if len(hints) > 1:
        quality_hint = "mixed"
    elif hints:
        quality_hint = hints[0]
    else:
        quality_hint = "ok"
    return FrameDiagnostics(
        brightness_01=brightness_01,
        texture_variance=texture_variance,
        low_light=low_light,
        glare_risk=glare_risk,
        low_contrast=low_contrast,
        quality_hint=quality_hint,
    )


def bbox_iou_xyxy(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    aa = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    ba = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = aa + ba - inter
    return float(inter / union) if union > 0 else 0.0


def match_track(
    cx: float,
    cy: float,
    now_s: float,
    frame_w: float,
    frame_h: float,
) -> Optional[int]:
    gate_px = max(48.0, min(frame_w, frame_h) * 0.07)
    best_id = None
    best_d2 = 1e18
    for tid, st in TRACKS.items():
        if now_s - st.t_s > 1.5:
            continue
        dx, dy = cx - st.cx, cy - st.cy
        d2 = dx * dx + dy * dy
        if d2 < gate_px * gate_px and d2 < best_d2:
            best_id = tid
            best_d2 = d2
    return best_id


def update_track(tid: int, cx: float, cy: float, now_s: float) -> float:
    prev = TRACKS.get(tid)
    if not prev:
        TRACKS[tid] = TrackState(cx=cx, cy=cy, t_s=now_s, speed_mps=0.0)
        return 0.0
    dt = max(now_s - prev.t_s, 1e-3)
    px_dist = ((cx - prev.cx) ** 2 + (cy - prev.cy) ** 2) ** 0.5
    speed_mps = (px_dist * CALIBRATION.meters_per_px) / dt
    TRACKS[tid] = TrackState(cx=cx, cy=cy, t_s=now_s, speed_mps=speed_mps)
    return speed_mps


def smooth_label_from_history(tid: int, raw_label: str, raw_conf: float) -> Tuple[str, float]:
    """Weighted vote + hysteresis on YOLO class names only (reduces single-frame mislabels)."""
    hist = TRACK_CLASS_HIST[tid]
    hist.append((raw_label, raw_conf))
    scores: Dict[str, float] = defaultdict(float)
    counts: Dict[str, int] = defaultdict(int)
    for lbl, c in hist:
        scores[lbl] += c
        counts[lbl] += 1
    winner = max(scores.keys(), key=lambda k: scores[k])
    stable = TRACK_STABLE_LABEL.get(tid)
    if stable is None:
        out_label = winner
        TRACK_STABLE_LABEL[tid] = winner
    elif winner == stable:
        out_label = stable
    else:
        recent = list(hist)[-CLASS_SWITCH_AFTER:]
        consecutive_new = len(recent) >= CLASS_SWITCH_AFTER and all(l == winner for l, _ in recent)
        margin = scores[winner] > scores.get(stable, 0.0) * 1.12
        if consecutive_new or margin:
            out_label = winner
            TRACK_STABLE_LABEL[tid] = winner
        else:
            out_label = stable
    avg_conf = scores[out_label] / max(counts[out_label], 1)
    return out_label, float(min(0.99, max(raw_conf, avg_conf)))


def person_false_positive_filter(
    label: str,
    conf: float,
    bbox: Tuple[float, float, float, float],
    frame_w: float,
    frame_h: float,
) -> bool:
    """Return True to keep detection, False to drop weak person-like boxes."""
    if label != "person":
        return True
    x1, y1, x2, y2 = bbox
    w = max(x2 - x1, 1.0)
    h = max(y2 - y1, 1.0)
    area = w * h
    area_frac = area / max(frame_w * frame_h, 1.0)
    ar = h / w
    if area_frac < PERSON_MAX_AREA_FRAC_WEAK and conf < PERSON_MIN_CONF_SMALL:
        return False
    if ar >= PERSON_TALL_NARROW_AR and w < 0.035 * frame_w and conf < PERSON_TALL_NARROW_MAX_CONF:
        return False
    return True


def resolve_person_cell_phone_conflicts(
    rows: List[Tuple[str, float, Tuple[float, float, float, float]]],
) -> List[Tuple[str, float, Tuple[float, float, float, float]]]:
    """
    If a 'person' box strongly overlaps a 'cell phone' box and phone conf is competitive, remove person
    (common misread: phone in hand).
    """
    people: List[int] = []
    phones: List[int] = []
    for i, (lab, conf, bb) in enumerate(rows):
        if lab == "person":
            people.append(i)
        elif lab in ("cell phone", "cell_phone"):
            phones.append(i)
    drop: set[int] = set()
    for pi in people:
        plab, pconf, pbb = rows[pi]
        for hi in phones:
            _, hconf, hbb = rows[hi]
            iou = bbox_iou_xyxy(pbb, hbb)
            if iou >= PHONE_PERSON_IOU and hconf >= pconf * PHONE_PERSON_CONF_RATIO:
                drop.add(pi)
                break
    return [rows[i] for i in range(len(rows)) if i not in drop]


@app.get("/health")
def health() -> dict:
    return {
        "ok": True,
        "api_version": API_VERSION,
        "mode": "yolo" if MODEL is not None else "demo",
        "message": "Backend ready",
        "model": YOLO_MODEL_NAME if MODEL else None,
        "model_load_error": MODEL_LOAD_ERROR,
        "yolo_conf": YOLO_CONF,
        "max_det": YOLO_MAX_DET,
        "yolo_tta": YOLO_TTA,
        "disable_person_fp_filter": DISABLE_PERSON_FP_FILTER,
        "calibration": CALIBRATION.model_dump(),
    }


@app.get("/calibration", response_model=CalibrationConfig)
def get_calibration() -> CalibrationConfig:
    return CALIBRATION


@app.post("/calibration", response_model=CalibrationConfig)
def set_calibration(patch: CalibrationPatch) -> CalibrationConfig:
    global CALIBRATION
    data = CALIBRATION.model_dump()
    if patch.focal_like is not None:
        data["focal_like"] = float(max(200.0, min(2500.0, patch.focal_like)))
    if patch.meters_per_px is not None:
        data["meters_per_px"] = float(max(0.005, min(0.5, patch.meters_per_px)))
    if patch.default_object_height_m is not None:
        data["default_object_height_m"] = float(max(0.3, min(4.5, patch.default_object_height_m)))
    CALIBRATION = CalibrationConfig(**data)
    return CALIBRATION


@app.get("/demo-detections", response_model=AnalyzeResponse)
def demo_detections() -> AnalyzeResponse:
    """No random labels: empty list when only demo mode, or run real model if loaded."""
    now_s = time.time()
    return AnalyzeResponse(frame_time_s=now_s, detections=[], trip=trip_snapshot(now_s))


@app.get("/trip/stats", response_model=TripStatsOut)
def trip_stats() -> TripStatsOut:
    """Phase 4: session counters and recent near-miss / caution events (newest first)."""
    now_s = time.time()
    events = [TripEventOut(**e) for e in TRIP_EVENTS]
    return TripStatsOut(
        trip_started_s=TRIP_STARTED_S,
        frames=TRIP_FRAMES,
        danger_frames=TRIP_DANGER_FRAMES,
        caution_frames=TRIP_CAUTION_FRAMES,
        safe_frames=TRIP_SAFE_FRAMES,
        near_miss_count=TRIP_NEAR_MISS_COUNT,
        trip_elapsed_s=round(max(0.0, now_s - TRIP_STARTED_S), 1),
        events=events,
    )


@app.post("/trip/reset")
def trip_reset() -> dict:
    """Clear Phase 4 trip stats and event log for a new session."""
    reset_trip_state()
    return {"ok": True}


@app.post("/analyze-image", response_model=AnalyzeResponse)
async def analyze_image(file: UploadFile = File(...)) -> AnalyzeResponse:
    now_s = time.time()
    if np is None or Image is None:
        return AnalyzeResponse(frame_time_s=now_s, detections=[], trip=trip_snapshot(now_s))

    raw = await file.read()
    image = Image.open(io_bytes(raw)).convert("RGB")
    frame = np.array(image)
    fh, fw = float(frame.shape[0]), float(frame.shape[1])
    diagnostics = compute_frame_diagnostics(frame)

    if MODEL is None:
        return AnalyzeResponse(
            frame_time_s=now_s,
            detections=[],
            trip=trip_snapshot(now_s),
            frame_diagnostics=diagnostics,
        )

    results = MODEL.predict(
        frame,
        verbose=False,
        conf=YOLO_CONF,
        iou=YOLO_IOU,
        max_det=YOLO_MAX_DET,
        imgsz=YOLO_IMGSZ,
        augment=YOLO_TTA,
    )

    raw_rows: List[Tuple[str, float, Tuple[float, float, float, float]]] = []
    for r in results:
        if r.boxes is None:
            continue
        for b in r.boxes:
            cls_idx = int(b.cls[0].item())
            label = MODEL.names.get(cls_idx, str(cls_idx))
            conf = float(b.conf[0].item())
            x1, y1, x2, y2 = [float(v) for v in b.xyxy[0].tolist()]
            bbox = (x1, y1, x2, y2)
            if not DISABLE_PERSON_FP_FILTER and not person_false_positive_filter(label, conf, bbox, fw, fh):
                continue
            raw_rows.append((label, conf, bbox))

    raw_rows = resolve_person_cell_phone_conflicts(raw_rows)

    out: List[DetectionOut] = []
    global NEXT_ID

    for label, conf, bbox in raw_rows:
        cx, cy = centroid(bbox)
        tid = match_track(cx, cy, now_s, fw, fh)
        if tid is None:
            tid = NEXT_ID
            NEXT_ID += 1
        if tid not in TRACK_FIRST_SEEN_S:
            TRACK_FIRST_SEEN_S[tid] = now_s
        track_age_s = max(0.0, now_s - TRACK_FIRST_SEEN_S[tid])
        speed_mps = update_track(tid, cx, cy, now_s)
        speed_kmh_raw = max(0.0, speed_mps * 3.6)
        is_moving = speed_kmh_raw >= MOVING_SPEED_THRESHOLD_KMH
        speed_kmh = speed_kmh_raw if is_moving else 0.0
        smooth_label, smooth_conf = smooth_label_from_history(tid, label, conf)
        distance_m = estimate_distance_m(smooth_label, bbox, fw, fh)

        if smooth_label in VEHICLE_LABELS and is_moving:
            ttc_s, risk = ttc_and_risk(distance_m, speed_kmh)
        else:
            ttc_s, risk = 999.0, 0.0

        x1, y1, x2, y2 = bbox
        out.append(
            DetectionOut(
                track_id=tid,
                label=smooth_label,
                confidence=smooth_conf,
                bbox_xyxy=[x1, y1, x2, y2],
                distance_m=distance_m,
                speed_kmh=speed_kmh,
                is_moving=is_moving,
                ttc_s=ttc_s,
                risk_percent=risk,
                track_age_s=track_age_s,
            )
        )

    out.sort(key=lambda d: d.confidence, reverse=True)
    update_trip_stats(out, now_s)
    return AnalyzeResponse(
        frame_time_s=now_s,
        detections=out,
        trip=trip_snapshot(now_s),
        frame_diagnostics=diagnostics,
    )


def io_bytes(raw: bytes):
    import io

    return io.BytesIO(raw)


# One-server deploy option: serve exported web build from backend process.
WEB_BUILD_DIR = Path(
    os.environ.get(
        "CAR_VISION_WEB_DIR",
        str((Path(__file__).resolve().parent.parent / "web-dist")),
    )
).resolve()

if WEB_BUILD_DIR.exists():
    for static_subdir in ("assets", "_expo", "static"):
        static_path = WEB_BUILD_DIR / static_subdir
        if static_path.exists():
            app.mount(f"/{static_subdir}", StaticFiles(directory=str(static_path)), name=f"web_{static_subdir}")

    @app.get("/", include_in_schema=False)
    def web_index() -> FileResponse:
        return FileResponse(str(WEB_BUILD_DIR / "index.html"))

    @app.get("/{full_path:path}", include_in_schema=False)
    def web_spa_fallback(full_path: str) -> FileResponse:
        # Never shadow API/documentation routes.
        if full_path.startswith(("health", "calibration", "demo-detections", "trip", "analyze-image", "docs", "openapi.json", "redoc")):
            raise HTTPException(status_code=404, detail="Not found")
        target = (WEB_BUILD_DIR / full_path).resolve()
        try:
            target.relative_to(WEB_BUILD_DIR)
        except ValueError:
            return FileResponse(str(WEB_BUILD_DIR / "index.html"))
        if target.exists() and target.is_file():
            return FileResponse(str(target))
        return FileResponse(str(WEB_BUILD_DIR / "index.html"))
