# Car Vision Backend (Python)

This backend provides:
- `/health` for status and config summary
- `/demo-detections` returns an empty list (no synthetic random labels)
- `/analyze-image` runs YOLO on each uploaded frame (multi-object, real-time friendly) and returns **`trip`** counters (Phase 4)
- **Phase 4 — trip / near-miss (in-memory per server process):**
  - `GET /trip/stats` — frames, danger/caution/safe frame counts, elapsed time, `near_miss_count`, and `events` (newest first; debounced CAUTION/DANGER on moving vehicles)
  - `POST /trip/reset` — clears counters and the event log (`{"ok": true}`)

## 1) Setup

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## 2) Run

```bash
uvicorn main:app --host 0.0.0.0 --port 8001
```

## 3) Test

- Health: `http://127.0.0.1:8001/health`
- Swagger: `http://127.0.0.1:8001/docs`

## Robustness tuning (environment variables)

All labels come from the model only, plus temporal smoothing and heuristics to cut common false positives (e.g. phone vs person).

| Variable | Default | Purpose |
|----------|---------|---------|
| `CAR_VISION_YOLO_MODEL` | `yolov8s.pt` | Stronger than `n`; use `yolov8m.pt` for more accuracy if GPU/CPU allows |
| `CAR_VISION_YOLO_CONF` | `0.38` | Raise (e.g. `0.45`) if you still see false positives |
| `CAR_VISION_YOLO_IOU` | `0.5` | NMS IoU |
| `CAR_VISION_YOLO_MAX_DET` | `60` | Max simultaneous objects per frame |
| `CAR_VISION_YOLO_IMGSZ` | `640` | Inference size |
| `CAR_VISION_CLASS_HISTORY` | `7` | Frames of class history for temporal vote |
| `CAR_VISION_CLASS_SWITCH_FRAMES` | `3` | Hysteresis: need this many consistent frames to change class |
| `CAR_VISION_PERSON_MIN_CONF_SMALL` | `0.62` | Drop tiny `person` boxes below this confidence |
| `CAR_VISION_PERSON_MAX_AREA_FRAC` | `0.0045` | Max bbox area fraction for weak-person filter |
| `CAR_VISION_PHONE_PERSON_IOU` | `0.25` | If person overlaps phone above this IoU, prefer phone |
| `CAR_VISION_NEAR_MISS_DEBOUNCE_S` | `4.0` | Min seconds between duplicate log lines for same track + severity |
| `CAR_VISION_TRIP_EVENTS_MAX` | `80` | Max stored trip events (ring buffer) |

## Notes

- Speed, distance, and collision risk use heuristics; for bike production, calibrate with camera height and road geometry.
- CORS is enabled for browser dev.
- For best rider safety, consider a **fine-tuned** YOLO on your own dash/bike-mounted dataset; generic COCO will always have edge cases.
