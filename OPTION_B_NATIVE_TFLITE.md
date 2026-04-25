# Option B - Native TFLite (High Performance) + Python Backend (Recommended)

Main goal:
- Detect cars in real time
- Show car class/name above each car
- Estimate distance from camera (and between cars if needed)
- Estimate speed
- Estimate collision risk percentage (or time-to-collision)

This document gives a practical production path.

## Best Practical Architecture

For your feature set, the most reliable setup is:
- **Mobile App (React Native)**: camera, UI overlays, user controls
- **Python Inference Service**: detection, tracking, speed, distance, collision metrics

Why:
- React Native + TFLite is great for fast on-device detection
- Advanced metrics (tracking, perspective calibration, TTC logic) are easier and more accurate in Python/OpenCV
- You can still move some models on-device later for offline mode

## Core Pipeline (What To Build)

1. **Vehicle Detection**
   - Model: YOLO family (`yolov8n`/`yolov8s`, or TFLite export)
   - Output: bounding boxes + class label (`car`, `truck`, `bus`, `motorcycle`)

2. **Multi-Object Tracking**
   - Tracker: ByteTrack (or DeepSORT)
   - Output: stable `track_id` for each vehicle across frames

3. **Speed Estimation**
   - Use per-track position history over time
   - Convert pixels -> meters using camera calibration/perspective transform
   - Speed formula:
     - `speed_mps = distance_m / delta_time_s`
     - `speed_kmh = speed_mps * 3.6`

4. **Distance Estimation**
   - **Best accuracy**: camera calibration + known lane/road markers + perspective transform
   - **Optional boost**: monocular depth model (MiDaS) for relative depth support
   - Show distance from camera per tracked car

5. **Collision Risk / TTC**
   - Compute relative distance and relative closing speed
   - Time-to-collision:
     - `ttc = relative_distance / max(relative_closing_speed, eps)`
   - Convert to risk percentage with thresholds, example:
     - TTC < 1.5s => 90-100% (high risk)
     - TTC 1.5-3.0s => 50-90% (medium risk)
     - TTC > 3.0s => low risk

6. **Overlay Rendering**
   - For each tracked car draw:
     - label: `car #17`
     - speed: `42 km/h`
     - distance: `18.7 m`
     - risk: `72%`

## React Native Native-TFLite Status

Current project already has:
- `react-native-vision-camera`
- `react-native-fast-tflite`
- `vision-camera-resize-plugin`
- `react-native-worklets-core`
- Expo config plugins in `app.json`
- `.tflite` asset support in `metro.config.js`
- Worklets Babel plugin in `babel.config.js`

Important:
- This setup does **not** run in Expo Go
- Use development build / EAS build

## Immediate Next Steps (In This Repo)

1. Install dependencies:

```bash
npm install --legacy-peer-deps
```

2. Prebuild native projects:

```bash
npx expo prebuild
```

3. Run Android dev build:

```bash
npx expo run:android
```

4. Add model file(s) in `assets/models/`:
   - Detection `.tflite` model first (fast and lightweight)

5. Integrate frame processor path:
   - Capture frame via VisionCamera
   - Resize/normalize input tensor
   - Run `model.runSync([input])`
   - Parse detections
   - Draw overlay boxes + labels

6. Add Python backend service for advanced metrics:
   - Input: frame stream or periodic detections
   - Output per object: `track_id`, `speed_kmh`, `distance_m`, `ttc_s`, `risk_percent`

## References Used (High Value)

- Roboflow tutorial: YOLO + ByteTrack + perspective transform for speed estimation
- Open-source projects on vehicle speed estimation pipelines (YOLO + tracking + calibration)
- MiDaS for monocular depth (relative depth; needs calibration for metric distance)

## Accuracy Notes (Important)

- Detection alone is easy; **accurate speed and distance need calibration**
- Monocular depth gives relative depth, not perfect meters by default
- Collision percentage should be treated as a heuristic unless trained on collision datasets
- Start with TTC-based risk and improve using real driving data

## If You Want "Most Accurate" First Version

Use **Python-first** for core CV:
- Python app: YOLO + ByteTrack + calibration + TTC
- React Native app: only UI and visualization

After quality is good, optimize parts to on-device TFLite.
