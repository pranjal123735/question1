# Cloud Server Options + Deployment

## Frontend-only on Render (recommended for your current setup)

You are currently running backend on local machine via ngrok, so deploy only the frontend to Render static hosting.

### Steps

1. Push latest code to GitHub (already includes `render.yaml`).
2. In Render: **New +** -> **Blueprint** -> connect repo.
3. Render will detect `render.yaml` and create static site `camera-app-frontend`.
4. In Render env vars, set:
   - `EXPO_PUBLIC_BACKEND_URL=https://uncleansed-overserene-elijah.ngrok-free.dev`
5. Deploy.
6. Open Render frontend URL and test detection.

### Important

- Keep your local backend + ngrok running, otherwise frontend cannot call API.
- If ngrok URL changes, only update `EXPO_PUBLIC_BACKEND_URL` in Render and redeploy.

---

This project is also ready to deploy as **one container** (backend + frontend).

## Best server choices (for your project)

1. **RunPod (GPU optional)**  
   Best for heavier YOLO inference and future scaling.

2. **Railway (easy Docker deploy)**  
   Fastest setup for prototype, good developer UX.

3. **Render (easy Docker deploy)**  
   Stable PaaS flow, simple web service deploy.

4. **VPS (Hetzner/Contabo/DigitalOcean)**  
   Full control. Good if you want your own infra.

## Quick recommendation

- If you want easiest prototype: **Railway** or **Render**
- If you want strong inference: **RunPod GPU**

---

## A) Deploy with Railway (Docker)

1. Push this repo to GitHub.
2. In Railway:
   - New Project -> Deploy from GitHub Repo
   - Railway will detect `Dockerfile`.
3. Add env vars (optional):
   - `CAR_VISION_YOLO_MODEL=yolov8s.pt`
   - `CAR_VISION_YOLO_CONF=0.38`
4. Deploy.
5. Open generated URL:
   - `/` frontend
   - `/docs` backend docs

---

## B) Deploy with Render (Docker)

1. Push repo to GitHub.
2. Render -> New -> Web Service -> Connect repo.
3. Environment:
   - Runtime: Docker
   - Auto deploy: ON
4. Set env vars as needed.
5. Deploy and open service URL.

---

## C) Deploy on your own VPS (Docker)

```bash
git clone <your-repo-url>
cd camera-app
docker build -t bike-vision:latest .
docker run -d --name bike-vision -p 8001:8001 bike-vision:latest
```

Open:
- `http://<server-ip>:8001/`
- `http://<server-ip>:8001/docs`

---

## Notes

- Frontend uses same-origin backend by default (`window.location.origin`), so one-container deploy works out of the box.
- If you run behind reverse proxy, keep forwarded headers configured normally.
- For production safety use, monitor latency/FPS and consider GPU instance or on-edge fallback.

## Render-specific troubleshooting

If `/health` shows:
- `"mode":"demo"`
- `"model": null`

check `"model_load_error"` from `/health` and redeploy with:
- lighter model: `CAR_VISION_YOLO_MODEL=yolov8n.pt`
- Docker image including OpenCV runtime libs (`libglib2.0-0`, `libgl1`, `libgomp1`)
