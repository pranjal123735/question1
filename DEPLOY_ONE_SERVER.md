# Deploy Backend + Frontend on One Server

This project can be deployed as a single service:
- Backend API (FastAPI)
- Frontend web app (exported static build)

The backend now serves `web-dist` automatically when present.

## Structure

- `backend/main.py` -> API + static hosting
- `web-dist/` -> exported frontend files

## 1) Build frontend

From project root:

```bash
npm install
npm run build:web
```

This creates `web-dist/`.

## 2) Setup backend

From `backend/`:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## 3) Run one-server app

From `backend/`:

```bash
uvicorn main:app --host 0.0.0.0 --port 8001
```

Open:
- `http://<server-ip>:8001/` -> frontend
- `http://<server-ip>:8001/docs` -> API docs

## Optional env vars

- `CAR_VISION_WEB_DIR`  
  Override static web path (default: `../web-dist` from `backend/main.py`).

Example:

```bash
set CAR_VISION_WEB_DIR=D:\deploy\camera-app\web-dist
uvicorn main:app --host 0.0.0.0 --port 8001
```

## Notes

- Frontend default backend URL now auto-uses `window.location.origin` in web mode, so same-host deploy works by default.
- Rebuild frontend (`npm run build:web`) whenever UI code changes.
