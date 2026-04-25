# syntax=docker/dockerfile:1.7

FROM node:20-bookworm-slim AS web_build
WORKDIR /app

COPY package*.json ./
RUN npm install --legacy-peer-deps

COPY . .
RUN npx expo export --platform web --output-dir web-dist


FROM python:3.11-slim AS runtime
WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

COPY backend/requirements.txt /app/backend/requirements.txt
RUN pip install -r /app/backend/requirements.txt

COPY backend /app/backend
COPY --from=web_build /app/web-dist /app/web-dist

ENV CAR_VISION_WEB_DIR=/app/web-dist
ENV PORT=8001

EXPOSE 8001
WORKDIR /app/backend
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT}"]
