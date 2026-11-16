FROM node:20-bookworm-slim AS frontend
WORKDIR /build/game
COPY game/package*.json ./
RUN npm ci
COPY game/ ./
RUN npm run build

FROM python:3.12-slim-bookworm
WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    PORT=7860 \
    HOST=0.0.0.0 \
    DEPLOYMENT_MODE=online

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
COPY --from=frontend /build/game/dist ./game/dist

EXPOSE 7860
CMD ["python", "app.py"]
