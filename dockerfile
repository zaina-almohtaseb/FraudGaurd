# Dockerfile
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    HOST=0.0.0.0 \
    PORT=5000

WORKDIR /app

# Install deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App code
COPY backend ./backend
COPY model   ./model
COPY src     ./src
RUN mkdir -p /app/data

# Default env (compose can override)
ENV ADMIN_TOKEN=changeme \
    RETRAIN_THRESHOLD=10 \
    CHECK_INTERVAL_MIN=1 \
    FRONTEND_ORIGIN=http://localhost:8080 \
    APP_DB_PATH=/app/data/fraud.db

EXPOSE 5000

# IMPORTANT: run as a module so `from backend...` works
CMD ["python", "-m", "backend.app"]
