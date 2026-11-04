# Multi-stage Dockerfile for CircuitGuard AI on Render
# Stage 1: Build React Frontend
FROM node:18-alpine AS frontend-builder
WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm install
COPY frontend/ ./
RUN npm run build

# Stage 2: Python Backend & Static Serving
FROM python:3.11-slim
WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV OMP_NUM_THREADS=1
ENV KMP_DUPLICATE_LIB_OK=True

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy backend, models, dataset, reports
COPY backend/ ./backend/
COPY models/ ./models/
COPY dataset/ ./dataset/
COPY reports/ ./reports/

# Copy built frontend assets from Stage 1
COPY --from=frontend-builder /app/frontend/dist ./frontend/dist

# Expose default port
EXPOSE 8000

# Start Uvicorn bound to Render's $PORT (defaults to 8000 if not set)
CMD uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-8000}

