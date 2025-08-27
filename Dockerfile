# syntax=docker/dockerfile:1

############################
# Stage 1: builder
############################
FROM python:3.11-slim AS builder
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

# System deps (build tools for some optional libs)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# Copy project
COPY . /app

# Optional: if you add requirements.txt later, uncomment:
# COPY requirements.txt /app/
# RUN pip install --upgrade pip && pip wheel --no-cache-dir --wheel-dir /wheels -r requirements.txt

# Install minimal runtime deps directly (since we don't have a requirements.txt here)
RUN pip install --upgrade pip \ 
    && pip install fastapi uvicorn[standard] prometheus-client pydantic-settings structlog

############################
# Stage 2: runtime
############################
FROM python:3.11-slim AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

# Copy installed site-packages from builder
# (This is a simple approach; for production, prefer pip install --no-cache-dir with hash-locked requirements.)
COPY --from=builder /usr/local/lib/python3.11 /usr/local/lib/python3.11
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy app code
COPY . /app

# Non-root user
RUN useradd -ms /bin/bash appuser
USER appuser

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
