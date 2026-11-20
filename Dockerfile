FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    FLASK_APP=gated_finetune.app

WORKDIR /app

# Compilers for any dependency without a prebuilt wheel. (The Postgres driver,
# psycopg[binary], bundles its own libpq.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Run as an unprivileged user. The code stays root-owned (read-only to the app);
# only /app/models (uploaded models) and /app/.cache/fastembed (the prompt
# embedding model's weights) are writable. Creating them here means the named
# volumes mounted over them start out owned by this user.
ENV FASTEMBED_CACHE_PATH=/app/.cache/fastembed
RUN useradd --create-home --uid 1000 app \
    && mkdir -p /app/models /app/.cache/fastembed \
    && chown app /app/models /app/.cache/fastembed

COPY . .

USER app

# Bind address, worker count and Prometheus setup live in gunicorn.conf.py.
CMD ["gunicorn", "gated_finetune.app:create_app()"]
