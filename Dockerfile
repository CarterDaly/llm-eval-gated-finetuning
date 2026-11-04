FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    FLASK_APP=gated_finetune.app

WORKDIR /app

# System deps for psycopg2-binary and umap-learn
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Run as an unprivileged user. The code stays root-owned (read-only to the app);
# only /app/models, where uploaded models are saved, is writable. Creating it
# here means the named volume mounted over it starts out owned by this user.
RUN useradd --create-home --uid 1000 app \
    && mkdir -p /app/models \
    && chown app /app/models

COPY . .

USER app

# Bind address, worker count and Prometheus setup live in gunicorn.conf.py.
CMD ["gunicorn", "gated_finetune.app:create_app()"]
