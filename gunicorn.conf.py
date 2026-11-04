"""Gunicorn settings. Gunicorn loads ./gunicorn.conf.py automatically."""
import os
import shutil

bind = os.environ.get("GUNICORN_BIND", "0.0.0.0:5000")
workers = int(os.environ.get("GUNICORN_WORKERS", "2"))
timeout = 120

# Prometheus counters live in each worker process, so prometheus_client shares
# them through files in this directory (see gated_finetune/monitoring.py). It
# picks file-backed values at import time, so the variable is set here in the
# master, before any worker imports prometheus_client.
_PROMETHEUS_DIR = os.environ.setdefault("PROMETHEUS_MULTIPROC_DIR", "/tmp/prometheus_multiproc")


def on_starting(server):
    # Start from an empty directory so a restart doesn't resume stale totals.
    shutil.rmtree(_PROMETHEUS_DIR, ignore_errors=True)
    os.makedirs(_PROMETHEUS_DIR)


def child_exit(server, worker):
    # Imported here rather than at the top so the master never loads
    # prometheus_client before PROMETHEUS_MULTIPROC_DIR is set.
    from prometheus_client import multiprocess

    multiprocess.mark_process_dead(worker.pid)
