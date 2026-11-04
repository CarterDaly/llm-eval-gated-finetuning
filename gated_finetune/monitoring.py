import os
import time

from flask import request
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    REGISTRY,
    CollectorRegistry,
    Counter,
    Histogram,
    generate_latest,
    multiprocess,
)

REQUEST_COUNT = Counter(
    "request_count", "Total number of requests", ["method", "endpoint", "http_status"]
)
REQUEST_LATENCY = Histogram(
    "request_latency_seconds", "Request latency in seconds", ["method", "endpoint", "http_status"]
)


def render_metrics() -> tuple[bytes, str]:
    """Return the Prometheus exposition body and its content type.

    Under gunicorn every worker is a separate process with its own counters.
    gunicorn.conf.py sets PROMETHEUS_MULTIPROC_DIR, prometheus_client then writes
    each worker's values to files there, and a fresh collector sums them on every
    scrape. Without the variable (flask run, tests) there is one process and the
    default in-memory registry is already complete.
    """
    if "PROMETHEUS_MULTIPROC_DIR" in os.environ:
        registry = CollectorRegistry()
        multiprocess.MultiProcessCollector(registry)
        return generate_latest(registry), CONTENT_TYPE_LATEST
    return generate_latest(REGISTRY), CONTENT_TYPE_LATEST


def _start_timer():
    request.start_time = time.time()


def _record_request(response):
    latency = time.time() - getattr(request, "start_time", time.time())
    path = getattr(request.url_rule, "rule", request.path)
    labels = dict(method=request.method, endpoint=path, http_status=response.status_code)
    REQUEST_COUNT.labels(**labels).inc()
    REQUEST_LATENCY.labels(**labels).observe(latency)
    return response


def init_metrics(app):
    app.before_request(_start_timer)
    app.after_request(_record_request)
