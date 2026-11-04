import os
import subprocess
import sys

import pytest

from gated_finetune.app import create_app
from gated_finetune.monitoring import render_metrics


@pytest.fixture
def client():
    app = create_app()
    app.config["TESTING"] = True

    with app.test_client() as client:
        yield client


def test_health_route(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.get_json() == {"status": "ok"}


def test_metrics_endpoint_returns_prometheus_text(client):
    response = client.get("/metrics")
    assert response.status_code == 200
    assert response.content_type.startswith("text/plain")


def test_metrics_sum_counters_across_worker_processes(tmp_path, monkeypatch):
    """Under gunicorn each worker counts in its own process; a scrape must
    report the total, once, not one worker's share or duplicate series."""
    increment = (
        "from gated_finetune.monitoring import REQUEST_COUNT; "
        "REQUEST_COUNT.labels(method='GET', endpoint='/x', http_status=200).inc()"
    )
    # A dedicated directory: the collector reads every *.db file in it, and
    # tmp_path also holds this test's SQLite database.
    prom_dir = tmp_path / "prometheus"
    prom_dir.mkdir()
    env = {**os.environ, "PROMETHEUS_MULTIPROC_DIR": str(prom_dir)}
    for _ in range(2):
        subprocess.run([sys.executable, "-c", increment], env=env, check=True)

    monkeypatch.setenv("PROMETHEUS_MULTIPROC_DIR", str(prom_dir))
    body, _ = render_metrics()

    series = [line for line in body.decode().splitlines() if line.startswith("request_count_total")]
    assert series == ['request_count_total{endpoint="/x",http_status="200",method="GET"} 2.0']
