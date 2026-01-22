import time

from flask import request
from prometheus_client import CollectorRegistry, Counter, Histogram, multiprocess

registry = CollectorRegistry()
multiprocess.MultiProcessCollector(registry=registry)

REQUEST_COUNT = Counter(name='requests_count', documentation='Total number of requests', labelnames=['method','endpoint'])


def before_request():
    request.start_time = time.time()

def after_request(response):
    latency = time.time() - request.start_time
    path = request.url_rule.rule
    REQUEST_COUNT.labels(method=request.method, endpoint=path, http_status=response.status_code).inc()
    REQUEST_COUNT.labels(method=request.method, endpoint=path, http_status=response.status_code).observe(latency)
    return response

def init_metrics(app):
    with app.app_context():
        app.before_request(before_request)
                                                                                                 