# core/metrics.py
from prometheus_client import Counter, Histogram

REQUEST_COUNT = Counter(
    "esnafiz_request_total",
    "Total HTTP requests",
    ["method", "endpoint", "status_code"]
)

REQUEST_DURATION = Histogram(
    "esnafiz_request_duration_seconds",
    "Request duration in seconds",
    ["endpoint"]
)

ERROR_COUNT = Counter(
    "esnafiz_error_total",
    "Total errors",
    ["type"]
)
