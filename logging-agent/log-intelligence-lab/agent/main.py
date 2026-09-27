from flask import Flask, request, jsonify
import socket
import logging
import time
import random
import threading
from prometheus_client import start_http_server, Counter, Histogram, generate_latest
from logging.handlers import RotatingFileHandler
from pathlib import Path
from utils.correlation import analyze_logs

Path("logs").mkdir(exist_ok=True)

from prometheus_client import start_http_server

start_http_server(9000)

app = Flask(__name__)


formatter = logging.Formatter("[%(asctime)s] %(levelname)s in %(module)s: %(message)s")
handler = logging.StreamHandler()
handler.setFormatter(formatter)
app.logger.setLevel(logging.DEBUG)
app.logger.addHandler(handler)


REQUEST_COUNT = Counter(
    "app_requests_total", "Total Requests", ["service", "method", "endpoint"]
)
REQUEST_LATENCY = Histogram(
    "app_request_latency_seconds", "Request latency", ["service", "endpoint"]
)


def spam_logs(service_name):

    logger = logging.getLogger(service_name)
    logger.setLevel(logging.DEBUG)

    file_handler = RotatingFileHandler(
        f"logs/{service_name}.log", maxBytes=5 * 1024 * 1024, backupCount=3
    )
    file_handler.setFormatter(formatter)

    if not logger.handlers:
        logger.addHandler(handler)
        logger.addHandler(file_handler)

    levels = [logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR]
    while True:
        log_level = random.choice(levels)
        msg = f"{service_name} - Simulated log message at level {logging.getLevelName(log_level)}"
        logger.log(log_level, msg)
        time.sleep(random.uniform(0.7, 0.9))


def create_service_route(service_name):
    def handler():
        logger = logging.getLogger(service_name)
        start_time = time.time()
        latency = random.uniform(0.1, 1.5)
        time.sleep(latency)

        hostname = socket.gethostname()

        REQUEST_COUNT.labels(
            service=service_name, method=request.method, endpoint=f"/{service_name}"
        ).inc()
        REQUEST_LATENCY.labels(
            service=service_name, endpoint=f"/{service_name}"
        ).observe(latency)

        if random.random() < 0.2:
            logger.error(f"{service_name} - Simulated error occurred on {hostname}")
            return jsonify(
                {"status": "error", "service": service_name, "hostname": hostname}
            ), 500

        logger.info(f"{service_name} - Request handled successfully on {hostname}")
        return jsonify(
            {
                "status": "ok",
                "service": service_name,
                "latency": latency,
                "hostname": hostname,
            }
        )

    app.add_url_rule(
        f"/{service_name}", endpoint=service_name, view_func=handler, methods=["GET"]
    )


for svc in ["service1", "service2", "service3"]:
    create_service_route(svc)
    threading.Thread(target=spam_logs, args=(svc,), daemon=True).start()


@app.route("/summary")
def summary():
    return jsonify(analyze_logs())


@app.route("/metrics")
def metrics():
    return generate_latest(), 200, {"Content-Type": "text/plain; charset=utf-8"}


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
