# Logging lab

A local stack that generates application logs and collects them with Promtail and Loki. Grafana is included for browsing logs. The agent exposes sample service routes, metrics and a basic correlation summary.

Install Docker with Compose, then run from this folder:

```powershell
docker compose -f log-intelligence-lab/docker-compose.yml up --build -d
```

- Agent: http://localhost:5050
- Metrics: http://localhost:5050/metrics
- Correlation: http://localhost:5050/summary
- Loki: http://localhost:3100

The summary groups matching log lines by a request-like identifier and reports the first error. It is a heuristic, not a verified root-cause diagnosis. Generated service errors are intentional test traffic.

To stop the stack:

```powershell
docker compose -f log-intelligence-lab/docker-compose.yml down
```

Logs are written under `log-intelligence-lab/logs/`. Keep them out of Git. The older nested stack has been consolidated into this one; its Loki correlation code is in `agent/utils/correlation.py`.
