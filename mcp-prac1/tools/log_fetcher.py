import requests
import time


def infer_with_ollama(prompt, model="mistral"):
    OLLAMA_ENDPOINT = "http://localhost:11434/api/generate"
    response = requests.post(
        OLLAMA_ENDPOINT,
        json={"prompt": prompt, "model": model, "stream": False},
        proxies={},
        timeout=180,
    )
    response.raise_for_status()
    result = response.json()["response"]
    return result


def daily_quote(dummy):
    return infer_with_ollama("Tell me a motivational quote", model="mistral")


def fetch_logs(loki_url, loki_query, duration: int = 1800, limit: int = 200):
    start = int(time.time() * 1e9 - duration * 1e9)
    end = int(time.time() * 1e9)
    url = f"{loki_url}/loki/api/v1/query_range"

    params = {
        "query": loki_query,
        "start": start,
        "end": end,
        "limit": limit,
    }
    response = requests.get(url, params=params, proxies={}, timeout=30)
    response.raise_for_status()
    logs = response.json()
    return logs
