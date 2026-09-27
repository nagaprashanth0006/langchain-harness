import requests
import streamlit as st

SERVER_URL = "http://localhost:8000/call"


def call_tool(name, params):
    response = requests.post(
        SERVER_URL, json={"tool_name": name, "params": params}, timeout=180
    )
    response.raise_for_status()
    return response.json()


st.title("Log analysis")
query = st.text_input("Question", "Which applications appear in these logs?")
loki_url = st.text_input("Loki URL", "http://localhost:3100")
loki_query = st.text_input("LogQL query", '{stream="stdout"} != `LoadGen`')

if st.button("Analyse"):
    try:
        with st.status("Reading logs"):
            result = call_tool(
                "log_tool",
                {
                    "loki_url": loki_url,
                    "loki_query": loki_query,
                    "duration": 1800,
                    "limit": 500,
                },
            )
            logs = [
                entry[1]
                for stream in result.get("data", {}).get("result", [])
                for entry in stream.get("values", [])
            ]
            if not logs:
                st.info("No logs matched. Try a different query or time range.")
            else:
                with st.expander(f"Logs ({len(logs)})"):
                    st.code("\n".join(logs))
                prompt = (
                    f"Answer using only these logs. Say when the evidence is missing.\nQuestion: {query}\nLogs:\n"
                    + "\n".join(logs)
                )
                answer = call_tool(
                    "run_inference", {"prompt": prompt, "model": "mistral"}
                )
                st.write(answer)
    except (requests.RequestException, ValueError) as exc:
        st.error(f"Could not complete the request: {exc}")
