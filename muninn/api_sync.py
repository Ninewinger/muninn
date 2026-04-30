"""
Muninn API — synchronous Flask server (fallback when asyncio/uvicorn is broken).
Drop-in replacement for api.py, serves the /api/v1/route/inject endpoint.
"""

import os
from flask import Flask, request, jsonify

# Muninn imports
from muninn.router_v2 import route_with_context_injection

app = Flask(__name__)

# Simple request model
class RouteRequest:
    def __init__(self, text: str, top_k: int = 3):
        self.text = text
        self.top_k = top_k


@app.post("/api/v1/route/inject")
def route_inject():
    """Route text and return formatted context injection string."""
    body = request.get_json(force=True)
    text = body.get("text", "")
    top_k = body.get("top_k", 3)

    if not text:
        return jsonify({"query": "", "injection": None})

    try:
        injection = route_with_context_injection(
            text=text,
            top_k=top_k,
        )
        return jsonify({"query": text, "injection": injection})
    except Exception as e:
        return jsonify({"query": text, "injection": None, "error": str(e)}), 500


@app.get("/api/v1/health")
def health():
    return jsonify({"status": "ok", "server": "flask-sync"})


@app.get("/api/v1/route")
def route_get():
    """Route text via GET for quick testing."""
    text = request.args.get("text", "")
    top_k = int(request.args.get("top_k", 3))

    if not text:
        return jsonify({"activations": []})

    try:
        result = route_with_context_injection(
            text=text,
            top_k=top_k,
        )
        return jsonify({"query": text, "injection": result})
    except Exception as e:
        return jsonify({"query": text, "error": str(e)}), 500


if __name__ == "__main__":
    os.environ.setdefault("HF_HOME", "D:\\hf_cache")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("DB_PATH", "D:\\github\\muninn\\muninn.db")
    os.environ.setdefault("SYSTEMROOT", "C:\\Windows")
    print("=" * 50)
    print("Muninn API (Flask sync) starting on :8000")
    print("=" * 50)
    app.run(host="0.0.0.0", port=8000, debug=False)
