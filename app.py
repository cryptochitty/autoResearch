"""
Flask web UI for AutoResearch.
Run:  python app.py
Open: http://localhost:5000
"""

import json
import queue
import threading
import uuid
import os
from datetime import datetime, timedelta
from flask import Flask, Response, jsonify, render_template, request, send_file
import anthropic

from crew import run_research_streamed, SECTION_LABELS, OUTPUT_DIR

app = Flask(__name__)

# session_id → {"queue": Queue, "status": str, "started": datetime}
_sessions: dict[str, dict] = {}
_sessions_lock = threading.Lock()

AGENT_NAMES = [
    "Discovery Agent",
    "Reader Agent",
    "Innovation Agent",
    "Validation Agent",
    "Writer Agent",
    "Builder Agent",
    "Evaluation Agent",
    "Explainer Agent",
]


def _cleanup_old_sessions() -> None:
    """Drop sessions older than 2 hours."""
    cutoff = datetime.utcnow() - timedelta(hours=2)
    with _sessions_lock:
        stale = [k for k, v in _sessions.items() if v["started"] < cutoff]
        for k in stale:
            del _sessions[k]


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/start", methods=["POST"])
def start():
    """Start a research job. Returns {session_id}."""
    data = request.get_json(force=True)
    topic = (data.get("topic") or "").strip()
    if not topic:
        return jsonify({"error": "topic is required"}), 400

    _cleanup_old_sessions()

    session_id = str(uuid.uuid4())
    q: queue.Queue = queue.Queue()

    with _sessions_lock:
        _sessions[session_id] = {
            "queue": q,
            "topic": topic,
            "status": "running",
            "started": datetime.utcnow(),
        }

    def worker():
        run_research_streamed(topic, q)
        with _sessions_lock:
            if session_id in _sessions:
                _sessions[session_id]["status"] = "done"

    threading.Thread(target=worker, daemon=True).start()
    return jsonify({"session_id": session_id})


@app.route("/stream/<session_id>")
def stream(session_id: str):
    """Server-Sent Events endpoint. Emits events until job is done."""

    with _sessions_lock:
        session = _sessions.get(session_id)

    if not session:
        def _err():
            yield 'data: {"type":"error","message":"Session not found"}\n\n'
        return Response(_err(), mimetype="text/event-stream")

    q: queue.Queue = session["queue"]

    def generate():
        # Initial handshake
        yield f'data: {json.dumps({"type": "connected", "topic": session["topic"]})}\n\n'

        while True:
            try:
                event = q.get(timeout=45)
                yield f"data: {json.dumps(event)}\n\n"
                if event.get("type") in ("complete", "error"):
                    break
            except queue.Empty:
                # keepalive ping so the browser doesn't close the connection
                yield 'data: {"type":"ping"}\n\n'

    return Response(generate(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.route("/download/<session_id>")
def download(session_id: str):
    """Download the markdown report for a completed session."""
    with _sessions_lock:
        session = _sessions.get(session_id)
    if not session:
        return jsonify({"error": "session not found"}), 404

    # Find the most recent .md file for this topic
    slug = session["topic"][:40].replace(" ", "_").replace("/", "-")
    candidates = [
        f for f in os.listdir(OUTPUT_DIR)
        if f.endswith(".md") and slug in f
    ]
    if not candidates:
        return jsonify({"error": "report not yet saved"}), 404

    latest = sorted(candidates)[-1]
    return send_file(
        os.path.join(os.path.abspath(OUTPUT_DIR), latest),
        as_attachment=True,
        download_name=f"research_{slug}.md",
    )


@app.route("/ai/ask", methods=["POST"])
def ai_ask():
    """Simple AI Q&A endpoint for mobile app."""
    data = request.get_json(force=True)
    prompt = (data.get("prompt") or "").strip()
    tier = data.get("tier", "free")

    if not prompt:
        return jsonify({"error": "prompt is required"}), 400

    system = (
        "You are an advanced AI assistant. Give detailed, comprehensive answers."
        if tier == "premium"
        else "You are a helpful AI assistant. Give concise answers under 200 words."
    )

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=2048 if tier == "premium" else 512,
        system=system,
        messages=[{"role": "user", "content": prompt}],
    )
    return jsonify({"response": message.content[0].text})


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    app.run(debug=False, host="0.0.0.0", port=5000)
