"""
Assembles the crew and runs it. Saves a structured Markdown report on completion.
"""

import os
import json
import queue
from datetime import datetime
from crewai import Crew, Process

from agents import create_agents
from tasks import create_tasks
from config import OUTPUT_DIR


# Section headings that match the 8-task order
SECTION_LABELS = [
    "📚 Papers Found",
    "🔍 Analysis Summary",
    "💡 Proposed Idea",
    "🧪 Experiment Plan",
    "📝 Research Paper",
    "⚙️ Prototype Code",
    "📊 Evaluation",
    "🎤 Explanation & Pitch",
]


def run_research(topic: str) -> dict:
    """Run the full 8-agent pipeline. Returns a dict of section → output."""

    agents = create_agents()
    tasks = create_tasks(agents, topic)

    crew = Crew(
        agents=list(agents.values()),
        tasks=tasks,
        process=Process.sequential,
        verbose=True,
    )

    print(f"\n{'='*60}")
    print(f"  AutoResearch  |  Topic: {topic}")
    print(f"  Model: {os.getenv('MODEL', 'claude-sonnet-4-6')}")
    print(f"{'='*60}\n")

    crew.kickoff()

    # Collect per-task outputs
    results: dict[str, str] = {}
    for label, task in zip(SECTION_LABELS, tasks):
        output = task.output
        results[label] = str(output.raw) if output else "(no output)"

    md_path, json_path = _save_report(topic, results)
    print(f"\n✅ Report saved:\n   Markdown → {md_path}\n   JSON     → {json_path}")
    return results


def run_research_streamed(topic: str, event_queue: queue.Queue) -> None:
    """
    Run the full pipeline in the calling thread, pushing SSE-ready dicts
    into event_queue as each task completes.

    Event shapes
    ────────────
    {"type": "start",     "total": 8}
    {"type": "agent_start","step": N, "label": "...", "agent": "..."}
    {"type": "step",      "step": N, "message": "..."}      # tool calls etc.
    {"type": "task_done", "step": N, "label": "...", "preview": "..."}
    {"type": "complete",  "results": {...}, "md_path": "...", "json_path": "..."}
    {"type": "error",     "message": "..."}
    """
    try:
        agents = create_agents()
        tasks_list = create_tasks(agents, topic)
        total = len(tasks_list)

        event_queue.put({"type": "start", "total": total})

        # Track which task is currently running via step_callback
        current_step = {"n": 0}

        def step_callback(output):
            try:
                msg = (
                    output.thought
                    if hasattr(output, "thought") and output.thought
                    else str(output)[:160]
                )
            except Exception:
                msg = str(output)[:160]
            event_queue.put({
                "type": "step",
                "step": current_step["n"],
                "message": msg,
            })

        # Attach per-task callbacks so we know exactly when each task finishes
        for i, (label, task) in enumerate(zip(SECTION_LABELS, tasks_list)):
            # capture loop vars
            def _make_cb(step_i, lbl):
                def _cb(output):
                    current_step["n"] = step_i + 1  # advance to next
                    preview = str(output.raw)[:400] if output else ""
                    event_queue.put({
                        "type": "task_done",
                        "step": step_i,
                        "label": lbl,
                        "preview": preview,
                    })
                return _cb
            task.callback = _make_cb(i, label)

            # Emit agent_start before kickoff (we know the order is sequential)
            # We do this inside the callback of the *previous* task, or upfront for task 0
            if i == 0:
                agent_name = list(agents.values())[0].role
                event_queue.put({
                    "type": "agent_start",
                    "step": 0,
                    "label": label,
                    "agent": agent_name,
                })

        crew = Crew(
            agents=list(agents.values()),
            tasks=tasks_list,
            process=Process.sequential,
            verbose=False,
            step_callback=step_callback,
        )

        crew.kickoff()

        # Collect results
        results: dict[str, str] = {}
        for label, task in zip(SECTION_LABELS, tasks_list):
            output = task.output
            results[label] = str(output.raw) if output else "(no output)"

        md_path, json_path = _save_report(topic, results)

        event_queue.put({
            "type": "complete",
            "results": results,
            "md_path": md_path,
            "json_path": json_path,
        })

    except Exception as exc:
        event_queue.put({"type": "error", "message": str(exc)})


def _save_report(topic: str, results: dict[str, str]) -> tuple[str, str]:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    slug = topic[:40].replace(" ", "_").replace("/", "-")

    md_path = os.path.join(OUTPUT_DIR, f"{timestamp}_{slug}.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# AutoResearch Report\n\n")
        f.write(f"**Topic:** {topic}  \n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}  \n\n")
        f.write("---\n\n")
        for section, content in results.items():
            f.write(f"## {section}\n\n{content}\n\n---\n\n")

    json_path = os.path.join(OUTPUT_DIR, f"{timestamp}_{slug}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"topic": topic, "timestamp": timestamp, "results": results}, f, indent=2)

    return md_path, json_path
