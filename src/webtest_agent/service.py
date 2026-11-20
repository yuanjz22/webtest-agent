"""Shared execution service for CLI and the local Web workbench."""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from .actor import ActorAgent
from .assertor import AssertorAgent
from .browser import PlaywrightBrowser
from .demo_planner import DemoPlanner
from .llm import ModelClient
from .models import Scenario
from .planner import OrchestratorAgent
from .runner import run_scenario
from .storage import RunStore


def execute_run(store: RunStore, scenario: Scenario, run_id: str,
                mode: str = "llm", strategy: str = "checkpoint") -> dict:
    """Persist terminal status even if browser startup or model setup fails."""
    if mode not in {"llm", "demo"} or strategy not in {"baseline", "checkpoint"}:
        raise ValueError("Unsupported execution mode or strategy")
    store.set_status(run_id, "running")
    directory = store.artifact_dir(run_id)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "input.json").write_text(json.dumps(asdict(scenario), ensure_ascii=False, indent=2), encoding="utf-8")

    def on_event(event: dict) -> None:
        with (directory / "events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event, ensure_ascii=False) + "\n")

    try:
        if mode == "demo":
            orchestrator = actor = assertor = DemoPlanner()
        else:
            client = ModelClient()
            orchestrator = OrchestratorAgent(client)
            actor = ActorAgent(client)
            assertor = AssertorAgent(client)
        with PlaywrightBrowser(scenario.base_url, directory) as browser:
            report = run_scenario(scenario, orchestrator, browser, run_id, actor=actor, assertor=assertor,
                                  strategy=strategy, on_event=on_event)
        # The browser context has closed and finalized trace.zip before publishing.
        store.save(report)
        data = asdict(report)
        (directory / "report.json").write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        return data
    except Exception as exc:
        # Do not expose request headers or model credentials in the API response.
        message = str(exc)
        import os
        secret = os.environ.get("MODEL_API_KEY", "")
        if secret:
            message = message.replace(secret, "[REDACTED]")
        store.set_status(run_id, "failed", error=f"{type(exc).__name__}: {message[:600]}")
        raise RuntimeError(f"Run {run_id} failed: {message[:600]}") from None


def recent_events(directory: Path, limit: int = 20) -> list[dict]:
    source = directory / "events.jsonl"
    if not source.exists():
        return []
    events = []
    for line in source.read_text(encoding="utf-8").splitlines()[-limit:]:
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            pass  # A writer may be in the middle of its latest line.
    return events


