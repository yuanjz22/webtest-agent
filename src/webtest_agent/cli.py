"""Command line entry points. Demo mode is a fixture, never an experiment result."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

from .demo import DEFAULT_DEMO_PORT, serve_demo
from .models import Scenario
from .storage import RunStore


def load_scenarios(path: Path) -> list[Scenario]:
    raw = path.read_text(encoding="utf-8-sig")
    rows = [json.loads(line) for line in raw.splitlines() if line.strip()] if path.suffix == ".jsonl" else json.loads(raw)
    if isinstance(rows, dict):
        rows = [rows]
    if not isinstance(rows, list) or not rows:
        raise ValueError("Scenario file must contain at least one scenario")
    ids = set()
    scenarios = []
    for row in rows:
        if not isinstance(row, dict) or set(row) - {"id", "description", "base_url", "max_steps"}:
            raise ValueError("Unexpected fields: do not pass oracle labels to the agent")
        from .api import RunInput
        validated = RunInput(**row)
        if validated.id in ids:
            raise ValueError(f"Duplicate scenario id: {validated.id}")
        ids.add(validated.id)
        scenarios.append(Scenario(validated.id, validated.description, validated.base_url, validated.max_steps))
    return scenarios


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="webtest")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="Run JSON/JSONL scenarios sequentially")
    run.add_argument("scenarios", type=Path)
    run.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    run.add_argument("--strategy", choices=["baseline", "checkpoint"], default="checkpoint")
    run.add_argument("--demo", action="store_true", help="Local scripted fixture only; not a model benchmark")
    serve = sub.add_parser("serve", help="Start local workbench")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    demo = sub.add_parser("demo-server", help="Start the isolated demonstration application")
    demo.add_argument("--port", type=int, default=DEFAULT_DEMO_PORT)
    evaluate = sub.add_parser("evaluate", help="Score independently reviewed reports offline")
    evaluate.add_argument("manifest", type=Path)
    evaluate.add_argument("reviews", type=Path)
    evaluate.add_argument("--output", type=Path, default=Path("artifacts/evaluation.json"))
    args = parser.parse_args(argv)
    try:
        load_dotenv(Path.cwd() / ".env", override=False, encoding="utf-8-sig")
        if args.command == "serve":
            import uvicorn
            from .api import create_app
            uvicorn.run(create_app(args.artifacts), host="127.0.0.1", port=args.port)
            return 0
        if args.command == "demo-server":
            serve_demo(port=args.port)
            return 0
        if args.command == "evaluate":
            from .evaluation import evaluate_manifest
            result = evaluate_manifest(args.manifest, args.reviews)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0
        from .service import execute_run
        scenarios = load_scenarios(args.scenarios)
        store = RunStore(args.artifacts)
        errors = 0
        results = []
        for scenario in scenarios:
            if args.demo:
                from urllib.parse import urlsplit
                if scenario.id != "local-demo" or urlsplit(scenario.base_url).hostname not in {"localhost", "127.0.0.1", "::1"}:
                    raise ValueError("--demo supports only local-demo on a loopback address")
            run_id = store.create(scenario)
            try:
                report = execute_run(store, scenario, run_id, "demo" if args.demo else "llm", args.strategy)
                results.append({"run_id": run_id, "scenario_id": scenario.id, "verdict": report["verdict"], "steps": report["steps"]})
                errors += report["verdict"] == "INCONCLUSIVE"
            except RuntimeError as exc:
                errors += 1
                results.append({"run_id": run_id, "scenario_id": scenario.id, "error": str(exc)})
        print(json.dumps(results, ensure_ascii=False, indent=2))
        return 2 if errors else 0
    except (ValueError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())


