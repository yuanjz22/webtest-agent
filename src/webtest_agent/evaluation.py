"""Offline human-reviewed scoring; labels never enter the runner."""
import json
from pathlib import Path


def evaluate_manifest(manifest: Path, reviews: Path) -> dict:
    rows = json.loads(manifest.read_text(encoding="utf-8"))
    labels = json.loads(reviews.read_text(encoding="utf-8"))
    results = []
    for row in rows:
        accepted = False
        review = labels.get(row["run_id"], {})
        path = manifest.parent / row["report"]
        reason = "missing report or independent review"
        if path.is_file():
            report = json.loads(path.read_text(encoding="utf-8"))
            accepted = (report.get("planner_kind") != "demo_fixture"
                        and report.get("verdict") in {"PASS", "FAIL"}
                        and report.get("verdict") == review.get("expected_verdict")
                        and type(report.get("steps")) is int and 0 <= report["steps"] <= 35
                        and review.get("evidence_sufficient") is True
                        and review.get("materials_complete") is True
                        and (path.parent / "trace.zip").is_file())
            reason = "accepted" if accepted else "verdict, evidence, materials or budget not accepted"
        results.append({"run_id": row["run_id"], "accepted": accepted, "reason": reason})
    return {"total": len(results), "accepted": sum(r["accepted"] for r in results),
            "acceptance_rate": sum(r["accepted"] for r in results) / len(results) if results else None,
            "results": results}
