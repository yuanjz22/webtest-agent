"""SQLite task index plus per-run, local evidence directories."""
from __future__ import annotations

import json
import re
import sqlite3
import uuid
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from .models import RunReport, Scenario


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class RunStore:
    def __init__(self, root: Path) -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.db_path = self.root / "runs.sqlite3"
        with self._connect() as db:
            db.execute("""CREATE TABLE IF NOT EXISTS runs (
                id TEXT PRIMARY KEY, scenario_id TEXT NOT NULL, scenario_json TEXT NOT NULL,
                status TEXT NOT NULL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
                verdict TEXT, error TEXT, report_json TEXT)""")

    @contextmanager
    def _connect(self):
        connection = sqlite3.connect(str(self.db_path), timeout=15)
        connection.row_factory = sqlite3.Row
        try:
            # sqlite3's own context manager commits/rolls back, but does not close.
            with connection:
                yield connection
        finally:
            connection.close()

    def artifact_dir(self, run_id: str) -> Path:
        if not isinstance(run_id, str) or not re.fullmatch(r"[a-f0-9]{32}", run_id):
            raise ValueError("invalid run id")
        return self.root / "artifacts" / run_id

    def create(self, scenario: Scenario) -> str:
        run_id = uuid.uuid4().hex
        timestamp = _now()
        with self._connect() as db:
            db.execute("INSERT INTO runs (id,scenario_id,scenario_json,status,created_at,updated_at) VALUES (?,?,?,?,?,?)",
                       (run_id, scenario.id, json.dumps(asdict(scenario), ensure_ascii=False), "queued", timestamp, timestamp))
        self.artifact_dir(run_id).mkdir(parents=True, exist_ok=False)
        return run_id

    def set_status(self, run_id: str, status: str, error=None) -> None:
        self.artifact_dir(run_id)
        if status not in {"queued", "running", "completed", "failed"}:
            raise ValueError("invalid run status")
        with self._connect() as db:
            changed = db.execute("UPDATE runs SET status=?,error=?,updated_at=? WHERE id=?",
                                 (status, error, _now(), run_id)).rowcount
        if not changed:
            raise KeyError(run_id)

    def save(self, report: RunReport) -> None:
        directory = self.artifact_dir(report.run_id)
        if self.get(report.run_id) is None:
            raise KeyError(report.run_id)
        directory.mkdir(parents=True, exist_ok=True)
        data = json.dumps(asdict(report), ensure_ascii=False, indent=2)
        temporary = directory / "report.json.tmp"
        temporary.write_text(data, encoding="utf-8")
        temporary.replace(directory / "report.json")
        with self._connect() as db:
            db.execute("UPDATE runs SET status='completed',verdict=?,report_json=?,error=NULL,updated_at=? WHERE id=?",
                       (report.verdict, data, _now(), report.run_id))

    @staticmethod
    def _decode(row, include_report=True):
        result = dict(row)
        result["run_id"] = result["id"]
        result["scenario"] = json.loads(result.pop("scenario_json"))
        report_json = result.pop("report_json", None)
        if include_report:
            result["report"] = json.loads(report_json) if report_json else None
        return result

    def list_runs(self) -> list[dict]:
        with self._connect() as db:
            rows = db.execute("SELECT id,scenario_id,scenario_json,status,created_at,updated_at,verdict,error FROM runs ORDER BY created_at DESC").fetchall()
        return [self._decode(row, False) for row in rows]

    def get(self, run_id: str) -> dict | None:
        self.artifact_dir(run_id)
        with self._connect() as db:
            row = db.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
        return self._decode(row) if row else None
