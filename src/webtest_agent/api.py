"""Local-only task API with an isolated, single-worker execution queue."""
from __future__ import annotations

import io
import os
import zipfile
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import urlsplit

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, field_validator

from .models import MODEL_NAME, Scenario
from .service import execute_run, recent_events
from .storage import RunStore


class RunInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str = Field(default="custom", min_length=1, max_length=100)
    description: str = Field(min_length=10, max_length=20000)
    base_url: str
    max_steps: int = Field(default=35, ge=1, le=35)
    mode: str = "llm"
    strategy: str = "checkpoint"

    @field_validator("base_url")
    @classmethod
    def valid_url(cls, value: str) -> str:
        parsed = urlsplit(value)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password:
            raise ValueError("Use an http(s) URL without embedded credentials")
        return value

    @field_validator("mode")
    @classmethod
    def valid_mode(cls, value: str) -> str:
        if value not in {"llm", "demo"}:
            raise ValueError("mode must be llm or demo")
        return value

    @field_validator("strategy")
    @classmethod
    def valid_strategy(cls, value: str) -> str:
        if value not in {"baseline", "checkpoint"}:
            raise ValueError("strategy must be baseline or checkpoint")
        return value


def create_app(root: Path | None = None) -> FastAPI:
    load_dotenv(Path.cwd() / ".env", override=False, encoding="utf-8-sig")
    store = RunStore(root or Path(os.getenv("WEBTEST_ARTIFACTS", "artifacts")))

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # The MVP runs in one process. A restart cannot resume a browser session.
        for record in store.list_runs():
            if record.get("status") in {"queued", "running"}:
                store.set_status(record["run_id"], "failed", error="Service restarted; browser session cannot resume")
        app.state.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="webtest")
        yield
        app.state.executor.shutdown(wait=True, cancel_futures=False)

    app = FastAPI(title="Web Test Agent", version="0.1.0", lifespan=lifespan)
    app.state.store = store
    static = Path(__file__).parent / "static"
    app.mount("/static", StaticFiles(directory=static), name="static")

    @app.get("/")
    def index():
        return FileResponse(static / "index.html")

    @app.get("/api/health")
    def health():
        return {"status": "ok", "model": MODEL_NAME,
                "model_configured": bool(os.getenv("MODEL_BASE_URL") and os.getenv("MODEL_API_KEY"))}

    @app.post("/api/runs", status_code=202)
    def start_run(body: RunInput):
        if body.mode == "llm" and not (os.getenv("MODEL_BASE_URL") and os.getenv("MODEL_API_KEY")):
            raise HTTPException(422, "请先在服务端配置 MODEL_BASE_URL 和 MODEL_API_KEY")
        if body.mode == "demo" and (body.id != "local-demo" or urlsplit(body.base_url).hostname not in {"localhost", "127.0.0.1", "::1"}):
            raise HTTPException(422, "演示夹具仅支持 local-demo 和本机示例应用")
        scenario = Scenario(body.id, body.description, body.base_url, body.max_steps)
        run_id = store.create(scenario)
        app.state.executor.submit(execute_run, store, scenario, run_id, body.mode, body.strategy)
        return {"run_id": run_id, "status": "queued"}

    @app.get("/api/runs")
    def list_runs():
        return store.list_runs()

    def require_run(run_id: str) -> dict:
        result = store.get(run_id)
        if result is None:
            raise HTTPException(404, "Run not found")
        return result

    @app.get("/api/runs/{run_id}")
    def get_run(run_id: str):
        result = require_run(run_id)
        result["recent_events"] = recent_events(store.artifact_dir(run_id))
        directory = store.artifact_dir(run_id)
        result["files"] = sorted(str(p.relative_to(directory)).replace("\\", "/") for p in directory.rglob("*") if p.is_file())
        return result

    @app.get("/api/runs/{run_id}/files/{filename:path}")
    def artifact(run_id: str, filename: str):
        require_run(run_id)
        directory = store.artifact_dir(run_id).resolve()
        path = (directory / filename).resolve()
        if not path.is_relative_to(directory) or not path.is_file():
            raise HTTPException(404, "Artifact not found")
        # Raw HTML is downloaded, never executed in the workbench's origin.
        return FileResponse(path, filename=path.name if path.suffix not in {".png", ".jpg"} else None)

    @app.get("/api/runs/{run_id}/export")
    def export(run_id: str):
        result = require_run(run_id)
        if result.get("status") in {"queued", "running"}:
            raise HTTPException(409, "Wait for completion before exporting")
        directory = store.artifact_dir(run_id)
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
            for path in directory.rglob("*"):
                if path.is_file():
                    archive.write(path, path.relative_to(directory))
        buffer.seek(0)
        return StreamingResponse(buffer, media_type="application/zip",
                                 headers={"Content-Disposition": f'attachment; filename="{run_id}.zip"'})

    return app


