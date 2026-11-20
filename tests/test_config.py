import json
import os

import pytest
from fastapi.testclient import TestClient

from webtest_agent.api import create_app
from webtest_agent.cli import main


@pytest.fixture
def local_config(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    for name in ("MODEL_BASE_URL", "MODEL_API_KEY", "PYTHON_DOTENV_DISABLED"):
        # Record the original state so dotenv's direct writes are undone too.
        monkeypatch.setenv(name, "")
        monkeypatch.delenv(name)
    return tmp_path


def test_app_loads_bom_and_quoted_values(local_config):
    (local_config / ".env").write_text(
        'MODEL_BASE_URL="https://example.invalid/v1"\n'
        'MODEL_API_KEY="test-key#quoted"\n',
        encoding="utf-8-sig",
    )

    create_app(local_config / "artifacts")

    assert os.environ["MODEL_BASE_URL"] == "https://example.invalid/v1"
    assert os.environ["MODEL_API_KEY"] == "test-key#quoted"


def test_app_keeps_existing_shell_values(local_config, monkeypatch):
    (local_config / ".env").write_text(
        "MODEL_BASE_URL=https://file.invalid/v1\nMODEL_API_KEY=file-key\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("MODEL_BASE_URL", "https://shell.invalid/v1")
    monkeypatch.setenv("MODEL_API_KEY", "shell-key")

    create_app(local_config / "artifacts")

    assert os.environ["MODEL_BASE_URL"] == "https://shell.invalid/v1"
    assert os.environ["MODEL_API_KEY"] == "shell-key"


def test_missing_env_file_is_allowed(local_config):
    create_app(local_config / "artifacts")

    assert "MODEL_BASE_URL" not in os.environ
    assert "MODEL_API_KEY" not in os.environ


def test_app_loads_dotenv_and_reports_presence_without_values(local_config):
    (local_config / ".env").write_text(
        "MODEL_BASE_URL=https://example.invalid/v1\nMODEL_API_KEY=test-health-key\n",
        encoding="utf-8",
    )

    with TestClient(create_app(local_config / "artifacts")) as client:
        response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json()["model_configured"] is True
    assert "test-health-key" not in response.text
    assert "https://example.invalid/v1" not in response.text


def test_cli_run_loads_dotenv_before_execution(local_config, monkeypatch):
    (local_config / ".env").write_text(
        "MODEL_BASE_URL=https://example.invalid/v1\nMODEL_API_KEY=test-cli-key\n",
        encoding="utf-8",
    )
    scenario_file = local_config / "scenario.json"
    scenario_file.write_text(json.dumps({
        "id": "config-check",
        "description": "Check that the application is available.",
        "base_url": "http://127.0.0.1:18080",
    }), encoding="utf-8")
    observed = {}

    def execute_without_network(*args):
        observed["base_url"] = os.environ.get("MODEL_BASE_URL")
        observed["api_key"] = os.environ.get("MODEL_API_KEY")
        return {"verdict": "PASS", "steps": 0}

    monkeypatch.setattr("webtest_agent.service.execute_run", execute_without_network)

    result = main(["run", str(scenario_file), "--artifacts", str(local_config / "artifacts")])

    assert result == 0
    assert observed == {"base_url": "https://example.invalid/v1", "api_key": "test-cli-key"}
