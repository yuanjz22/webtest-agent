import sqlite3

import pytest

from webtest_agent.models import Scenario
from webtest_agent.storage import RunStore


def test_connection_commits_and_closes_after_use(tmp_path):
    store = RunStore(tmp_path)
    run_id = store.create(Scenario("test", "Check result", "http://localhost"))
    with store._connect() as connection:
        connection.execute("UPDATE runs SET status='running' WHERE id=?", (run_id,))
    assert store.get(run_id)["status"] == "running"
    with pytest.raises(sqlite3.ProgrammingError):
        connection.execute("SELECT 1")


def test_connection_rolls_back_and_closes_on_error(tmp_path):
    store = RunStore(tmp_path)
    run_id = store.create(Scenario("test", "Check result", "http://localhost"))
    with pytest.raises(RuntimeError):
        with store._connect() as connection:
            connection.execute("UPDATE runs SET status='running' WHERE id=?", (run_id,))
            raise RuntimeError("abort transaction")
    assert store.get(run_id)["status"] == "queued"
    with pytest.raises(sqlite3.ProgrammingError):
        connection.execute("SELECT 1")
