"""Local browser doubles; no model endpoints or browser processes are used here."""
from pathlib import Path

import pytest

from webtest_agent.models import Observation


class FakeBrowser:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.actions = []
        self.text = "ready"
        self.observation_count = 0
        self.fail_actions = set()
        self.stay_same = False
        self.missing_screenshot = False
        self.break_observation = False

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def execute(self, action):
        self.actions.append(action)
        if action.kind in self.fail_actions:
            raise RuntimeError("synthetic tool failure")
        if action.kind == "click" and not self.stay_same:
            self.text = "done"

    def observe(self):
        if self.break_observation:
            raise RuntimeError("synthetic observation failure")
        self.observation_count += 1
        evidence_id = "E" + str(self.observation_count)
        screenshot = self.directory / (evidence_id + ".png")
        if not self.missing_screenshot:
            screenshot.write_bytes(b"fake image for existence checks, not real browser evidence")
        return Observation(evidence_id, "http://localhost", self.text,
                           [{"id": evidence_id + "-button", "name": "Create"}],
                           "" if self.missing_screenshot else str(screenshot), self.text)


@pytest.fixture
def browser_factory():
    return FakeBrowser


@pytest.fixture
def fake_browser(tmp_path, browser_factory):
    return browser_factory(tmp_path / "evidence")
