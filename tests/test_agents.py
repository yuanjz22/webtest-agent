"""Three-role boundaries and shared model transport, with all model HTTP mocked."""
import io
import json
from urllib.error import HTTPError

import pytest

from webtest_agent.actor import parse_action_decision
from webtest_agent.assertor import AssertorAgent, parse_assessment_result
from webtest_agent.llm import AgentError, ModelClient
from webtest_agent.models import Action, ActionOutcome, Checkpoint, MODEL_NAME, Observation, Scenario
from webtest_agent.planner import OrchestratorAgent, apply_stage_review, parse_plan, parse_task
from webtest_agent.service import execute_run
from webtest_agent.storage import RunStore


@pytest.fixture
def model_env(monkeypatch):
    monkeypatch.setenv("MODEL_BASE_URL", "https://model.invalid/v1")
    monkeypatch.setenv("MODEL_API_KEY", "unit-test-only-key")
    monkeypatch.setenv("MODEL_NAME", MODEL_NAME)


def response(output, usage=None):
    return io.BytesIO(json.dumps({
        "choices": [{"message": {"content": json.dumps(output)}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 3} if usage is None else usage,
    }).encode("utf-8"))


def test_service_assembles_three_distinct_roles_and_persists_usage(model_env, monkeypatch, tmp_path, browser_factory):
    replies = [
        {"stages": [[{"id": "C1", "description": "Check result"}]]},
        {"assessments": [], "reason": "Result not yet visible"},
        {"checkpoint_ids": ["C1"], "objective": "Create the result", "reason": "Advance C1", "finish": False},
        {"action": {"kind": "click", "target": "E1-button"}, "reason": "Click Create"},
        {"assessments": [{"checkpoint_id": "C1", "status": "PASS", "evidence_ids": ["E2"],
                          "quote": "done", "reason": "Result is now visible"}], "reason": "Verified C1"},
    ]
    requests = []

    def fake_urlopen(request, timeout):
        assert request.full_url == "https://model.invalid/v1/chat/completions"
        assert timeout == 60
        payload = json.loads(request.data)
        assert payload["model"] == MODEL_NAME
        requests.append(payload)
        return response(replies.pop(0))

    monkeypatch.setattr("webtest_agent.llm.urlopen", fake_urlopen)
    monkeypatch.setattr("webtest_agent.service.PlaywrightBrowser", lambda url, directory: browser_factory(directory))
    store = RunStore(tmp_path / "runs")
    scenario = Scenario("test", "Check result", "http://localhost", 2)
    run_id = store.create(scenario)
    report = execute_run(store, scenario, run_id)

    assert report["verdict"] == "PASS" and report["steps"] == 2
    assert report["agent_architecture"] == "three_role" and report["planner_kind"] == "llm"
    assert not replies and len(requests) == 5
    assert report["usage"]["requests"] == 5
    assert report["usage"]["input_tokens"] == 50 and report["usage"]["output_tokens"] == 15
    assert {role: counters["requests"] for role, counters in report["usage"]["by_role"].items()} == {
        "orchestrator": 2, "actor": 1, "assertor": 2,
    }
    # The actor receives an objective, while the assertor sees facts rather than actor reasoning.
    actor_input = json.loads(requests[3]["messages"][1]["content"])
    assert actor_input["task"]["checkpoint_ids"] == ["C1"]
    assert "ACTOR" in requests[3]["messages"][0]["content"]
    assertor_input = json.loads(requests[4]["messages"][1]["content"])
    assert "ASSERTOR" in requests[4]["messages"][0]["content"]
    assert assertor_input["previous_observation"]["text"] == "ready"
    assert assertor_input["observation"]["text"] == "done"
    assert assertor_input["last_action"]["success"] is True
    assert "task" not in assertor_input and "history" not in assertor_input
    assert "reason" not in assertor_input["last_action"]
    assert "screenshot" not in actor_input["observation"]
    assert "screenshot" not in assertor_input["observation"]
    assert store.get(run_id)["status"] == "completed"
    persisted = json.loads((store.artifact_dir(run_id) / "report.json").read_text(encoding="utf-8"))
    assert persisted["usage"] == report["usage"]
    assert "unit-test-only-key" not in json.dumps(persisted)
    assert (store.artifact_dir(run_id) / "events.jsonl").is_file()


def test_actor_schema_cannot_supply_verdicts():
    with pytest.raises(AgentError, match="assessments are not allowed"):
        parse_action_decision({"action": {"kind": "finish"}, "reason": "Done", "assessments": []})


def test_assertor_schema_cannot_supply_browser_actions():
    with pytest.raises(AgentError, match="actions are not allowed"):
        parse_assessment_result({"assessments": [], "reason": "Done", "action": {"kind": "click"}})


def test_assertor_retries_quote_using_real_page_evidence():
    class Client:
        def __init__(self):
            self.prompts = []

        def request(self, role, prompt, content):
            self.prompts.append(prompt)
            quote = "<relative time>\nDraft is ready." if len(self.prompts) == 1 else "Draft is ready."
            return {"assessments": [{"checkpoint_id": "C1", "status": "PASS",
                                     "evidence_ids": ["E1"], "quote": quote,
                                     "reason": "The card contains its expected description."}], "reason": "Observed"}

    page = Observation("E1", "http://localhost", "10 seconds ago\nDraft is ready.", [], "", "a",
                       [{"label": "10 seconds ago", "absolute": "21.11.2026 18:20"}])
    client = Client()
    result = AssertorAgent(client).assess(Scenario("detail", "Verify description.", page.url), page,
                                         [Checkpoint("C1", "Verify the card description")], None, None)
    assert len(client.prompts) == 2
    assert "not present in the captured page" in client.prompts[1]
    assert result.assessments[0]["quote"] == "Draft is ready."


@pytest.mark.parametrize("action_kind,value,confirmed_ids,status", [
    ("fill", "New item", [], "UNKNOWN"),
    ("press", "Enter", ["C1"], "FAIL"),
])
def test_assertor_returns_only_confirmed_failures(action_kind, value, confirmed_ids, status):
    class Client:
        def __init__(self):
            self.calls = []

        def request(self, role, prompt, content):
            self.calls.append(content)
            if len(self.calls) == 1:
                return {"reason": "Candidate", "assessments": [{
                    "checkpoint_id": "C1", "status": "FAIL", "evidence_ids": ["E2"],
                    "quote": "No items", "reason": "The current list has no item.",
                }]}
            return {"confirmed_failure_ids": confirmed_ids}

    client = Client()
    previous = Observation("E1", "http://localhost", "No items", [
        {"id": "E1-e1", "name": "Item title", "value": ""},
    ], "", "before")
    current = Observation("E2", "http://localhost", "No items", [], "", "after")
    assessment = AssertorAgent(client).assess(
        Scenario("t", "Enter and submit a new list item", "http://localhost"), current,
        [Checkpoint("C1", "Verify the submitted item appears")], previous,
        ActionOutcome(Action(action_kind, "E1-e1", value), True),
    )
    assert client.calls[1]["last_action"]["action"]["kind"] == action_kind
    assert client.calls[1]["proposed_failures"][0]["checkpoint_id"] == "C1"
    assert assessment.assessments[0]["status"] == status
    assert assessment.assessments[0]["evidence_ids"] == (["E2"] if confirmed_ids else [])


@pytest.mark.parametrize("status", [[], "PENDING", True])
def test_assertor_rejects_invalid_statuses(status):
    with pytest.raises(AgentError):
        parse_assessment_result({"assessments": [{"checkpoint_id": "C1", "status": status,
            "evidence_ids": [], "quote": "", "reason": ""}], "reason": "Check"})


@pytest.mark.parametrize("payload", [
    {"stages": []},
    {"stages": [[{"id": "C1", "description": "Check", "status": "PASS"}]]},
    {"stages": [[{"id": "C1", "description": "One"}], [{"id": "C1", "description": "Two"}]]},
])
def test_orchestrator_plan_rejects_empty_duplicate_or_prejudged_requirements(payload):
    with pytest.raises(AgentError):
        parse_plan(payload)


def test_plan_review_preserves_evidence_across_a_later_state_change():
    plan = {"stages": [[{"id": "on", "description": "Confirm both alerts are enabled."},
                        {"id": "badge", "description": "Confirm the banner says Enabled."},
                        {"id": "off", "description": "Disable the alerts and confirm both are off."}]]}

    class Client:
        def request(self, role, prompt, content):
            if "stages" not in content:
                return plan
            assert content["stages"] == plan["stages"]
            return {"stages": [["on", "badge"], ["off"]]}

    scenario = Scenario("alerts", "Enable both alerts and verify both are on and the banner says Enabled. "
                        "Then disable both and verify both are off.", "http://localhost")
    checks = OrchestratorAgent(Client()).plan(scenario)
    assert [(item.id, item.stage, item.phase) for item in checks] == [
        ("on", 0, "intermediate"), ("badge", 0, "intermediate"), ("off", 1, "final")]
    assert [item.description for item in checks] == [item["description"] for item in plan["stages"][0]]


@pytest.mark.parametrize("groups", [[['on', 'off']], [['off'], ['on']], [['on']], [['on'], ['on']]])
def test_stage_review_cannot_merge_reorder_or_drop_planned_requirements(groups):
    checks = [Checkpoint("on", "Enabled", phase="intermediate", stage=0),
              Checkpoint("off", "Disabled", phase="final", stage=1)]
    with pytest.raises(AgentError, match="stage review"):
        apply_stage_review({"stages": groups}, checks)


@pytest.mark.parametrize("changes", [
    {"checkpoint_ids": ["unknown"]}, {"finish": "false"}, {"checkpoint_ids": []},
    {"objective": ""}, {"action": {"kind": "click"}}, {"verdict": "PASS"},
])
def test_task_directive_requires_a_valid_scope_and_cannot_judge(changes):
    payload = {"checkpoint_ids": ["C1"], "objective": "Check result", "reason": "Next goal", "finish": False}
    payload.update(changes)
    with pytest.raises(AgentError):
        parse_task(payload, [Checkpoint("C1", "Check result")])


def test_shared_client_keeps_fixed_model_and_does_not_expose_local_provider_config(model_env, monkeypatch):
    captured = []
    def fake_urlopen(request, timeout):
        captured.append(request)
        return response({"ok": True})
    monkeypatch.setattr("webtest_agent.llm.urlopen", fake_urlopen)
    client = ModelClient()
    client.model = "cannot-change-the-fixed-request-model"
    assert client.request("actor", "system", {"task": "local test"}) == {"ok": True}
    payload = json.loads(captured[0].data)
    assert payload["model"] == MODEL_NAME
    assert captured[0].get_header("Authorization") == "Bearer unit-test-only-key"
    assert "unit-test-only-key" not in json.dumps(client.usage)


def test_unknown_token_usage_is_not_reported_as_zero(model_env, monkeypatch):
    outputs = [response({}, {}), response({})]
    monkeypatch.setattr("webtest_agent.llm.urlopen", lambda *args, **kwargs: outputs.pop(0))
    client = ModelClient()
    client.request("actor", "system", {})
    client.request("assertor", "system", {})
    assert client.usage["requests"] == 2
    assert client.usage["input_tokens"] is None and client.usage["output_tokens"] is None
    assert client.usage["by_role"]["actor"]["input_tokens"] is None
    assert client.usage["by_role"]["assertor"]["input_tokens"] == 10


def test_http_failure_records_role_usage_without_echoing_provider_body(model_env, monkeypatch):
    def fail(*args, **kwargs):
        raise HTTPError("https://model.invalid", 401, "unauthorized", {}, io.BytesIO(b"unit-test-only-key"))
    monkeypatch.setattr("webtest_agent.llm.urlopen", fail)
    client = ModelClient()
    with pytest.raises(AgentError, match="HTTP 401") as caught:
        client.request("orchestrator", "system", {})
    assert "unit-test-only-key" not in str(caught.value)
    assert client.usage["requests"] == client.usage["by_role"]["orchestrator"]["requests"] == 1
    assert client.usage["input_tokens"] is None


@pytest.mark.parametrize("body", [b"not json", b'[]', b'{"choices":[]}',
    json.dumps({"choices": [{"message": {"content": '```json\n{}\n```'}}]}).encode("utf-8"),
    json.dumps({"choices": [{"message": {"content": '[]'}}]}).encode("utf-8"),
])
def test_transport_rejects_malformed_or_non_object_output(model_env, monkeypatch, body):
    monkeypatch.setattr("webtest_agent.llm.urlopen", lambda *args, **kwargs: io.BytesIO(body))
    with pytest.raises(AgentError):
        ModelClient().request("assertor", "system", {})
