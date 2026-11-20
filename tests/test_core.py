from copy import deepcopy

import pytest

from webtest_agent.models import (Action, ActionDecision, AssessmentResult, Checkpoint,
                                  Scenario, TaskDirective)
from webtest_agent.planner import parse_plan
from webtest_agent.runner import run_scenario


class Orchestrator:
    def __init__(self):
        self.checkpoints = [Checkpoint("C1", "Check result")]
        self.calls = []
        self.finish = False

    def plan(self, scenario):
        return self.checkpoints

    def select_task(self, scenario, observation, checkpoints, history, remaining):
        self.calls.append((deepcopy(checkpoints), remaining))
        return TaskDirective(["C1"], "Check result", "Select next objective", self.finish)


class Actor:
    def __init__(self):
        self.calls = []
        self.finish = False
        self.wait = False
        self.repeat = False

    def decide(self, scenario, observation, task, history, remaining):
        self.calls.append((observation.id, task, remaining))
        if self.finish:
            return ActionDecision(Action("finish"), "I think it is done")
        if self.wait:
            value = "1" if self.repeat else str(len(self.calls))
            return ActionDecision(Action("wait", value=value))
        return ActionDecision(Action("click", observation.elements[0]["id"]), "Create the result")


class Assertor:
    def __init__(self):
        self.calls = []
        self.pass_when_done = False
        self.bad_quote = False
        self.stale_evidence = False

    def assess(self, scenario, observation, checkpoints, previous, last_action):
        self.calls.append((observation, previous, last_action))
        if self.pass_when_done and observation.text == "done":
            return AssessmentResult([{
                "checkpoint_id": "C1", "status": "PASS",
                "evidence_ids": ["old" if self.stale_evidence else observation.id],
                "quote": "invented" if self.bad_quote else "done", "reason": "Result is visible",
            }])
        return AssessmentResult()


@pytest.fixture
def roles():
    return Orchestrator(), Actor(), Assertor()


def run(roles, browser, max_steps=2, **kwargs):
    orchestrator, actor, assertor = roles
    return run_scenario(Scenario("test", "Check result", "http://localhost", max_steps),
                        orchestrator, browser, "test", actor=actor, assertor=assertor, **kwargs)


def test_budget_is_charged_before_failed_action(roles, fake_browser):
    fake_browser.fail_actions.add("navigate")
    report = run(roles, fake_browser, max_steps=1)
    assert report.steps == len(fake_browser.actions) == 1
    assert report.verdict == "INCONCLUSIVE"
    assert not roles[0].calls and not roles[1].calls
    assert len(roles[2].calls) == 1
    assert roles[2].calls[0][2].success is False


def test_empty_plan_cannot_pass(roles, fake_browser):
    roles[0].checkpoints = []
    report = run(roles, fake_browser)
    assert report.verdict == "INCONCLUSIVE"
    assert report.steps == 0
    assert not fake_browser.actions


def test_stage_progress_preserves_earlier_evidence_and_rejects_premature_assessments(roles, fake_browser):
    orchestrator, actor, assertor = roles
    orchestrator.checkpoints = parse_plan({"stages": [
        [{"id": "before", "description": "Observe enabled"}],
        [{"id": "after", "description": "Turn off and observe disabled"}],
    ]})
    fake_browser.text = "enabled"
    original_execute = fake_browser.execute

    def execute(action):
        original_execute(action)
        if action.kind == "click":
            fake_browser.text = "disabled"

    def assess(scenario, observation, checks, previous, last_action):
        expected_id, = [item.id for item in checks]
        if expected_id == "after" and observation.text == "enabled":
            return AssessmentResult()
        assert expected_id == ("before" if observation.text == "enabled" else "after")
        accepted = {"checkpoint_id": expected_id, "status": "PASS",
                    "evidence_ids": [observation.id], "quote": observation.text, "reason": "Observed state"}
        # An earlier model answer must not approve a check reserved for the next stage.
        if expected_id == "before":
            return AssessmentResult([accepted, {**accepted, "checkpoint_id": "after", "status": "FAIL"}])
        return AssessmentResult([accepted])

    def select_task(scenario, observation, checks, history, remaining):
        assert [item.id for item in checks] == ["after"]
        return TaskDirective(["after"], "Turn off")

    fake_browser.execute = execute
    assertor.assess = assess
    orchestrator.select_task = select_task
    report = run(roles, fake_browser)
    assert report.verdict == "PASS" and report.steps == 2
    assert [check.evidence_ids for check in report.checkpoints] == [["E1"], ["E2"]]


def test_stage_transition_verifies_current_snapshot_before_actor_can_change_it(roles, fake_browser):
    orchestrator, actor, assertor = roles
    orchestrator.checkpoints = parse_plan({"stages": [
        [{"id": "opened", "description": "Open the confirmation page"}],
        [{"id": "preview", "description": "Check the displayed total before confirming"}],
        [{"id": "saved", "description": "Verify the confirmation receipt"}],
    ]})
    fake_browser.text = "Open confirmation dialog"
    execute = fake_browser.execute
    states = iter(("Confirmation dialog: Total $42", "Receipt created"))

    def change_page(action):
        execute(action)
        if action.kind == "click":
            fake_browser.text = next(states)

    def assess(scenario, observation, checks, previous, last_action):
        checkpoint, = checks
        visible = {"opened": "Confirmation dialog", "preview": "Total $42", "saved": "Receipt created"}
        quote = visible[checkpoint.id]
        if quote not in observation.text:
            return AssessmentResult()
        return AssessmentResult([{"checkpoint_id": checkpoint.id, "status": "PASS",
                                  "evidence_ids": [observation.id], "quote": quote,
                                  "reason": "Required state is visible"}])

    scheduled = []

    def select_task(scenario, observation, checks, history, remaining):
        checkpoint, = checks
        scheduled.append(checkpoint.id)
        return TaskDirective([checkpoint.id], "Continue the confirmation workflow")

    fake_browser.execute = change_page
    assertor.assess = assess
    orchestrator.select_task = select_task
    report = run(roles, fake_browser, max_steps=3)
    assert report.verdict == "PASS" and report.steps == 3
    assert scheduled == ["opened", "saved"]
    assert [c.evidence_ids for c in report.checkpoints] == [["E2"], ["E2"], ["E3"]]


def test_last_allowed_action_is_verified_with_before_after_evidence(roles, fake_browser):
    roles[2].pass_when_done = True
    report = run(roles, fake_browser)
    assert report.verdict == "PASS"
    assert report.steps == len(fake_browser.actions) == 2
    assert len(roles[0].calls) == len(roles[1].calls) == 1
    current, previous, outcome = roles[2].calls[-1]
    assert current.text == "done" and previous.text == "ready"
    assert outcome.action.kind == "click" and outcome.success is True
    calls = [(event["role"], event["operation"]) for event in report.events if event["kind"] == "agent_started"]
    assert calls == [("orchestrator", "plan"), ("assertor", "assess"),
                     ("orchestrator", "select_task"), ("actor", "decide"), ("assertor", "assess")]


def test_action_history_keeps_original_target_after_element_positions_change(roles, fake_browser):
    observe = fake_browser.observe

    def snapshot():
        observation = observe()
        if observation.text == "done":
            observation.elements[0]["name"] = "Delete"
        return observation

    fake_browser.observe = snapshot
    original_decide = roles[1].decide

    def decide(scenario, observation, task, history, remaining):
        if observation.text == "done":
            event = next(e for e in reversed(history) if e["kind"] == "action")
            assert event["source_observation_id"] == "E1"
            assert event["target_element"] == {"name": "Create"}
            assert event["action"]["target"] != observation.elements[0]["id"]
            assert observation.elements[0]["name"] == "Delete"
            return ActionDecision(Action("finish"))
        return original_decide(scenario, observation, task, history, remaining)

    roles[1].decide = decide
    report = run(roles, fake_browser, max_steps=3)
    assert report.steps == 2
    assert report.reason.startswith("Actor stopped")


@pytest.mark.parametrize("invalid", ["bad_quote", "stale_evidence", "missing_screenshot"])
def test_assertor_pass_requires_valid_current_evidence(roles, fake_browser, invalid):
    roles[2].pass_when_done = True
    setattr(fake_browser if invalid == "missing_screenshot" else roles[2], invalid, True)
    report = run(roles, fake_browser)
    assert report.verdict == "INCONCLUSIVE"
    assert report.checkpoints[0].status == "PENDING"
    assert any(event["kind"] == "assessments" and not event["results"][0]["accepted"] for event in report.events)


@pytest.mark.parametrize("role_index", [0, 1])
def test_stop_claims_cannot_mark_unverified_checks_passed(roles, fake_browser, role_index):
    roles[role_index].finish = True
    report = run(roles, fake_browser)
    assert report.verdict == "INCONCLUSIVE"
    assert report.steps == 1
    assert report.checkpoints[0].status == "PENDING"


@pytest.mark.parametrize("index,method,role", [
    (0, "plan", "orchestrator"), (0, "select_task", "orchestrator"),
    (1, "decide", "actor"), (2, "assess", "assertor"),
])
def test_agent_errors_are_attributed_without_exposing_exception_text(roles, fake_browser, index, method, role):
    def fail(*args):
        raise RuntimeError("private-provider-message")
    setattr(roles[index], method, fail)
    report = run(roles, fake_browser)
    assert report.verdict == "INCONCLUSIVE"
    assert any(event["kind"] == "agent_error" and event["role"] == role for event in report.events)
    assert "private-provider-message" not in repr(report)


def test_orchestrator_cannot_schedule_unknown_checkpoint(roles, fake_browser):
    roles[0].select_task = lambda *args: TaskDirective(["invented"], "Wrong objective")
    report = run(roles, fake_browser)
    assert report.verdict == "INCONCLUSIVE"
    assert report.steps == 1 and not roles[1].calls


def test_repeated_action_stops_before_third_identical_attempt(roles, fake_browser):
    roles[1].wait = roles[1].repeat = True
    report = run(roles, fake_browser, max_steps=10)
    assert report.verdict == "INCONCLUSIVE"
    assert report.steps == len(fake_browser.actions) == 3
    assert "Repeated action" in report.reason


def test_no_progress_stops_after_final_independent_assessment(roles, fake_browser):
    roles[1].wait = True
    report = run(roles, fake_browser, max_steps=10)
    assert report.verdict == "INCONCLUSIVE"
    assert report.steps == len(fake_browser.actions) == 4
    assert len(roles[2].calls) == 4
    assert "no observable progress" in report.reason


def test_three_tool_errors_remain_inconclusive(roles, fake_browser):
    fake_browser.fail_actions.add("click")
    report = run(roles, fake_browser, max_steps=10)
    assert report.verdict == "INCONCLUSIVE"
    assert report.steps == 4
    assert "Three consecutive browser errors" in report.reason
    assert roles[2].calls[-1][2].success is False


def test_progress_callback_cannot_break_execution(roles, fake_browser):
    roles[2].pass_when_done = True
    def broken_callback(event):
        raise RuntimeError("UI unavailable")
    report = run(roles, fake_browser, on_event=broken_callback)
    assert report.verdict == "PASS"
    assert any(event["kind"] == "progress_callback_error" for event in report.events)


@pytest.mark.parametrize("strategy,expected", [("baseline", "PENDING"), ("checkpoint", "PASS")])
def test_strategy_controls_accumulated_feedback(roles, fake_browser, strategy, expected):
    roles[0].checkpoints.append(Checkpoint("C2", "Another requirement"))
    roles[1].finish = True
    def assess(scenario, observation, *args):
        return AssessmentResult([{"checkpoint_id": "C1", "status": "PASS", "evidence_ids": [observation.id],
                                  "quote": "ready", "reason": "Initial condition observed"}])
    roles[2].assess = assess
    report = run(roles, fake_browser, strategy=strategy)
    assert roles[0].calls[0][0][0].status == expected
    assert report.verdict == "INCONCLUSIVE" and report.checkpoints[0].status == "PASS"


def test_plan_cannot_preload_a_pass(roles, fake_browser):
    roles[0].checkpoints[0].status = "PASS"
    roles[0].checkpoints[0].evidence_ids = ["invented"]
    roles[1].finish = True
    report = run(roles, fake_browser)
    assert report.verdict == "INCONCLUSIVE"
    assert report.checkpoints[0].status == "PENDING" and not report.checkpoints[0].evidence_ids


def test_observation_error_is_not_an_application_failure(roles, fake_browser):
    fake_browser.break_observation = True
    report = run(roles, fake_browser)
    assert report.verdict == "INCONCLUSIVE" and report.steps == 1
    assert not roles[1].calls and not roles[2].calls


@pytest.mark.parametrize("first_status,confirmed_status", [("FAIL", "PASS"), ("PASS", "FAIL")])
def test_terminal_verdict_is_rechecked_if_page_changes_during_model_call(
        roles, fake_browser, first_status, confirmed_status):
    calls = []

    def assess(scenario, observation, *args):
        calls.append(observation.id)
        if len(calls) == 1:
            # An async application update finishes while the model evaluates the old page.
            fake_browser.text = "done"
        return AssessmentResult([{
            "checkpoint_id": "C1", "status": first_status if len(calls) == 1 else confirmed_status,
            "evidence_ids": [observation.id], "quote": observation.text, "reason": "Observed state evaluated",
        }])

    roles[2].assess = assess
    report = run(roles, fake_browser, max_steps=1)
    assert report.verdict == confirmed_status
    assert report.checkpoints[0].quote == "done"
    assert report.steps == len(fake_browser.actions) == 1
    assert len(calls) == 2
    assert not roles[1].calls
    assert any(event["kind"] == "verdict_recheck" for event in report.events)
    assert any(event["kind"] == "verdict_confirmed" for event in report.events)


def test_unavailable_confirmation_cannot_publish_a_candidate_fail(roles, fake_browser):
    def assess(scenario, observation, *args):
        fake_browser.break_observation = True
        return AssessmentResult([{"checkpoint_id": "C1", "status": "FAIL", "evidence_ids": [observation.id],
                                  "quote": "ready", "reason": "Proposed failure"}])
    roles[2].assess = assess
    report = run(roles, fake_browser, max_steps=1)
    assert report.verdict == "INCONCLUSIVE"
    assert report.checkpoints[0].status == "PENDING"
    assert "could not be confirmed" in report.reason


def test_stable_failure_is_not_silently_changed_to_pass(roles, fake_browser):
    def assess(scenario, observation, *args):
        return AssessmentResult([{"checkpoint_id": "C1", "status": "FAIL", "evidence_ids": [observation.id],
                                  "quote": "ready", "reason": "Stable contradicting evidence"}])
    roles[2].assess = assess
    report = run(roles, fake_browser, max_steps=1)
    assert report.verdict == "FAIL"
    assert fake_browser.observation_count == 2
    assert report.steps == 1


def test_terminal_rechecking_is_bounded_without_exceeding_action_budget(roles, fake_browser):
    calls = []
    def assess(scenario, observation, *args):
        calls.append(observation.id)
        fake_browser.text = "changed-" + str(len(calls))
        return AssessmentResult([{"checkpoint_id": "C1", "status": "FAIL", "evidence_ids": [observation.id],
                                  "quote": observation.text, "reason": "Stale failure"}])
    roles[2].assess = assess
    report = run(roles, fake_browser, max_steps=1)
    assert report.verdict == "INCONCLUSIVE"
    assert 1 < len(calls) <= 3
    assert report.steps == len(fake_browser.actions) == 1
    assert report.checkpoints[0].status == "PENDING"
