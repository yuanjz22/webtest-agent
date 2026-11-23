"""Serial orchestration of planner, actor and assertor with one shared action budget."""
from __future__ import annotations

import time
from copy import deepcopy
from dataclasses import asdict
from typing import Callable

from .llm import AgentError
from .models import (Action, ActionDecision, ActionOutcome, AssessmentResult, Checkpoint,
                     MODEL_NAME, RunReport, Scenario, TaskDirective)
from .verifier import (apply_assessments, invalidate_changed_page, terminal_evidence_still_current,
                       verdict_from_checkpoints)


MAX_VERDICT_REFRESHES = 3


def run_scenario(scenario: Scenario, orchestrator, browser, run_id: str, *, actor, assertor,
                 strategy: str = "checkpoint", on_event: Callable[[dict], None] | None = None) -> RunReport:
    """Run three roles against an already-open browser; the caller finalizes its trace.

    Only perform() executes browser actions and charges BEFORE each attempt.
    Agent calls do not consume browser steps; their costs are recorded separately.
    """
    if strategy not in {"baseline", "checkpoint"}:
        raise ValueError("strategy must be baseline or checkpoint")
    started = time.monotonic()
    events, observations, checkpoints = [], [], []
    steps = consecutive_errors = no_progress = 0
    duplicate_attempts = {}
    previous_observation = last_action = None
    verdict_refreshes = 0

    def emit(kind: str, **payload) -> None:
        event = {"index": len(events), "kind": kind, "step": steps,
                 "elapsed_seconds": round(time.monotonic() - started, 3), **payload}
        events.append(event)
        if on_event:
            try:
                on_event(event)
            except Exception:
                events.append({"index": len(events), "kind": "progress_callback_error", "step": steps})

    def agent_call(role, operation, function, *args, expected_type):
        emit("agent_started", role=role, operation=operation)
        try:
            result = function(*args)
            if not isinstance(result, expected_type):
                raise TypeError("Unexpected agent result type")
        except Exception as exc:
            # AgentError messages are locally generated diagnostics, never raw model/HTTP bodies.
            emit("agent_error", role=role, operation=operation, error_type=type(exc).__name__,
                 detail=str(exc) if type(exc) is AgentError else "role execution failed")
            raise
        emit("agent_completed", role=role, operation=operation)
        return result

    def report(reason: str = "") -> RunReport:
        verdict, derived_reason = verdict_from_checkpoints(checkpoints)
        if verdict == "INCONCLUSIVE" and reason:
            derived_reason = reason
        emit("finished", verdict=verdict, reason=derived_reason)
        return RunReport(
            run_id=run_id, scenario_id=scenario.id, verdict=verdict, reason=derived_reason,
            steps=steps, checkpoints=checkpoints, observations=observations, events=events,
            usage=deepcopy(getattr(orchestrator, "usage", {
                "requests": 0, "input_tokens": None, "output_tokens": None})),
            duration_seconds=round(time.monotonic() - started, 3), strategy=strategy,
            model=getattr(orchestrator, "model", MODEL_NAME),
            planner_kind=getattr(orchestrator, "kind", "custom"), scenario=asdict(scenario),
            agent_architecture=getattr(orchestrator, "agent_architecture", "three_role"),
        )

    def active_stage(items):
        stage = min(item.stage for item in items if item.status != "PASS")
        return [item for item in items if item.stage == stage]

    def feedback(items):
        # The baseline shares all three roles and evidence guards, but hides accumulated checkpoint state.
        if strategy == "checkpoint":
            return deepcopy(items)
        return [Checkpoint(c.id, c.description, phase=c.phase, stage=c.stage) for c in items]

    def perform(action: Action, source=None) -> None:
        nonlocal steps, consecutive_errors, last_action
        if action.kind == "finish" or steps >= scenario.max_steps:
            raise RuntimeError("Browser action outside the shared budget")
        steps += 1
        evidence = {}
        if source:
            by_id = {element["id"]: element for element in source.elements if "id" in element}
            evidence["source_observation_id"] = source.id
            targets = {"target_element": action.target}
            if action.kind == "drag":
                targets["destination_element"] = action.value
            for event_key, handle in targets.items():
                element = by_id.get(handle)
                if element is not None:
                    evidence[event_key] = {key: element[key] for key in ("name", "role", "dom_id", "test_id", "facts") if element.get(key)}
        emit("action", action=asdict(action), **evidence)
        try:
            browser.execute(action)
        except Exception as exc:
            consecutive_errors += 1
            error_kind = getattr(exc, "kind", "execution")
            last_action = ActionOutcome(action, False, error_kind)
            emit("action_error", error_kind=error_kind, error_type=type(exc).__name__)
            if strategy == "checkpoint":
                guidance = {
                    "timeout": "Inspect the fresh observation before deciding whether another action is needed.",
                    "locator": "Choose a target from fresh element ids; do not reuse the failed locator.",
                    "navigation": "Check the configured origin and observed page before deciding to retry.",
                    "unsupported": "Choose an available browser action and current observed target.",
                }
                emit("recovery", error_kind=error_kind,
                     guidance=guidance.get(error_kind, "Inspect the fresh page and adjust the objective."))
            return
        last_action = ActionOutcome(action, True)
        consecutive_errors = 0

    def observe():
        try:
            observation = browser.observe()
            observations.append(observation)
            invalidated = invalidate_changed_page(checkpoints, observation)
            emit("observation", observation_id=observation.id, url=observation.url,
                 state_hash=observation.state_hash, invalidated=invalidated)
            return observation
        except Exception as exc:
            emit("observation_error", error_type=type(exc).__name__)
            return None

    emit("started", strategy=strategy, planner_kind=getattr(orchestrator, "kind", "custom"),
         agent_architecture=getattr(orchestrator, "agent_architecture", "three_role"), max_steps=scenario.max_steps)
    try:
        checkpoints = agent_call("orchestrator", "plan", orchestrator.plan, scenario, expected_type=list)
        if (not 1 <= len(checkpoints) <= 30 or not all(isinstance(c, Checkpoint) for c in checkpoints)
                or len({c.id for c in checkpoints}) != len(checkpoints)):
            raise ValueError("plan requires distinct checkpoints")
        for checkpoint in checkpoints:
            checkpoint.status = "PENDING"
            checkpoint.evidence_ids = []
            checkpoint.state_hash = checkpoint.scope_url = checkpoint.quote = checkpoint.reason = ""
        emit("planned", checkpoints=[{"id": c.id, "description": c.description,
                                      "stage": c.stage, "phase": c.phase} for c in checkpoints])
    except Exception as exc:
        checkpoints = []
        emit("planning_error", error_type=type(exc).__name__)
        return report("Scenario planning failed; no application verdict was inferred.")

    perform(Action("navigate", value=scenario.base_url))
    observation = observe()
    if observation is None:
        return report("No reliable initial browser observation is available.")

    # Each new stage can be assessed on the current snapshot before spending another action.
    for _ in range(scenario.max_steps + len(checkpoints) + MAX_VERDICT_REFRESHES):
        current = active_stage(checkpoints)
        try:
            assessment = agent_call(
                "assertor", "assess", assertor.assess, scenario, observation, feedback(current),
                previous_observation, last_action, expected_type=AssessmentResult,
            )
            candidate = deepcopy(checkpoints)
            outcomes = apply_assessments(active_stage(candidate), assessment.assessments, observation)
        except Exception as exc:
            emit("verification_error", error_type=type(exc).__name__)
            return report("Assertor verification failed; no application failure was inferred.")
        verdict, _ = verdict_from_checkpoints(candidate)
        if verdict != "INCONCLUSIVE":
            # The application can finish an async update while the model evaluates an old snapshot.
            # Confirm with a read-only observation before committing a terminal conclusion.
            fresh = observe()
            if fresh is None:
                return report("Final verdict could not be confirmed against a fresh page observation.")
            current_quotes = [c.quote for c in candidate if c.scope_url == observation.url and c.quote]
            if not terminal_evidence_still_current(observation, fresh, scenario.description, current_quotes):
                verdict_refreshes += 1
                emit("verdict_recheck", previous_observation_id=observation.id,
                     observation_id=fresh.id, proposed_verdict=verdict,
                     evidence_quotes=current_quotes,
                     reason="Page changed while the assertor was evaluating; discard the stale proposal.")
                observation = fresh
                if verdict_refreshes >= MAX_VERDICT_REFRESHES:
                    return report("Page kept changing during final verification; no stable verdict could be confirmed.")
                continue
            emit("verdict_confirmed", observation_id=observation.id,
                 confirmation_observation_id=fresh.id, verdict=verdict,
                 same_snapshot=fresh.state_hash == observation.state_hash)
        checkpoints = candidate
        if outcomes:
            emit("assessments", role="assertor", stage=current[0].stage, observation_id=observation.id, results=outcomes,
                 assessments=assessment.assessments, reason=assessment.reason)
        if verdict != "INCONCLUSIVE":
            return report()

        if active_stage(checkpoints)[0].stage != current[0].stage:
            continue

        remaining = scenario.max_steps - steps
        if remaining == 0:
            emit("budget_exhausted")
            return report("Browser action budget exhausted; final observation did not verify every requirement.")
        if consecutive_errors >= 3:
            return report("Three consecutive browser errors prevented reliable verification.")
        if no_progress >= 3:
            emit("no_progress", reason="three browser actions left the page state unchanged")
            return report("Three actions made no observable progress; stopped after final assertion check.")

        history = [e for e in events if e["kind"] in {
            "action", "action_error", "recovery", "no_progress", "task_selected"
        } or (strategy == "checkpoint" and e["kind"] == "unchanged_after_action")]
        current = active_stage(checkpoints)
        try:
            task = agent_call("orchestrator", "select_task", orchestrator.select_task,
                              scenario, observation, feedback(current), history, remaining, expected_type=TaskDirective)
            if not set(task.checkpoint_ids).issubset({c.id for c in current}):
                raise ValueError("task references a checkpoint outside the active stage")
        except Exception as exc:
            emit("orchestration_error", error_type=type(exc).__name__)
            return report("Task scheduling failed; no application failure was inferred.")
        emit("task_selected", task=asdict(task), observation_id=observation.id, stage=current[0].stage)
        if task.finish:
            return report("Orchestrator stopped before all acceptance requirements were reliably verified.")

        try:
            decision = agent_call("actor", "decide", actor.decide, scenario, observation, task,
                                  history, remaining, expected_type=ActionDecision)
            if not isinstance(decision.action, Action):
                raise TypeError("actor must return an Action")
        except Exception as exc:
            emit("decision_error", role="actor", error_type=type(exc).__name__)
            return report("Actor decision failed; no application failure was inferred.")
        emit("actor_decision", reason=decision.reason, action=asdict(decision.action))
        if decision.action.kind == "finish":
            return report("Actor stopped before all acceptance requirements were reliably verified.")

        # IDs change between observations even when the state and element order are identical.
        indices = {element["id"]: index for index, element in enumerate(observation.elements) if "id" in element}
        action = decision.action
        target = indices.get(action.target, action.target)
        value = indices.get(action.value, action.value) if action.kind == "drag" else action.value
        key = (current[0].stage, observation.state_hash, action.kind, target, value)
        if duplicate_attempts.get(key, 0) >= 2:
            emit("no_progress", reason="same action repeated against the same observed state")
            return report("Repeated action made no observable progress; stopped for review.")
        previous_observation = observation
        perform(decision.action, observation)
        if last_action.success:
            duplicate_attempts[key] = duplicate_attempts.get(key, 0) + 1
        observation = observe()
        if observation is None:
            return report("Observation failed after an action; retained evidence is insufficient for a complete verdict.")
        unchanged = observation.state_hash == previous_observation.state_hash
        if unchanged and last_action.success:
            emit("unchanged_after_action", action_step=steps, observation_id=observation.id)
        no_progress = no_progress + 1 if unchanged else 0
    return report("Decision budget exhausted; no further browser actions were attempted.")
