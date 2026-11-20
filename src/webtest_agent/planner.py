"""Orchestrator agent: extract requirements and select the current testing objective."""
from __future__ import annotations

from dataclasses import asdict, replace

from .llm import (ACTION_HISTORY_RULE, SCENARIO_DATA_RULE, UNTRUSTED_DATA_RULE,
                  AgentError, ModelClient, json_object, observation_content)
from .models import Checkpoint, MODEL_NAME, Observation, Scenario, TaskDirective


def parse_plan(payload: dict) -> list[Checkpoint]:
    json_object(payload, "plan")
    if set(payload) != {"stages"}:
        raise AgentError("plan requires only stages")
    stages = payload["stages"]
    if not isinstance(stages, list) or not 1 <= len(stages) <= 30:
        raise AgentError("plan must contain between 1 and 30 stages")
    checkpoints, seen = [], set()
    for index, group in enumerate(stages):
        if not isinstance(group, list) or not group or len(checkpoints) + len(group) > 30:
            raise AgentError("each stage requires checks and the total cannot exceed 30")
        phase = "final" if index == len(stages) - 1 else "intermediate"
        for item in group:
            json_object(item, "checkpoint")
            if set(item) != {"id", "description"}:
                raise AgentError("checkpoint requires exactly id and description")
            try:
                checkpoint = Checkpoint(**item, stage=index, phase=phase)
            except (TypeError, ValueError):
                raise AgentError("invalid checkpoint") from None
            if checkpoint.id in seen or len(checkpoint.id) > 100 or len(checkpoint.description) > 4000:
                raise AgentError("duplicate or oversized checkpoint")
            seen.add(checkpoint.id)
            checkpoints.append(checkpoint)
    return checkpoints


def apply_stage_review(payload: dict, checkpoints: list[Checkpoint]) -> list[Checkpoint]:
    """Let a timing review split stages without losing any planned requirement."""
    json_object(payload, "stage review")
    stages = payload.get("stages")
    if set(payload) != {"stages"} or not isinstance(stages, list) or not 1 <= len(stages) <= len(checkpoints):
        raise AgentError("stage review requires groups of existing checkpoint ids")
    by_id = {checkpoint.id: checkpoint for checkpoint in checkpoints}
    seen = set()
    reviewed = []
    last_stage = -1
    for index, ids in enumerate(stages):
        if not isinstance(ids, list) or not ids:
            raise AgentError("stage review groups cannot be empty")
        group = []
        for identifier in ids:
            if not isinstance(identifier, str) or identifier not in by_id or identifier in seen:
                raise AgentError("stage review must use every checkpoint exactly once")
            seen.add(identifier)
            group.append(by_id[identifier])
        original = group[0].stage
        if original < last_stage or any(item.stage != original for item in group):
            raise AgentError("stage review cannot merge or reorder existing stages")
        last_stage = original
        phase = "final" if index == len(stages) - 1 else "intermediate"
        reviewed.extend(replace(item, stage=index, phase=phase) for item in group)
    if len(seen) != len(by_id):
        raise AgentError("stage review must use every checkpoint exactly once")
    return reviewed


def parse_task(payload: dict, checkpoints: list[Checkpoint]) -> TaskDirective:
    json_object(payload, "task")
    if set(payload) != {"checkpoint_ids", "objective", "reason", "finish"}:
        raise AgentError("task requires only checkpoint_ids, objective, reason, finish")
    try:
        task = TaskDirective(**payload)
    except (TypeError, ValueError):
        raise AgentError("invalid task directive") from None
    if not set(task.checkpoint_ids).issubset({checkpoint.id for checkpoint in checkpoints}):
        raise AgentError("task references an unknown checkpoint")
    return task


class OrchestratorAgent:
    """Select goals; the runner retains authority over budgets and final verdicts."""
    kind = "llm"
    model = MODEL_NAME
    agent_architecture = "three_role"

    def __init__(self, client: ModelClient) -> None:
        self.client = client

    @property
    def usage(self):
        return self.client.usage

    def plan(self, scenario: Scenario) -> list[Checkpoint]:
        prompt = (
            "You are the ORCHESTRATOR of a Web testing team. Extract every stated check into ordered stages. "
            'Return one JSON object with only "stages", an array of nonempty arrays of {id,description} '
            "checkpoints. Use unique IDs across all stages. Group checks on the same state in one stage. "
            "Start a new stage when another prescribed action must occur between checks. Every check tied "
            "to a state that a subsequent action intentionally replaces MUST be in an earlier stage. "
            'Example: "Enable feature, verify Enabled, then disable feature and verify Disabled" yields '
            '{"stages":[[{"id":"C1","description":"After enabling the feature, verify Enabled."}],'
            '[{"id":"C2","description":"After disabling the feature that was enabled, verify Disabled."}]]}. '
            "A scenario that only checks the outcome at the end has one stage. Preserve all explicit checks, "
            "including checks before submitting or at successive counts; do not invent new assertions for "
            "ordinary clicks or typing. State the required preceding action and exact expected evidence "
            "in each description. If no assertions are stated, specify the observable completion outcome. "
            "Preserve ordering, prerequisites and object identity. Do not fabricate state or verdicts. "
            + SCENARIO_DATA_RULE
            + "Include each specified value in the checkpoint where it is used or verified. Resolve references "
            "such as 'the item' to the specified object when its name is known."
        )
        checkpoints = parse_plan(self.client.request("orchestrator", prompt,
                                 {"scenario": scenario.description, "base_url": scenario.base_url}))
        groups: list[list[dict]] = []
        for item in checkpoints:
            if item.stage == len(groups):
                groups.append([])
            groups[item.stage].append({"id": item.id, "description": item.description})
        if len(groups) == len(checkpoints):
            return checkpoints
        review_prompt = (
            "Review the timing of the planned checkpoints against the original Web test scenario. "
            "Each group must contain only checks that can be verified in the same page state. "
            "Split a group whenever the scenario explicitly changes state between its checks. "
            "Evidence from before that change must be witnessed before moving on, even if the URL "
            "and control are the same. For example, verify the selected options, then reset and "
            "verify no options are selected: these checks need two groups. Checks for several "
            "properties of a single state stay together. Return only a JSON object with key "
            '"stages", with an array of nonempty arrays of existing checkpoint IDs, for example '
            '{"stages":[["C1","C2"],["C3"]]}. Use every ID exactly once, preserve existing group '
            "boundaries and their order, and do not add or rewrite any check. Return the same groups "
            "if no split is needed."
        )
        try:
            review = self.client.request("orchestrator", review_prompt,
                                         {"scenario": scenario.description, "stages": groups})
            return apply_stage_review(review, checkpoints)
        except AgentError:
            # An optional review must not replace a validated plan with an invalid one.
            return checkpoints

    def select_task(self, scenario: Scenario, observation: Observation, checkpoints: list[Checkpoint],
                    history: list[dict], remaining_steps: int) -> TaskDirective:
        prompt = (
            "You are the ORCHESTRATOR of a Web testing team. Select the next immediate testing objective "
            "for the actor, respecting the original scenario order, prerequisites, and acceptance conditions. "
            + UNTRUSTED_DATA_RULE + SCENARIO_DATA_RULE + ACTION_HISTORY_RULE
            + "Carry the original data into the objective when it is needed. Do not proceed on a substitute "
            "that was entered earlier by mistake. The supplied checkpoints are for the active stage. Achieve and verify this stage before "
            "doing any later operation that would change its evidence. "
            "Return only JSON with exactly checkpoint_ids (list of known ids), objective (string), "
            "reason (string), finish (boolean). The objective may include a necessary navigation or setup step "
            "toward the cited requirements. Prefer unresolved requirements; use error feedback to adjust the goal "
            "rather than repeating unproductive attempts. Do not propose a concrete browser action or assess "
            "PASS/FAIL yourself: the actor and assertor do those jobs. Use finish=true only when no useful "
            "permitted work remains; it is a request to stop, NEVER a declaration of PASS."
        )
        content = {"scenario": scenario.description, "base_url": scenario.base_url,
                   "observation": observation_content(observation),
                   "checkpoints": [asdict(checkpoint) for checkpoint in checkpoints],
                   "history": history[-12:], "remaining_steps": remaining_steps}
        return parse_task(self.client.request("orchestrator", prompt, content), checkpoints)
