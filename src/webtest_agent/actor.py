"""Actor agent: turn an assigned objective into one permitted browser action."""
from __future__ import annotations

from dataclasses import asdict

from .llm import (ACTION_HISTORY_RULE, SCENARIO_DATA_RULE, UNTRUSTED_DATA_RULE,
                  AgentError, ModelClient, json_object, observation_content)
from .models import ACTION_KINDS, Action, ActionDecision, Observation, Scenario, TaskDirective


def parse_action_decision(payload: dict) -> ActionDecision:
    json_object(payload, "actor output")
    if set(payload) != {"action", "reason"}:
        raise AgentError("actor output requires only action and reason; assessments are not allowed")
    raw = json_object(payload["action"], "action")
    if set(raw) - {"kind", "target", "value"}:
        raise AgentError("unexpected action fields")
    if not isinstance(payload["reason"], str) or len(payload["reason"]) > 4000:
        raise AgentError("invalid actor reason")
    try:
        return ActionDecision(Action(**raw), payload["reason"])
    except (TypeError, ValueError):
        raise AgentError("invalid actor action") from None


class ActorAgent:
    def __init__(self, client: ModelClient) -> None:
        self.client = client

    def decide(self, scenario: Scenario, observation: Observation, task: TaskDirective,
               history: list[dict], remaining_steps: int) -> ActionDecision:
        prompt = (
            "You are the ACTOR of a Web testing team. Choose ONE browser action toward the assigned objective. "
            + UNTRUSTED_DATA_RULE + SCENARIO_DATA_RULE + ACTION_HISTORY_RULE
            + "For fill, take any prescribed literal directly from the original scenario, even if the "
            "assigned objective omitted it or the field currently contains different text. "
            f"Return only JSON with exactly action and reason. action is {{kind,target,value}}; "
            f"kind is {', '.join(sorted(ACTION_KINDS))}. Use only CURRENT observation element ids for targets. "
            "Use dblclick for one double-click gesture and hover to reveal controls on pointer movement. "
            "Use reload with empty target and value when the scenario requires refreshing the current page. "
            "For navigate value is a same-origin URL; for drag value is a current destination "
            "element id; scroll value is pixel amount; press value is a key; wait value is milliseconds. "
            "For a native select, use the exact visible option label or its exact value. "
            "A drag between groups should use the source marked Drag handle and the destination marked "
            "Drop zone for the intended group. Its heading is normally not a drop target. "
            "Use a control's name, Title, and Within facts to distinguish equally named controls. "
            "Every browser action, including failures, retries, and waits, consumes one shared step. "
            "Do not propose actions outside the original scenario. Never return assessments or an application "
            "verdict. Use finish when no useful permitted action is available, or when remaining_steps is zero."
        )
        content = {"scenario": scenario.description, "base_url": scenario.base_url,
                   "observation": observation_content(observation), "task": asdict(task),
                   "history": history[-12:], "remaining_steps": remaining_steps}
        return parse_action_decision(self.client.request("actor", prompt, content))
