"""Shared, JSON-serializable contracts. No provider or browser dependencies."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlsplit


MODEL_NAME = "deepseek-flash"
MAX_STEPS = 35
ACTION_KINDS = frozenset({"navigate", "reload", "click", "dblclick", "hover", "fill", "select",
                          "scroll", "drag", "press", "wait", "finish"})
CHECKPOINT_STATUSES = frozenset({"PENDING", "PASS", "FAIL", "UNKNOWN"})


@dataclass
class Scenario:
    id: str
    description: str
    base_url: str
    max_steps: int = MAX_STEPS

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id.strip() or len(self.id) > 200:
            raise ValueError("scenario id must be a nonempty string of at most 200 characters")
        if not isinstance(self.description, str) or not self.description.strip() or len(self.description) > 20000:
            raise ValueError("scenario description must be a nonempty string of at most 20000 characters")
        if not isinstance(self.base_url, str):
            raise ValueError("base_url must be an HTTP(S) URL")
        parsed = urlsplit(self.base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password:
            raise ValueError("base_url must be HTTP(S), without embedded credentials")
        if type(self.max_steps) is not int or not 1 <= self.max_steps <= MAX_STEPS:
            raise ValueError("max_steps must be an integer between 1 and 35")


@dataclass
class Action:
    kind: str
    target: str = ""
    value: str = ""

    def __post_init__(self) -> None:
        if self.kind not in ACTION_KINDS:
            raise ValueError("unsupported action kind")
        if not isinstance(self.target, str) or not isinstance(self.value, str):
            raise ValueError("action target and value must be strings")
        if len(self.target) > 500 or len(self.value) > 20000:
            raise ValueError("action field exceeds size limit")
        if self.kind in {"click", "dblclick", "hover", "fill", "select", "drag"} and not self.target:
            raise ValueError("action requires an observed element id")
        if self.kind == "navigate" and not self.value:
            raise ValueError("navigate requires a URL in value")
        if self.kind == "reload" and (self.target or self.value):
            raise ValueError("reload takes no target or value")


@dataclass
class Checkpoint:
    id: str
    description: str
    status: str = "PENDING"
    evidence_ids: list[str] = field(default_factory=list)
    reason: str = ""
    quote: str = ""
    state_hash: str = ""
    scope_url: str = ""
    phase: str = "final"
    stage: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id or not isinstance(self.description, str) or not self.description.strip():
            raise ValueError("checkpoint id and description are required")
        if self.status not in CHECKPOINT_STATUSES:
            raise ValueError("invalid checkpoint status")
        if self.phase not in ("intermediate", "final"):
            raise ValueError("invalid checkpoint phase")
        if type(self.stage) is not int or not 0 <= self.stage < 30:
            raise ValueError("invalid checkpoint stage")


@dataclass
class Observation:
    id: str
    url: str
    text: str
    elements: list[dict]
    screenshot: str
    state_hash: str
    relative_times: list[dict[str, str]] = field(default_factory=list)


@dataclass
class TaskDirective:
    """The orchestrator's current objective; never an application verdict."""
    checkpoint_ids: list[str]
    objective: str
    reason: str = ""
    finish: bool = False

    def __post_init__(self) -> None:
        if (not isinstance(self.checkpoint_ids, list)
                or not all(isinstance(item, str) and item for item in self.checkpoint_ids)
                or len(self.checkpoint_ids) > 30
                or len(set(self.checkpoint_ids)) != len(self.checkpoint_ids)):
            raise ValueError("task requires distinct checkpoint ids")
        if (not isinstance(self.objective, str) or len(self.objective) > 4000
                or not isinstance(self.reason, str) or len(self.reason) > 4000
                or type(self.finish) is not bool):
            raise ValueError("invalid task objective, reason, or finish flag")
        if not self.finish and (not self.checkpoint_ids or not self.objective.strip()):
            raise ValueError("an active task requires checkpoints and an objective")


@dataclass
class ActionDecision:
    """An actor may propose an action, but cannot submit checkpoint assessments."""
    action: Action
    reason: str = ""


@dataclass
class ActionOutcome:
    action: Action
    success: bool
    error_kind: str = ""


@dataclass
class AssessmentResult:
    """An assertor may assess evidence, but cannot execute or propose actions."""
    assessments: list[dict] = field(default_factory=list)
    reason: str = ""


@dataclass
class RunReport:
    run_id: str
    scenario_id: str
    verdict: str
    reason: str
    steps: int
    checkpoints: list[Checkpoint]
    observations: list[Observation]
    events: list[dict]
    usage: dict[str, Any]
    duration_seconds: float
    strategy: str = "checkpoint"
    model: str | None = MODEL_NAME
    planner_kind: str = "llm"
    scenario: dict = field(default_factory=dict)
    agent_architecture: str = "three_role"

