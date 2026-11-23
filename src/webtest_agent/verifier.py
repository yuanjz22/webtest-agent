"""Conservative evidence checks, independent of planner action selection.

Text grounding is necessary, not sufficient, for semantic correctness. The model
still interprets the requirement. Independent teacher review is not replaced by
these checks; object-level before/after validation is a subsequent milestone.
"""
from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

from .models import Checkpoint, Observation


def element_supports_quote(element: dict, quote: str) -> bool:
    return quote in element.get("facts", []) or any(
        isinstance(value, str) and "[redacted]" not in value and quote in value
        for value in (element.get("text"), element.get("name"), element.get("value"))
    )


def quote_is_observed(quote: str, observation: Observation) -> bool:
    """Ground a quote in rendered text or a captured visible element's textual fields."""
    if not isinstance(quote, str) or not quote.strip() or "[redacted]" in quote:
        return False
    if quote in observation.text:
        return True
    return any(element_supports_quote(element, quote) for element in observation.elements)


RELATIVE_TIME_REQUIREMENT = re.compile(
    r"\b(?:relative time|elapsed|timestamp|time since|\d+ seconds?|countdown|minutes? ago)\b"
    r"|相对时间|倒计时|时间戳|秒前|分钟前", re.I,
)


def mask_relative_time_lines(text: str, labels) -> str:
    return "\n".join("<relative time>" if line in labels else line for line in text.split("\n"))


def terminal_evidence_still_current(before: Observation, after: Observation,
                                    requirements: str, quotes: list[str]) -> bool:
    """Allow a display clock to tick if its absolute value and the evidence stay the same."""
    if before.url != after.url:
        return False
    if before.state_hash == after.state_hash:
        return True
    old, new = before.relative_times, after.relative_times
    if not old or old == new or [t["absolute"] for t in old] != [t["absolute"] for t in new]:
        return False
    if RELATIVE_TIME_REQUIREMENT.search(requirements):
        return False
    labels = {t["label"] for t in old + new}
    if any(any(label in quote for label in labels) or not quote_is_observed(quote, after) for quote in quotes):
        return False

    def normalized(observation):
        counts = Counter(t["label"] for t in observation.relative_times)

        def mask(text):
            return mask_relative_time_lines(text, counts)

        # Ambiguous or inline occurrences need the ordinary, stricter recheck.
        if Counter(line for line in observation.text.splitlines() if line in counts) != counts:
            return None
        elements = [{key: mask(value) if key in ("text", "name") and isinstance(value, str) else value
                     for key, value in item.items() if key != "id"} for item in observation.elements]
        return mask(observation.text), elements

    old_state = normalized(before)
    return old_state is not None and old_state == normalized(after)


def invalidate_changed_page(checkpoints: list[Checkpoint], observation: Observation) -> list[str]:
    """Invalidate checks on a changed page, retain evidence captured on other URLs.

This is intentionally coarse. Cross-page data dependencies are not yet modeled.
Invalidation means recheck, never an application failure.
"""
    invalidated = []
    for checkpoint in checkpoints:
        if (checkpoint.phase == "final" and checkpoint.status in {"PASS", "FAIL"}
                and checkpoint.scope_url == observation.url
                and checkpoint.state_hash != observation.state_hash):
            checkpoint.status = "PENDING"
            checkpoint.evidence_ids = []
            checkpoint.quote = ""
            checkpoint.reason = "Page state changed; prior evidence must be rechecked."
            checkpoint.state_hash = ""
            invalidated.append(checkpoint.id)
    return invalidated


def apply_assessments(checkpoints: list[Checkpoint], assessments: list[dict],
                      observation: Observation) -> list[dict]:
    """Only accept verdicts grounded in the current saved browser observation."""
    by_id = {checkpoint.id: checkpoint for checkpoint in checkpoints}
    results = []
    seen = set()
    for assessment in assessments:
        checkpoint_id = assessment.get("checkpoint_id") if isinstance(assessment, dict) else None
        rejection = ""
        if checkpoint_id not in by_id:
            rejection = "unknown checkpoint id"
        elif by_id[checkpoint_id].phase == "intermediate" and by_id[checkpoint_id].status == "PASS":
            rejection = "intermediate requirement already witnessed"
        elif checkpoint_id in seen:
            rejection = "duplicate assessment in one decision"
        else:
            seen.add(checkpoint_id)
        status = assessment.get("status") if isinstance(assessment, dict) else None
        if not rejection and status not in {"PASS", "FAIL", "UNKNOWN"}:
            rejection = "invalid assessment status"
        evidence = assessment.get("evidence_ids", []) if isinstance(assessment, dict) else []
        quote = assessment.get("quote", "") if isinstance(assessment, dict) else ""
        reason = assessment.get("reason", "") if isinstance(assessment, dict) else ""
        if not rejection and status in {"PASS", "FAIL"}:
            # Accept a narrower citation only if its exact live element supports this quote.
            if (isinstance(quote, str) and quote.strip() and "[redacted]" not in quote
                    and isinstance(evidence, list) and len(evidence) == 1 and evidence[0] != observation.id):
                for element in observation.elements:
                    if element.get("id") == evidence[0] and element_supports_quote(element, quote):
                        evidence = [observation.id]
                        break
            if not isinstance(evidence, list) or evidence != [observation.id]:
                rejection = "assessment must cite the current observation exactly once"
            elif not quote_is_observed(quote, observation):
                rejection = "quote is not grounded in observed page text or visible element fields"
            elif not observation.screenshot or not Path(observation.screenshot).is_file():
                rejection = "saved screenshot is missing"
            elif not isinstance(reason, str) or not reason.strip():
                rejection = "assessment must explain how evidence meets or contradicts the requirement"
        if not rejection:
            checkpoint = by_id[checkpoint_id]
            checkpoint.status = status
            checkpoint.evidence_ids = list(evidence) if status != "UNKNOWN" else []
            checkpoint.quote = quote if status != "UNKNOWN" else ""
            checkpoint.reason = reason if isinstance(reason, str) else ""
            checkpoint.state_hash = observation.state_hash
            checkpoint.scope_url = observation.url
        results.append({"checkpoint_id": checkpoint_id, "accepted": not bool(rejection),
                        "status": status, "rejection": rejection})
    return results


def verdict_from_checkpoints(checkpoints: list[Checkpoint]) -> tuple[str, str]:
    if any(checkpoint.status == "FAIL" and checkpoint.evidence_ids for checkpoint in checkpoints):
        return "FAIL", "Observed evidence contradicts at least one stated requirement."
    if checkpoints and all(checkpoint.status == "PASS" and checkpoint.evidence_ids for checkpoint in checkpoints):
        return "PASS", "All stated checkpoints have grounded observation and screenshot evidence."
    return "INCONCLUSIVE", "One or more requirements lack reliable verification."

