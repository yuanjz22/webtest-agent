"""Assertor agent: assess page evidence independently from action selection."""
from __future__ import annotations

from dataclasses import asdict

from .llm import AgentError, ModelClient, SCENARIO_DATA_RULE, UNTRUSTED_DATA_RULE, json_object, observation_content
from .models import ActionOutcome, AssessmentResult, Checkpoint, Observation, Scenario


def parse_assessment_result(payload: dict) -> AssessmentResult:
    json_object(payload, "assertor output")
    if set(payload) != {"assessments", "reason"}:
        raise AgentError("assertor output requires only assessments and reason; actions are not allowed")
    assessments = payload["assessments"]
    if not isinstance(assessments, list) or len(assessments) > 30:
        raise AgentError("assessments must be a list with at most 30 entries")
    if not isinstance(payload["reason"], str) or len(payload["reason"]) > 4000:
        raise AgentError("invalid assertor reason")
    for item in assessments:
        json_object(item, "assessment")
        if set(item) != {"checkpoint_id", "status", "evidence_ids", "quote", "reason"}:
            raise AgentError("assessment fields must be checkpoint_id, status, evidence_ids, quote, reason")
        if (not isinstance(item["checkpoint_id"], str)
                or not isinstance(item["status"], str)
                or item["status"] not in {"PASS", "FAIL", "UNKNOWN"}):
            raise AgentError("invalid assessment id or status")
        if not isinstance(item["evidence_ids"], list) or not all(isinstance(i, str) for i in item["evidence_ids"]):
            raise AgentError("evidence_ids must be a list of strings")
        if not isinstance(item["quote"], str) or not isinstance(item["reason"], str):
            raise AgentError("assessment quote and reason must be strings")
    return AssessmentResult(assessments, payload["reason"])


class AssertorAgent:
    def __init__(self, client: ModelClient) -> None:
        self.client = client

    def _confirm_failures(self, result: AssessmentResult, content: dict) -> AssessmentResult:
        proposed = [item for item in result.assessments if item["status"] == "FAIL"]
        if not proposed:
            return result

        prompt = (
            "Review proposed failure verdicts for the active stage. "
            + UNTRUSTED_DATA_RULE +
            'Return only {"confirmed_failure_ids":[]} using a subset of the proposed checkpoint IDs. '
            "Confirm a FAIL only if the exact prerequisite operation was attempted on the intended "
            "object and the current evidence then contradicts the requirement. Check the last action, "
            "its target, and both observations. Filling an input does not submit its form. Prepared "
            "input plus a missing list item is an unfinished step, not a failure. Prior success on "
            "another object does not count. If the relevant action has yet to occur, evidence is "
            "still settling, or the proposal reason itself says the prerequisite was not done, "
            "exclude that ID so the runner can continue the scenario. Do not add or rewrite any check."
        )
        review = self.client.request("assertor", prompt, {**content, "proposed_failures": proposed})
        ids = review.get("confirmed_failure_ids")
        allowed = {item["checkpoint_id"] for item in proposed}
        if (set(review) != {"confirmed_failure_ids"} or not isinstance(ids, list)
                or not all(isinstance(value, str) and value in allowed for value in ids)):
            raise AgentError("failure review must return a subset of the proposed checkpoint ids")
        confirmed = set(ids)
        if confirmed == allowed:
            return result
        reason = "Failure review found insufficient evidence that the prerequisite was completed."
        assessments = [
            {**item, "status": "UNKNOWN", "evidence_ids": [], "quote": "", "reason": reason}
            if item["status"] == "FAIL" and item["checkpoint_id"] not in confirmed else item
            for item in result.assessments
        ]
        return AssessmentResult(assessments, reason)

    def assess(self, scenario: Scenario, observation: Observation, checkpoints: list[Checkpoint],
               previous_observation: Observation | None, last_action: ActionOutcome | None) -> AssessmentResult:
        prompt = (
            "You are the ASSERTOR of a Web testing team. Verify acceptance conditions using page evidence. "
            + UNTRUSTED_DATA_RULE + SCENARIO_DATA_RULE
            + "Compare the checkpoint with the original scenario if a name or value is omitted. "
            "Evidence of the workflow on a substituted value or object cannot pass the requested check. "
            "If the test acted on the wrong data, mark the affected check UNKNOWN rather than PASS or FAIL. "
            "Return only JSON with exactly assessments and reason. Each assessment is "
            "{checkpoint_id,status,evidence_ids,quote,reason}; status is PASS, FAIL, or UNKNOWN. "
            f'For PASS/FAIL use evidence_ids=["{observation.id}"] exactly, including for an element fact. '
            "Quote EXACT current page text or a captured "
            "visible element's text/name/value or copy one entire string from its facts array. Use checked=true/false "
            "facts for checkbox state, never the value 'on'. For exact whitespace use the raw quoted row text fact. "
            "Prefer one short assessment per supported checkpoint and omit unresolved ones. "
            "Assess only the supplied active stage and omit intermediate checks already marked PASS. "
            "Do not assess a later phase until its procedure occurred; a later expected state is not a failure "
            "of an earlier intermediate check. Return a JSON object, e.g. "
            '{"assessments":[],"reason":"Waiting for prerequisite"}. Every item must have all FIVE fields; '
            'evidence_ids is an array of strings; quote and reason are strings. Explain '
            "why it satisfies or contradicts the requirement. Compare previous/current observations when "
            "a state change is required; verify the same object's identity, not merely a matching name. "
            "The previous observation is context, never a replacement for current evidence. "
            "An action succeeding does not imply an acceptance condition passed. An action error, timeout, "
            "missing information, or unobserved state is NOT proof of an application failure. "
            "Absence alone without a completed scoped check is NOT proof of FAIL. Use UNKNOWN when evidence "
            "shows loading, an unfinished prerequisite, or inconsistent/transitional page state. Do not fail "
            "a postcondition before its prerequisite action and resulting page update have completed. "
            "Trace the exact last action to its target in the previous observation. State what prerequisite "
            "was actually performed before proposing FAIL. Creating an object does not perform subsequent "
            "edits to it. A state that the actor is still supposed to change is UNKNOWN, not FAIL. "
            "Input values are in elements[].value and may not appear in the body text. "
            "Use UNKNOWN when evidence "
            "is insufficient. Omit checks outside the current page's scope so valid other-page evidence can "
            "be retained. Do not propose actions, change the requirements, or decide when execution stops."
        )
        # The actor's reasoning and the orchestrator's stopping claims are deliberately absent.
        content = {"scenario": scenario.description, "base_url": scenario.base_url,
                   "observation": observation_content(observation),
                   "checkpoints": [asdict(checkpoint) for checkpoint in checkpoints],
                   "previous_observation": observation_content(previous_observation) if previous_observation else None,
                   "last_action": asdict(last_action) if last_action else None}
        if previous_observation and last_action:
            content["last_action_target"] = next((item for item in previous_observation.elements
                                                   if item.get("id") == last_action.action.target), None)
        payload = self.client.request("assertor", prompt, content)
        try:
            result = parse_assessment_result(payload)
        except AgentError as exc:
            # One schema correction, without introducing a second browser action or changing evidence.
            correction = (" Previous reply failed validation: " + str(exc) +
                          '. Example item: {"checkpoint_id":"C1","status":"UNKNOWN","evidence_ids":[],'
                          '"quote":"","reason":"Insufficient evidence"}.')
            result = parse_assessment_result(self.client.request("assertor", prompt + correction, content))
        return self._confirm_failures(result, content)
