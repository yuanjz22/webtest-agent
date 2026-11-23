import pytest

from webtest_agent.models import Checkpoint, Observation
from webtest_agent.verifier import (apply_assessments, invalidate_changed_page, quote_is_observed,
                                    terminal_evidence_still_current)


def test_stable_evidence_survives_only_a_relative_timestamp_tick():
    def snapshot(seconds, description="Draft is ready.", absolute="21.11.2026 18:20"):
        label = f"{seconds} seconds ago"
        text = f"Card details\n{label}\n{description}"
        return Observation(str(seconds), "http://localhost/card/42", text,
                           [{"id": f"obs-{seconds}-modal", "name": text, "role": "dialog"}],
                           "", text, [{"label": label, "absolute": absolute}])

    start, end = snapshot(8), snapshot(10)
    assert terminal_evidence_still_current(start, end, "Verify the description.", ["Draft is ready."])
    assert not terminal_evidence_still_current(start, end, "Verify the relative time.", ["Draft is ready."])
    assert not terminal_evidence_still_current(start, end, "Verify the card.", ["8 seconds ago"])
    assert not terminal_evidence_still_current(start, snapshot(10, "Draft was rejected."), "Verify the card.", ["Draft is ready."])
    assert not terminal_evidence_still_current(start, snapshot(10, absolute="22.11.2026 18:20"), "Verify the card.", ["Draft is ready."])


@pytest.mark.parametrize("quote,expected", [
    ("Ready", True), ("Quantity", True), ("17", True),
    ("invented", False), ("", False), ("[redacted]", False),
    ("https://example.invalid", False),
])
def test_quotes_use_observed_textual_fields_not_arbitrary_metadata(quote, expected):
    observation = Observation("E1", "http://localhost", "Ready", [
        {"name": "Quantity", "text": "", "value": "17", "href": "https://example.invalid"},
        {"name": "Password", "value": "[redacted]"},
    ], "", "state")
    assert quote_is_observed(quote, observation) is expected


def test_intermediate_evidence_survives_intended_transition_but_final_state_expires():
    common = dict(status="PASS", evidence_ids=["before"], scope_url="http://localhost", state_hash="old")
    milestone = Checkpoint("M", "Initially unchecked", phase="intermediate", **common)
    final = Checkpoint("F", "Finally unchecked", **common)
    after = Observation("after", "http://localhost", "", [{"facts": ["Row A | checkbox | checked=true"]}], "", "new")
    assert invalidate_changed_page([milestone, final], after) == ["F"]
    apply_assessments([milestone], [{"checkpoint_id": "M", "status": "UNKNOWN"}], after)
    assert milestone.evidence_ids == ["before"] and milestone.status == "PASS"
    assert quote_is_observed("Row A | checkbox | checked=true", after)
    assert not quote_is_observed("Row A | checkbox | checked=false", after)


@pytest.mark.parametrize("citation,accepted", [("E1-e1", True), ("E1-e2", False), ("E0-e1", False)])
def test_element_citation_must_support_the_quote_in_the_current_snapshot(tmp_path, citation, accepted):
    screenshot = tmp_path / "evidence.png"
    screenshot.touch()
    fact = "Row A | checkbox | checked=true"
    observation = Observation("E1", "http://localhost", "Row A", [
        {"id": "E1-e1", "facts": [fact]},
        {"id": "E1-e2", "facts": ["Row B | checkbox | checked=false"]},
    ], str(screenshot), "state")
    checkpoint = Checkpoint("C1", "Row A is checked")
    result = apply_assessments([checkpoint], [{
        "checkpoint_id": "C1", "status": "PASS", "evidence_ids": [citation], "quote": fact,
        "reason": "The control is checked",
    }], observation)
    assert result[0]["accepted"] is accepted
    assert checkpoint.evidence_ids == (["E1"] if accepted else [])
