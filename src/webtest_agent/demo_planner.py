"""Scripted local-demo fixture implementing the three role interfaces, without an LLM."""
from __future__ import annotations

from .llm import AgentError
from .models import Action, ActionDecision, AssessmentResult, Checkpoint, TaskDirective


class DemoPlanner:
    kind = "demo_fixture"
    model = None
    agent_architecture = "scripted_fixture"

    def __init__(self):
        self.usage = {"requests": 0, "input_tokens": 0, "output_tokens": 0}

    @staticmethod
    def _check(scenario):
        if scenario.id != "local-demo":
            raise AgentError("DemoPlanner is only a scripted local-demo fixture; never use it for evaluation.")

    def plan(self, scenario):
        self._check(scenario)
        return [Checkpoint("C1", "The cart contains Notebook."),
                Checkpoint("C2", "After updating the cart, its saved quantity is 2."),
                Checkpoint("C3", "Two Notebooks at CNY 25.00 each have a cart total of CNY 50.00.")]

    def select_task(self, scenario, observation, checkpoints, history, remaining_steps):
        self._check(scenario)
        pending = next((checkpoint for checkpoint in checkpoints if checkpoint.status != "PASS"), checkpoints[0])
        return TaskDirective([pending.id], pending.description, "Continue the local smoke fixture.")

    def decide(self, scenario, observation, task, history, remaining_steps):
        self._check(scenario)
        if remaining_steps <= 0:
            return ActionDecision(Action("finish"), "No browser budget remains.")
        by_name = {element.get("name", element.get("text", "")): element for element in observation.elements}
        lines = {line.strip() for line in observation.text.splitlines()}
        if lines.intersection({"Loading cart...", "Adding item...", "Updating cart...", "Resetting cart..."}):
            return ActionDecision(Action("wait", value="100"), "Wait for the fixture's cart request to complete.")
        if "Cart item: Notebook" not in lines:
            add = by_name.get("Add to cart")
            if add and not add.get("disabled"):
                return ActionDecision(Action("click", str(add["id"])), "Add Notebook to the cart.")
        field = by_name.get("Quantity")
        if field and not field.get("disabled") and field.get("value") != "2":
            return ActionDecision(Action("fill", str(field["id"]), "2"), "Set the requested quantity to 2.")
        update = by_name.get("Update cart")
        if "Cart quantity: 2" not in lines and update and not update.get("disabled"):
            return ActionDecision(Action("click", str(update["id"])), "Save the updated quantity before checking the total.")
        return ActionDecision(Action("finish"), "The local fixture controls were not found.")

    def assess(self, scenario, observation, checkpoints, previous_observation, last_action):
        self._check(scenario)
        assessments = []
        lines = {line.strip() for line in observation.text.splitlines()}
        for checkpoint_id, quote in (("C1", "Cart item: Notebook"), ("C2", "Cart quantity: 2"),
                                     ("C3", "Cart total: CNY 50.00")):
            if quote in lines:
                assessments.append({"checkpoint_id": checkpoint_id, "status": "PASS", "evidence_ids": [observation.id],
                                    "quote": quote, "reason": "The fixture shows the required saved cart state."})
        total = next((line for line in lines if line.startswith("Cart total: CNY ")), "")
        if "Cart quantity: 2" in lines and total and total != "Cart total: CNY 50.00":
            assessments.append({"checkpoint_id": "C3", "status": "FAIL", "evidence_ids": [observation.id],
                                "quote": total, "reason": "The saved quantity is 2 but the displayed total differs from 2 x CNY 25.00."})
        return AssessmentResult(assessments, "Check the fixture's current visible evidence.")
