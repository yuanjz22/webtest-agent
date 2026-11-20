"""Real browser integration tests, explicitly skipped when unavailable."""
import importlib.util
import json
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from webtest_agent.browser import BrowserError, PlaywrightBrowser, launch_browser, origin
from webtest_agent.demo import DEMO_SCENARIO, demo_server
from webtest_agent.models import Action, Checkpoint, Scenario
from webtest_agent.service import execute_run
from webtest_agent.storage import RunStore
from webtest_agent.verifier import apply_assessments


class OriginTests(unittest.TestCase):
    def test_origin_normalizes_default_port(self):
        self.assertEqual(origin("https://EXAMPLE.com/a"), origin("https://example.com:443/b"))
        self.assertNotEqual(origin("http://localhost:80"), origin("http://localhost:81"))

    def test_rejects_non_web_and_credentials(self):
        for url in ["file:///etc/passwd", "javascript:alert(1)", "http://user:password@localhost", "http://localhost:bad"]:
            with self.subTest(url=url), self.assertRaises(BrowserError):
                origin(url)


class DemoServerTests(unittest.TestCase):
    def _post(self, url, path, payload):
        request = Request(url + path, data=json.dumps(payload).encode("utf-8"),
                          headers={"Content-Type": "application/json"})
        with urlopen(request) as response:
            return json.load(response)

    def test_cart_add_update_and_reset(self):
        with demo_server() as url:
            added = self._post(url, "/cart/add", {})
            self.assertEqual(added["name"], "Notebook")
            self.assertEqual(added["quantity"], 1)
            updated = self._post(url, "/cart/update", {"quantity": 2})
            self.assertEqual(updated["quantity"], 2)
            self.assertEqual(updated["total_cents"], 5000)
            with urlopen(url + "/cart") as response:
                self.assertEqual(json.load(response), updated)
            self.assertEqual(self._post(url, "/reset", {})["quantity"], 0)
            with urlopen(url + "/cart") as response:
                self.assertEqual(json.load(response)["total_cents"], 0)

    def test_invalid_quantity_does_not_change_cart(self):
        with demo_server() as url:
            self._post(url, "/cart/add", {})
            for quantity in (0, 21, True, "2", 1.5, None, [2]):
                with self.subTest(quantity=quantity):
                    with self.assertRaises(HTTPError) as caught:
                        self._post(url, "/cart/update", {"quantity": quantity})
                    self.assertEqual(caught.exception.code, 400)
                    caught.exception.close()
            with urlopen(url + "/cart") as response:
                self.assertEqual(json.load(response)["quantity"], 1)

    def test_update_requires_an_item_and_add_respects_limit(self):
        with demo_server() as url:
            with self.assertRaises(HTTPError) as caught:
                self._post(url, "/cart/update", {"quantity": 2})
            self.assertEqual(caught.exception.code, 409)
            caught.exception.close()
            self._post(url, "/cart/add", {})
            self._post(url, "/cart/update", {"quantity": 20})
            with self.assertRaises(HTTPError) as caught:
                self._post(url, "/cart/add", {})
            self.assertEqual(caught.exception.code, 409)
            caught.exception.close()
            with urlopen(url + "/cart") as response:
                self.assertEqual(json.load(response)["quantity"], 20)


@unittest.skipUnless(importlib.util.find_spec("playwright"), "Playwright is not installed; run pip install -e .[dev]")
class BrowserIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from playwright.sync_api import sync_playwright
        try:
            with sync_playwright() as playwright:
                browser = launch_browser(playwright)
                browser.close()
        except Exception as exc:
            raise unittest.SkipTest(
                "Browser unavailable; install Google Chrome, or set WEBTEST_BROWSER_CHANNEL=chromium "
                "and run python -m playwright install chromium: " + str(exc)
            )

    def _target(self, observation, name):
        return next(e["id"] for e in observation.elements if e["name"] == name)

    def _open(self, browser, url):
        browser.execute(Action("navigate", value=url))
        browser._page.wait_for_function("!document.getElementById('add').disabled")

    def _update_cart(self, browser, url):
        self._open(browser, url)
        before = browser.observe()
        browser.execute(Action("click", self._target(before, "Add to cart")))
        browser._page.wait_for_selector("#cart-details", state="visible")
        form = browser.observe()
        browser.execute(Action("fill", self._target(form, "Quantity"), "2"))
        filled = browser.observe()
        self.assertIn("Cart quantity: 1", filled.text)
        self.assertIn("Cart total: CNY 25.00", filled.text)
        browser.execute(Action("click", self._target(filled, "Update cart")))
        # A separate explicit wait is a browser operation, as in the runner.
        browser.execute(Action("wait", value="200"))
        return browser.observe()

    def test_real_success_screenshots_trace_and_stale_targets(self):
        with demo_server() as url, tempfile.TemporaryDirectory() as directory:
            with PlaywrightBrowser(url, Path(directory)) as browser:
                observed = self._update_cart(browser, url)
                self.assertIn("Cart item: Notebook", observed.text)
                self.assertIn("Cart quantity: 2", observed.text)
                self.assertIn("Cart total: CNY 50.00", observed.text)
                self.assertTrue(Path(observed.screenshot).is_file())
                previous = self._target(observed, "Add to cart")
                browser.observe()
                with self.assertRaises(BrowserError) as caught:
                    browser.execute(Action("click", previous))
                self.assertEqual(caught.exception.kind, "locator")
            self.assertGreater((Path(directory) / "trace.zip").stat().st_size, 0)

    def test_wrong_total_is_visible_in_evidence(self):
        with demo_server() as url, tempfile.TemporaryDirectory() as directory:
            with PlaywrightBrowser(url, Path(directory)) as browser:
                observed = self._update_cart(browser, url + "/?fault=wrong_total")
                self.assertIn("Cart quantity: 2", observed.text)
                self.assertIn("Cart total: CNY 25.00", observed.text)
                self.assertNotIn("Cart total: CNY 50.00", observed.text)

    def test_dom_change_during_screenshot_retries_the_entire_snapshot(self):
        with demo_server() as url, tempfile.TemporaryDirectory() as directory:
            with PlaywrightBrowser(url, Path(directory)) as browser:
                self._open(browser, url)
                original = browser._page.screenshot
                captures = 0

                def screenshot_during_update(**kwargs):
                    nonlocal captures
                    captures += 1
                    if captures == 1:
                        browser._page.evaluate("""() => {
                            document.getElementById('status').textContent = 'Update complete';
                            document.getElementById('cart-empty').textContent = 'Cart snapshot updated';
                        }""")
                    return original(**kwargs)

                with patch.object(browser._page, "screenshot", side_effect=screenshot_during_update):
                    observed = browser.observe()
                self.assertEqual(captures, 2)
                self.assertIn("Update complete", observed.text)
                self.assertIn("Cart snapshot updated", observed.text)
                self.assertNotIn("Your cart is empty.", observed.text)
                status = next(item for item in observed.elements if item["dom_id"] == "status")
                self.assertEqual(status["text"], "Update complete")
                snapshot = json.loads((Path(directory) / (observed.id + ".json")).read_text(encoding="utf-8"))
                self.assertEqual(snapshot["capture_attempts"], 2)
                # Handles must still refer to live elements after the snapshot's JS handles were disposed.
                browser.execute(Action("click", self._target(observed, "Add to cart")))
                browser._page.wait_for_selector("#cart-details", state="visible")
                self.assertIsNotNone(self._target(browser.observe(), "Quantity"))

    def test_unstable_capture_does_not_publish_mixed_evidence(self):
        with demo_server() as url, tempfile.TemporaryDirectory() as directory:
            with PlaywrightBrowser(url, Path(directory)) as browser:
                self._open(browser, url)
                original = browser._page.screenshot
                captures = 0

                def changing_screenshot(**kwargs):
                    nonlocal captures
                    captures += 1
                    browser._page.evaluate("value => document.getElementById('status').textContent = value", str(captures))
                    return original(**kwargs)

                with patch.object(browser._page, "screenshot", side_effect=changing_screenshot):
                    with self.assertRaises(BrowserError) as caught:
                        browser.observe()
                self.assertEqual(caught.exception.kind, "observation")
                self.assertEqual(captures, browser.MAX_CAPTURE_ATTEMPTS)
                self.assertFalse(list(Path(directory).glob("obs-*.json")))
                self.assertFalse(list(Path(directory).glob("obs-*.png")))

    def test_visible_input_value_can_ground_an_assessment(self):
        with demo_server() as url, tempfile.TemporaryDirectory() as directory:
            with PlaywrightBrowser(url, Path(directory)) as browser:
                self._open(browser, url)
                browser.execute(Action("click", self._target(browser.observe(), "Add to cart")))
                browser._page.wait_for_selector("#cart-details", state="visible")
                browser.execute(Action("fill", self._target(browser.observe(), "Quantity"), "17"))
                observed = browser.observe()
                self.assertNotIn("17", observed.text)
                checkpoints = [Checkpoint("C1", "The Quantity input contains 17.")]
                results = apply_assessments(checkpoints, [{
                    "checkpoint_id": "C1", "status": "PASS", "evidence_ids": [observed.id],
                    "quote": "17", "reason": "The captured Quantity input value is 17.",
                }], observed)
                self.assertTrue(results[0]["accepted"])
                self.assertEqual(checkpoints[0].status, "PASS")

    def test_snapshot_preserves_open_shadow_root_targets_and_masks_passwords(self):
        with demo_server() as url, tempfile.TemporaryDirectory() as directory:
            with PlaywrightBrowser(url, Path(directory)) as browser:
                browser.execute(Action("navigate", value=url))
                browser._page.evaluate("""() => {
                    const host = document.createElement('div');
                    document.body.append(host);
                    host.attachShadow({mode: 'open'}).innerHTML =
                        '<label id="label">Secret input</label><input type="password" aria-labelledby="label" value="hidden-value">';
                }""")
                observed = browser.observe()
                element, = [item for item in observed.elements if item["name"] == "Secret input"]
                self.assertEqual(element["value"], "[redacted]")
                self.assertNotIn("hidden-value", json.dumps(observed.elements))
                browser.execute(Action("fill", element["id"], "another-hidden-value"))

    def test_hover_then_double_click_exposes_visible_controls_and_focus_evidence(self):
        html = """
        <style>.delete{display:none} li:hover .delete{display:inline}</style>
        <ul><li><label id="title" ondblclick="document.querySelector('#edit').hidden=false;
                    document.querySelector('#edit').focus(); this.hidden=true">Draft</label>
          <button aria-label="Delete Draft" class="delete">&times;</button>
          <input id="edit" hidden value="Draft" aria-label="Edit task">
        </li></ul>"""
        with demo_server() as url, tempfile.TemporaryDirectory() as directory:
            with PlaywrightBrowser(url, Path(directory)) as browser:
                page_url = url + "/gesture-fixture"
                browser._page.route(page_url, lambda route: route.fulfill(content_type="text/html", body=html))
                browser.execute(Action("navigate", value=page_url))
                initial = browser.observe()
                label = next(element for element in initial.elements if element["name"] == "Draft")
                self.assertEqual(label["tag"], "label")
                self.assertNotIn("Delete Draft", {element["name"] for element in initial.elements})
                browser.execute(Action("hover", label["id"]))
                hovered = browser.observe()
                self.assertIn("Delete Draft", {element["name"] for element in hovered.elements})
                browser.execute(Action("dblclick", self._target(hovered, "Draft")))
                edited = browser.observe()
                field = next(element for element in edited.elements if element["name"] == "Edit task")
                self.assertIn('Field Edit task value="Draft"', field["facts"])
                self.assertIn("Field Edit task focused=true", field["facts"])
                self.assertNotIn("Draft", {element["name"] for element in edited.elements})

    def test_closed_menu_controls_are_not_offered_until_menu_opens(self):
        html = """
        <button onclick="document.getElementById('drawer').setAttribute('aria-hidden', 'false')">
          Open menu
        </button>
        <aside id="drawer" aria-hidden="true">
          <button id="drawer-action" onclick="this.textContent = 'Done'">Menu action</button>
        </aside>
        <div inert><button>Inactive action</button></div>
        """
        with demo_server() as url, tempfile.TemporaryDirectory() as directory:
            with PlaywrightBrowser(url, Path(directory)) as browser:
                page_url = url + "/menu-fixture"
                browser._page.route(page_url, lambda route: route.fulfill(content_type="text/html", body=html))
                browser.execute(Action("navigate", value=page_url))
                closed = browser.observe()
                self.assertNotIn("Menu action", {element["name"] for element in closed.elements})
                self.assertNotIn("Inactive action", {element["name"] for element in closed.elements})
                browser.execute(Action("click", self._target(closed, "Open menu")))
                opened = browser.observe()
                browser.execute(Action("click", self._target(opened, "Menu action")))
                self.assertIn("Done", {element["name"] for element in browser.observe().elements})

    def test_reload_issues_a_new_request_with_fragment_and_preserves_current_page(self):
        loads = []

        def serve(route):
            loads.append(route.request.url)
            route.fulfill(content_type="text/html", body=(
                f"<p>Document load {len(loads)}</p>"
                '<a href="#/active" class="selected">Active</a>'))

        with demo_server() as url, tempfile.TemporaryDirectory() as directory:
            with PlaywrightBrowser(url, Path(directory)) as browser:
                page_url = url + "/reload-fixture"
                browser._page.route(page_url, serve)
                browser.execute(Action("navigate", value=page_url + "#/active"))
                before = browser.observe()
                self.assertIn("Document load 1", before.text)
                browser.execute(Action("reload"))
                after = browser.observe()
                self.assertEqual(after.url, before.url)
                self.assertEqual(loads, [page_url, page_url])
                self.assertIn("Document load 2", after.text)
                link = next(element for element in after.elements if element["name"] == "Active")
                self.assertIn('Link "Active" class=selected', link["facts"])

    def test_navigation_and_network_remain_same_origin(self):
        with demo_server() as url, demo_server() as other, tempfile.TemporaryDirectory() as directory:
            with PlaywrightBrowser(url, Path(directory)) as browser:
                browser.execute(Action("navigate", value=url))
                with self.assertRaises(BrowserError):
                    browser.execute(Action("navigate", value=other))
                # Test scaffolding injects a cross-origin request; models cannot
                # request evaluation through the adapter's execute interface.
                result = browser._page.evaluate("url => fetch(url).then(()=> 'sent').catch(()=> 'blocked')", other + "/health")
                self.assertEqual(result, "blocked")
                self.assertIn(other + "/health", browser.blocked_requests)
                with self.assertRaises(ValueError):
                    browser.execute(Action("evaluate", value="document.body.innerHTML=''"))

    def test_role_pipeline_saves_real_demo_success_and_failure_evidence(self):
        for query, expected in (("", "PASS"), ("/?fault=wrong_total", "FAIL")):
            with self.subTest(expected=expected), demo_server() as url, tempfile.TemporaryDirectory() as directory:
                store = RunStore(Path(directory))
                scenario = Scenario("local-demo", DEMO_SCENARIO, url + query, 10)
                run_id = store.create(scenario)
                report = execute_run(store, scenario, run_id, mode="demo")
                self.assertEqual(report["verdict"], expected)
                self.assertEqual(report["planner_kind"], "demo_fixture")
                self.assertEqual(report["agent_architecture"], "scripted_fixture")
                self.assertEqual(report["usage"]["requests"], 0)
                self.assertLessEqual(report["steps"], 10)
                roles = {event["role"] for event in report["events"] if event["kind"] == "agent_started"}
                self.assertEqual(roles, {"orchestrator", "actor", "assertor"})
                self.assertGreater((store.artifact_dir(run_id) / "trace.zip").stat().st_size, 0)
                for observation in report["observations"]:
                    self.assertTrue(Path(observation["screenshot"]).is_file())
                self.assertEqual(store.get(run_id)["report"]["verdict"], expected)


if __name__ == "__main__":
    unittest.main()

