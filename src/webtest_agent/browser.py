"""Small Playwright adapter. It executes tools, never plans or counts steps.

The runner owns the action budget. Observation IDs address the exact element
handles seen in the latest snapshot, so an old index cannot silently select a
different node after a re-render. This initial adapter targets the main frame.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import urljoin, urlsplit

from .models import Action, Observation


class BrowserError(RuntimeError):
    def __init__(self, kind: str, message: str):
        self.kind = kind
        super().__init__(message)


def origin(url: str) -> tuple[str, str, int]:
    """Reject credentials and non-Web URLs rather than guessing an origin."""
    try:
        parsed = urlsplit(url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("Expected an HTTP(S) URL")
        if parsed.username is not None or parsed.password is not None:
            raise ValueError("Credentials in URLs are not supported")
        return parsed.scheme, parsed.hostname.lower(), parsed.port or (443 if parsed.scheme == "https" else 80)
    except ValueError as exc:
        raise BrowserError("navigation", str(exc)) from exc


def launch_browser(playwright, *, headless: bool = True):
    """Use installed Chrome by default, or explicitly select bundled Chromium."""
    channel = os.environ.get("WEBTEST_BROWSER_CHANNEL", "chrome").strip().lower() or "chrome"
    if channel not in {"chrome", "chromium"}:
        raise ValueError("WEBTEST_BROWSER_CHANNEL must be 'chrome' or 'chromium'")
    return playwright.chromium.launch(
        headless=headless, channel="chrome" if channel == "chrome" else None,
    )


class PlaywrightBrowser:
    """One isolated browser context and a real trace per run.

    Strict same-origin routing deliberately excludes third-party CDN/auth
    services. Configure a local deployment with same-origin assets for this MVP.
    No model-produced JavaScript or CSS selector is evaluated.
    """

    ELEMENT_SELECTOR = (
        "a[href],button,input,textarea,select,label,[role],[contenteditable=true],[draggable=true],"
        "[data-rbd-droppable-id],[data-rfd-droppable-id]"
    )
    MAX_CAPTURE_ATTEMPTS = 3
    # One synchronous browser task reads text, values, and node identities together.
    SNAPSHOT_SCRIPT = r"""selector => {
        function* candidates(root) {
            for (const el of root.querySelectorAll('*')) {
                if (el.matches(selector)) yield el;
                if (el.shadowRoot) yield* candidates(el.shadowRoot);
            }
        }
        function inHiddenSubtree(el) {
            for (let node = el; node; node = node.parentElement || node.getRootNode().host) {
                if (node.inert || node.getAttribute('aria-hidden') === 'true') return true;
            }
            return false;
        }
        const dragHandle = '[data-rbd-drag-handle-draggable-id],[data-rfd-drag-handle-draggable-id]';
        const dragItem = '[data-rbd-draggable-id],[data-rfd-draggable-id]';
        const dropZone = '[data-rbd-droppable-id],[data-rfd-droppable-id]';
        const dragGroup = el => {
            const item = el.closest(dragItem);
            if (!item) return '';
            const handle = item.matches(dragHandle) ? item
                : Array.from(item.querySelectorAll(dragHandle)).find(handle => handle.closest(dragItem) === item);
            // Use the handle's visible label, not the library's opaque internal identifier.
            return (handle?.innerText || '').trim().split('\n')[0].slice(0, 120);
        };
        const labelIdsByRoot = new WeakMap();
        function isControlLabel(el) {
            if (el.control) return true;
            if (!el.id) return false;
            const root = el.getRootNode();
            if (!labelIdsByRoot.has(root)) {
                const ids = new Set();
                for (const control of root.querySelectorAll('[aria-labelledby]')) {
                    if (control.tagName !== 'LABEL' && control.matches(selector)) {
                        for (const id of control.getAttribute('aria-labelledby').trim().split(/\s+/)) ids.add(id);
                    }
                }
                labelIdsByRoot.set(root, ids);
            }
            return labelIdsByRoot.get(root).has(el.id);
        }
        const nodes = [], elements = [];
        for (const el of candidates(document)) {
            if (elements.length >= 200) break;
            if (inHiddenSubtree(el)) continue;
            const style = getComputedStyle(el), rect = el.getBoundingClientRect();
            if (style.visibility === 'hidden' || style.visibility === 'collapse'
                || !rect.width || !rect.height) continue;
            const tag = el.tagName.toLowerCase();
            // A label with an associated control is already represented by that control.
            if (tag === 'label' && isControlLabel(el)) continue;
            const defaults = {button:'button', a:'link', textarea:'textbox', select:'combobox', label:'label'};
            const type = tag === 'input' ? el.type : (el.getAttribute('type') || '');
            let role = el.getAttribute('role') || defaults[tag] || '';
            if (tag === 'input') role = ['checkbox','radio'].includes(type) ? type
                : ['submit','button'].includes(type) ? 'button' : 'textbox';
            const label = Array.from(el.labels || []).map(n => n.innerText).join(' ');
            const labelledBy = (el.getAttribute('aria-labelledby') || '').split(/\s+/)
                .map(id => el.getRootNode().getElementById?.(id)?.innerText || '').join(' ').trim();
            const text = (el.innerText || '').trim().slice(0, 500);
            const title = (el.getAttribute('title') || '').trim().slice(0, 160);
            const drop = el.matches(dropZone);
            const group = el.matches(dragHandle) ? dragGroup(el.parentElement) : dragGroup(el);
            const name = ((drop && group ? 'Drop zone for ' + group : '') || el.getAttribute('aria-label') || labelledBy || label
                || el.getAttribute('placeholder') || text
                || (['submit','button'].includes(type) ? el.value : '') || title).slice(0,500);
            const facts = [];
            if (title && title !== name) facts.push('Title=' + JSON.stringify(title));
            if (group && group !== name) facts.push('Within ' + JSON.stringify(group));
            if (el.matches(dragHandle) || el.getAttribute('draggable') === 'true') facts.push('Drag handle');
            if (drop) facts.push('Drop zone' + (group ? ' for ' + JSON.stringify(group) : ''));
            if (role === 'link') {
                const current = el.getAttribute('aria-current');
                if (current && current !== 'false') facts.push('Link ' + JSON.stringify(name) + ' aria-current=' + current);
                if (el.classList.contains('selected')) facts.push('Link ' + JSON.stringify(name) + ' class=selected');
            }
            const row = el.closest('li,tr,[role=row],[role=listitem]');
            const context = row ? row.innerText.trim().slice(0,160) : '';
            if (role === 'checkbox' || role === 'radio') {
                const checked = el.matches('input[type=checkbox],input[type=radio]')
                    ? (el.indeterminate ? 'mixed' : String(el.checked)) : el.getAttribute('aria-checked');
                if (checked !== null) facts.push([context, el.getAttribute('aria-label') || label || role,
                    'checked=' + checked].filter(Boolean).join(' | '));
            }
            if (tag === 'textarea' || (tag === 'input' && /^(text|search|email|number|url|tel|)$/.test(type))) {
                const field = 'Field ' + (name || tag);
                facts.push(field + ' value=' + JSON.stringify(el.value.slice(0,500)));
                if (el.getRootNode().activeElement === el) facts.push(field + ' focused=true');
            }
            if (tag === 'select') {
                const options = Array.from(el.options).filter(option => !option.matches(':disabled') && !option.hidden);
                for (const option of options.slice(0, 60)) {
                    facts.push('Option ' + JSON.stringify(option.label.slice(0, 120))
                        + (option.selected ? ' selected=true' : ''));
                }
            }
            // Preserve whitespace for exact-title checks; exclude hidden descendants and editable values.
            if (row) for (const node of row.querySelectorAll('label,td,[role=cell]')) {
                if (!node.children.length && node.getClientRects().length && getComputedStyle(node).visibility === 'visible'
                    && node.innerText && node.textContent.length <= 500)
                    facts.push('Row text=' + JSON.stringify(node.textContent));
                if (facts.length >= 6) break;
            }
            elements.push({tag, role: drop && !role ? 'dropzone' : role, type, text, facts, name,
                value: type === 'password' ? '[redacted]' : String(el.value || '').slice(0,500),
                href: el.getAttribute('href') || '', disabled: Boolean(el.disabled),
                dom_id: el.id || '', test_id: el.getAttribute('data-testid') || ''});
            nodes.push(el);
        }
        const relativeTimes = [];
        for (const el of document.body?.querySelectorAll('time[datetime], [title]') || []) {
            if (el.children.length || !el.getClientRects().length || inHiddenSubtree(el)
                    || getComputedStyle(el).visibility !== 'visible') continue;
            const absolute = el.getAttribute('datetime') || el.getAttribute('title') || '';
            // Require an explicit date and time from the page, not just a time-like phrase.
            if (!/\d{1,4}[./-]\d{1,2}[./-]\d{1,4}.*\d{1,2}:\d{2}/.test(absolute)) continue;
            const label = el.innerText.trim();
            if (!/^(?:\d+ (?:seconds?|minutes?|hours?|days?) ago|\d+\s*(?:秒|分钟|小时|天)前)$/i.test(label)) continue;
            relativeTimes.push({label, absolute});
        }
        return {data: {url: location.href, text: (document.body?.innerText || '').slice(0,24000), elements,
            relative_times: relativeTimes}, nodes};
    }"""

    def __init__(self, base_url: str, artifact_dir: Path, headless: bool = True):
        self.base_url = base_url
        self.allowed_origin = origin(base_url)
        self.artifact_dir = Path(artifact_dir).resolve()
        self.headless = headless
        self._playwright = self._browser = self._context = self._page = None
        self._handles = {}
        self._observation_number = 0
        self._pages = []
        self.blocked_requests: list[str] = []
        self.timeout_ms = 5000

    def __enter__(self):
        try:
            from playwright.sync_api import sync_playwright
            self.artifact_dir.mkdir(parents=True, exist_ok=True)
            self._playwright = sync_playwright().start()
            self._browser = launch_browser(self._playwright, headless=self.headless)
            self._context = self._browser.new_context(
                viewport={"width": 1280, "height": 900},
                service_workers="block", accept_downloads=False,
            )
            self._context.set_default_timeout(self.timeout_ms)
            self._context.route("**/*", self._route)
            # HTTP routing does not intercept WebSocket handshakes.
            if hasattr(self._context, "route_web_socket"):
                self._context.route_web_socket("**/*", self._route_websocket)
            self._context.on("page", self._on_page)
            self._context.tracing.start(screenshots=True, snapshots=True, sources=True)
            self._page = self._context.new_page()
            return self
        except Exception as exc:
            self.close()
            raise BrowserError("navigation", "Cannot start Playwright browser: " + str(exc)) from exc

    def _allowed(self, url: str) -> bool:
        try:
            return origin(url) == self.allowed_origin
        except BrowserError:
            return False

    def _route(self, route):
        if self._allowed(route.request.url):
            route.continue_()
        else:
            self.blocked_requests.append(route.request.url)
            route.abort("blockedbyclient")

    def _route_websocket(self, socket):
        url = socket.url
        http_url = "https:" + url[4:] if url.startswith("wss:") else "http:" + url[3:]
        if self._allowed(http_url):
            socket.connect_to_server()
        else:
            self.blocked_requests.append(url)
            socket.close(code=1008, reason="Cross-origin connection blocked")

    def _on_page(self, page):
        self._pages.append(page)
        page.on("dialog", lambda dialog: dialog.dismiss())

    def _sync_page(self):
        if self._context is None:
            raise BrowserError("navigation", "Browser is not open")
        opened = [p for p in self._pages if not p.is_closed()]
        if not opened:
            raise BrowserError("navigation", "All browser pages are closed")
        candidate = opened[-1]
        if candidate.url != "about:blank" and not self._allowed(candidate.url):
            raise BrowserError("navigation", "Page left the configured origin")
        if candidate is not self._page:
            self._clear_handles()
            self._page = candidate

    def _clear_handles(self):
        for handle in self._handles.values():
            try:
                handle.dispose()
            except Exception:
                pass
        self._handles.clear()

    @contextmanager
    def _dom_snapshot(self):
        snapshot = self._page.evaluate_handle(self.SNAPSHOT_SCRIPT, self.ELEMENT_SELECTOR)
        try:
            data_handle = snapshot.get_property("data")
            try:
                data = data_handle.json_value()
            finally:
                data_handle.dispose()
            yield data, snapshot
        finally:
            snapshot.dispose()

    def _state_hash_data(self, data: dict) -> dict:
        """Return the state compared across separately captured observations."""
        return data

    def observe(self) -> Observation:
        """Publish a snapshot only when DOM data agrees before and after its screenshot."""
        self._sync_page()
        self._clear_handles()
        self._observation_number += 1
        observation_id = f"obs-{self._observation_number:04d}"
        screenshot = self.artifact_dir / f"{observation_id}.png"
        try:
            for attempt in range(1, self.MAX_CAPTURE_ATTEMPTS + 1):
                page = self._page
                try:
                    with self._dom_snapshot() as (before, _):
                        image_data = self._page.screenshot(full_page=True, timeout=self.timeout_ms)
                        with self._dom_snapshot() as (data, snapshot_handle):
                            if before != data:
                                continue
                            nodes = snapshot_handle.get_property("nodes")
                            try:
                                handles = nodes.get_properties()
                            finally:
                                nodes.dispose()
                except Exception as exc:
                    message = str(exc).lower()
                    if ("execution context was destroyed" not in message
                            and "cannot find context with specified id" not in message
                            and "cannot take a screenshot while page is navigating" not in message):
                        raise
                    self._sync_page()
                    self._page.wait_for_load_state("domcontentloaded", timeout=self.timeout_ms)
                    continue
                self._sync_page()
                if self._page is not page or self._page.url != data["url"]:
                    for handle in handles.values():
                        handle.dispose()
                    continue
                state_hash = hashlib.sha256(json.dumps(
                    self._state_hash_data(data), sort_keys=True, ensure_ascii=False,
                ).encode("utf-8")).hexdigest()
                text, elements = data["text"], data["elements"]
                for index, metadata in enumerate(elements):
                    element_id = f"{observation_id}-e{index + 1}"
                    metadata["id"] = element_id
                    self._handles[element_id] = handles[str(index)].as_element()
                break
            else:
                raise BrowserError("observation", "Page kept changing during screenshot capture; no consistent observation available")
            screenshot.write_bytes(image_data)
            observation = Observation(
                id=observation_id, url=data["url"], text=text,
                elements=elements, screenshot=str(screenshot), state_hash=state_hash,
                relative_times=data["relative_times"],
            )
            snapshot = {"id": observation_id, "url": observation.url, "text": text,
                        "elements": elements, "screenshot": str(screenshot), "state_hash": state_hash,
                        "relative_times": observation.relative_times,
                        "blocked_requests": list(self.blocked_requests), "capture_attempts": attempt,
                        "snapshot_consistency": "DOM matched before and after screenshot",
                        "limits": {"text_characters": 24000, "visible_elements": 200, "frame": "main"}}
            (self.artifact_dir / f"{observation_id}.json").write_text(
                json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
            return observation
        except BrowserError:
            raise
        except Exception as exc:
            self._raise_browser_error(exc)

    def _handle(self, target: str):
        if target not in self._handles:
            raise BrowserError("locator", "Target is not in the latest observation; observe again")
        handle = self._handles[target]
        if not handle.evaluate("el => el.isConnected"):
            raise BrowserError("locator", "Observed target was detached; observe again")
        return handle

    def _drag_with_pointer(self, source, destination) -> None:
        """Send one continuous gesture for pointer-sensor drag controls."""
        destination.scroll_into_view_if_needed(timeout=self.timeout_ms)
        source.scroll_into_view_if_needed(timeout=self.timeout_ms)
        points = []
        viewport = self._page.viewport_size
        for element in (source, destination):
            box = element.bounding_box()
            if not box:
                raise BrowserError("locator", "The drag source or destination is not visible")
            x, y = box["x"] + box["width"] / 2, box["y"] + box["height"] / 2
            if viewport and not (0 <= x < viewport["width"] and 0 <= y < viewport["height"]):
                raise BrowserError("locator", "Scroll until both drag targets are in view")
            points.append((x, y))
        (sx, sy), (tx, ty) = points
        distance = math.hypot(tx - sx, ty - sy)
        if distance < 8:
            raise BrowserError("locator", "Drag points overlap; choose a different visible destination")
        mouse = self._page.mouse
        mouse.move(sx, sy)
        mouse.down()
        try:
            # Pointer sensors need movement beyond their activation threshold.
            lead = min(12, distance)
            mouse.move(sx + (tx - sx) * lead / distance, sy + (ty - sy) * lead / distance, steps=2)
            mouse.move(tx, ty, steps=18)
            self._page.evaluate("() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)))")
        finally:
            mouse.up()

    def execute(self, action: Action) -> None:
        """Execute exactly one requested tool; recovery belongs to the runner."""
        self._sync_page()
        try:
            if action.kind == "navigate":
                url = urljoin(self.base_url, action.value)
                if not self._allowed(url):
                    raise BrowserError("navigation", "Navigation must remain on the configured origin")
                self._clear_handles()
                self._page.goto(url, wait_until="domcontentloaded", timeout=self.timeout_ms)
            elif action.kind == "reload":
                if not self._allowed(self._page.url):
                    raise BrowserError("navigation", "Reload requires a page on the configured origin")
                self._clear_handles()
                self._page.reload(wait_until="domcontentloaded", timeout=self.timeout_ms)
                # A client-rendered page may still have an empty app shell after DOMContentLoaded.
                from playwright.sync_api import TimeoutError as PlaywrightTimeout
                try:
                    self._page.wait_for_function("() => Boolean(document.body?.innerText.trim())",
                                                 timeout=min(self.timeout_ms, 1200))
                except PlaywrightTimeout:
                    pass  # Empty pages can be the actual result. Capture the evidence either way.
            elif action.kind in {"click", "dblclick"}:
                handle = self._handle(action.target)
                href = handle.get_attribute("href")
                # Some JS-driven controls are anchors with a literal no-op href.
                # Their handlers still run under normal browser origin restrictions.
                inert_href = href and href.strip().lower() in {"javascript:void(0)", "javascript:void(0);"}
                if href and not inert_href and not self._allowed(urljoin(self._page.url, href)):
                    raise BrowserError("navigation", "Link leaves the configured origin")
                click = handle.dblclick if action.kind == "dblclick" else handle.click
                click(timeout=self.timeout_ms)
            elif action.kind == "hover":
                self._handle(action.target).hover(timeout=self.timeout_ms)
            elif action.kind == "fill":
                self._handle(action.target).fill(action.value, timeout=self.timeout_ms)
            elif action.kind == "select":
                handle = self._handle(action.target)
                mode = handle.evaluate("""(el, requested) => {
                    if (el.tagName !== 'SELECT') return '';
                    const options = Array.from(el.options).filter(option => !option.matches(':disabled') && !option.hidden);
                    const expected = options.filter(option => option.value === requested || option.label === requested);
                    if (expected.length !== 1) return '';
                    return expected[0].value === requested ? 'value' : 'label';
                }""", action.value)
                if not mode:
                    raise BrowserError("locator", "Select requires one available option with an exact label or value")
                handle.select_option(**{mode: action.value}, timeout=self.timeout_ms)
            elif action.kind == "press":
                self._handle(action.target).press(action.value, timeout=self.timeout_ms)
            elif action.kind == "drag":
                source = self._handle(action.target)
                destination = self._handle(action.value)
                if (source.get_attribute("data-rbd-drag-handle-draggable-id") is not None
                        or source.get_attribute("data-rfd-drag-handle-draggable-id") is not None):
                    self._drag_with_pointer(source, destination)
                else:
                    # Native HTML drag uses Playwright's own actionability logic.
                    token = self._observation_number
                    source.evaluate("(el, id) => el.setAttribute('data-webtest-drag-source', id)", str(token))
                    destination.evaluate("(el, id) => el.setAttribute('data-webtest-drag-target', id)", str(token))
                    try:
                        self._page.locator(f'[data-webtest-drag-source="{token}"]').drag_to(
                            self._page.locator(f'[data-webtest-drag-target="{token}"]'), timeout=self.timeout_ms)
                    finally:
                        source.evaluate("el => el.removeAttribute('data-webtest-drag-source')")
                        destination.evaluate("el => el.removeAttribute('data-webtest-drag-target')")
            elif action.kind == "scroll":
                parts = action.value.split(",") if action.value else ["0", "600"]
                if len(parts) == 1:
                    parts = ["0", parts[0]]
                if len(parts) != 2:
                    raise BrowserError("unsupported", "Scroll value must be dy or dx,dy")
                dx, dy = map(float, parts)
                if not all(math.isfinite(v) and abs(v) <= 5000 for v in (dx, dy)):
                    raise BrowserError("unsupported", "Scroll distance must be finite and at most 5000 pixels")
                self._page.mouse.wheel(dx, dy)
            elif action.kind == "wait":
                milliseconds = float(action.value or "500")
                if not math.isfinite(milliseconds) or not 0 <= milliseconds <= 2000:
                    raise BrowserError("unsupported", "Explicit wait must be between 0 and 2000 ms")
                self._page.wait_for_timeout(milliseconds)
            else:
                raise BrowserError("unsupported", f"Unsupported browser action: {action.kind}")
            self._sync_page()
        except BrowserError:
            raise
        except (TypeError, ValueError) as exc:
            raise BrowserError("unsupported", str(exc)) from exc
        except Exception as exc:
            self._raise_browser_error(exc)

    @staticmethod
    def _raise_browser_error(exc):
        if "Timeout" in type(exc).__name__:
            kind = "timeout"
        elif any(word in str(exc).lower() for word in ("detached", "element", "selector")):
            kind = "locator"
        else:
            kind = "navigation"
        raise BrowserError(kind, str(exc)) from exc

    def close(self):
        """Save trace before closing, including when execution raised an error."""
        self._clear_handles()
        close_errors = []
        if self._context is not None:
            try:
                self._context.tracing.stop(path=str(self.artifact_dir / "trace.zip"))
            except Exception as exc:
                close_errors.append("trace: " + str(exc))
            try:
                self._context.close()
            except Exception as exc:
                close_errors.append("context: " + str(exc))
        if self._browser is not None:
            try:
                self._browser.close()
            except Exception as exc:
                close_errors.append("browser: " + str(exc))
        if self._playwright is not None:
            try:
                self._playwright.stop()
            except Exception as exc:
                close_errors.append("playwright: " + str(exc))
        self._context = self._browser = self._playwright = self._page = None
        if close_errors and self.artifact_dir.exists():
            (self.artifact_dir / "browser-close-errors.json").write_text(
                json.dumps(close_errors, indent=2), encoding="utf-8")

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False
