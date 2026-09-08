"""Capture the README screenshots by driving the running application.

Usage::

    docker compose up -d          # or: streamlit run app/streamlit_app.py
    python scripts/capture_screenshots.py

Screenshots taken by hand go stale silently: the interface changes, the images do not,
and the README ends up describing a version of the app that no longer exists. This
script regenerates all six from the app as it is right now, so a stale image is one
command away from being fixed rather than a photo session.

The AI panel is captured *before* any request is made, which shows the payload preview
and costs nothing. Pass ``--with-ai`` to press the button first and capture real
suggestions instead - that makes a billed call to the Anthropic API with the key in your
environment.

Requires ``playwright`` and its Chromium build::

    pip install playwright && playwright install chromium
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - import kept out of the runtime path so the
    # missing-dependency message below stays friendlier than an ImportError traceback.
    from playwright.sync_api import Page

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "docs" / "images"
DEFAULT_DATASET = PROJECT_ROOT / "data" / "demo" / "customers.csv"
DEFAULT_URL = "http://localhost:8501"

VIEWPORT = {"width": 1440, "height": 950}
#: Streamlit re-runs the whole script on every interaction, so each click is followed by
#: a settle period rather than a single element wait.
SETTLE_MS = 2500


def _require_playwright():  # noqa: ANN202 - the import is the point
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print(
            "playwright is not installed. Run:\n"
            "  pip install playwright && playwright install chromium",
            file=sys.stderr,
        )
        raise SystemExit(2) from None
    return sync_playwright


def _open_tab(page: Page, label: str) -> None:
    """Select one of the app's tabs."""
    tab = page.get_by_role("tab", name=label)
    # Clicking the tab that is already open hangs: Streamlit re-runs on the event and
    # Playwright waits for a state change that never comes.
    if tab.get_attribute("aria-selected") != "true":
        tab.click(timeout=CLICK_TIMEOUT_MS)
        page.wait_for_timeout(SETTLE_MS)


def _scroll_tabs_into_frame(page: Page) -> None:
    """Put the tab bar just below the sticky header.

    Two traps here. The document itself does not scroll - Streamlit puts the overflow on
    an inner container, so ``window.scrollTo`` is silently a no-op and the screenshot
    keeps showing the page header. And a plain ``scrollIntoView`` parks the tab bar
    *underneath* the fixed header, which then intercepts every click aimed at a tab, so
    the header's height has to be subtracted.
    """
    page.evaluate(
        """() => {
            const tabs = document.querySelector('[role=tablist]');
            if (!tabs) return;

            let node = tabs.parentElement;
            let scroller = document.scrollingElement;
            while (node && node !== document.body) {
                const style = getComputedStyle(node);
                const scrolls = style.overflowY === 'auto' || style.overflowY === 'scroll';
                if (scrolls && node.scrollHeight > node.clientHeight + 5) {
                    scroller = node;
                    break;
                }
                node = node.parentElement;
            }

            const header = document.querySelector('[data-testid="stHeader"]');
            const offset = header ? header.getBoundingClientRect().height + 12 : 80;
            scroller.scrollTop += tabs.getBoundingClientRect().top - offset;
        }"""
    )
    page.wait_for_timeout(600)


#: Playwright blocks on `document.fonts.ready` before every screenshot. Streamlit pulls
#: its typeface from Google Fonts, so on a slow or filtered network that wait is where
#: the script appears to hang. Waiting explicitly turns an indefinite stall into an error
#: with a message, and the fallback shot below still produces a usable image.
FONT_TIMEOUT_MS = 20_000
SCREENSHOT_TIMEOUT_MS = 60_000
CLICK_TIMEOUT_MS = 30_000
RENDER_TIMEOUT_MS = 60_000


def _wait_for_render(page: Page) -> None:
    """Block until Streamlit has replaced its loading skeletons with real content.

    A fixed sleep is not enough: charts and metrics arrive on their own schedule, and a
    screenshot taken a moment early captures grey placeholder boxes that look like a
    broken app rather than an empty one.
    """
    try:
        page.wait_for_function(
            # Only visible skeletons count. Streamlit keeps the other tabs' content in
            # the DOM, hidden, and skeletons parked in there never resolve - waiting on
            # those would time out on every screenshot no matter how ready the page is.
            """() => !Array.from(document.querySelectorAll('[data-testid="stSkeleton"]'))
                     .some(node => node.offsetParent !== null)""",
            timeout=RENDER_TIMEOUT_MS,
        )
    except Exception:  # noqa: BLE001 - reported, not fatal
        print("  (content was still loading; the capture may show placeholders)")
    page.wait_for_timeout(1200)


def _expand(page: Page, limit: int) -> None:
    """Open the first few expanders on the current tab.

    The `:visible` filter matters: Streamlit leaves the other tabs' expanders in the DOM,
    hidden, and a click aimed at one of those waits for a visibility that never comes.
    """
    for expander in page.locator("details summary:visible").all()[:limit]:
        expander.click()
        page.wait_for_timeout(400)
    page.wait_for_timeout(SETTLE_MS)


def _shot(page: Page, output: Path, name: str, scroll_to_tabs: bool = False) -> None:
    destination = output / name
    # Render first, then scroll: on a page that is still loading skeletons the content
    # below the fold does not exist yet, so scrolling to it lands nowhere.
    _wait_for_render(page)
    if scroll_to_tabs:
        _scroll_tabs_into_frame(page)
    try:
        page.wait_for_function("document.fonts.status === 'loaded'", timeout=FONT_TIMEOUT_MS)
    except Exception:  # noqa: BLE001 - a missing web font is not worth failing the run
        print(f"  ({name}: web fonts did not finish loading; capturing anyway)")
    page.screenshot(path=str(destination), timeout=SCREENSHOT_TIMEOUT_MS)
    print(f"  {name}")


def capture(url: str, dataset: Path, output: Path, with_ai: bool) -> int:
    """Drive the app and write every screenshot. Returns a process exit code."""
    sync_playwright = _require_playwright()
    output.mkdir(parents=True, exist_ok=True)

    from dqcopilot.reporting.report import report_bytes
    from dqcopilot.services.analysis import analyze_bytes
    from dqcopilot.services.review import ReviewSession

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        page = browser.new_page(viewport=VIEWPORT, device_scale_factor=2)
        page.goto(url, wait_until="networkidle")
        page.wait_for_timeout(SETTLE_MS)

        print(f"Capturing into {output}:")
        _shot(page, output, "01-upload.png")

        page.set_input_files("input[type=file]", str(dataset))
        page.wait_for_selector("text=Quality score", timeout=60_000)
        page.wait_for_timeout(SETTLE_MS)

        _open_tab(page, "📊 Dashboard")
        _shot(page, output, "02-dashboard.png", scroll_to_tabs=True)

        _open_tab(page, "🔍 Findings")
        # A list of collapsed rows shows nothing of the explanations, which are the part
        # worth showing.
        _expand(page, limit=3)
        _shot(page, output, "03-findings.png")

        _open_tab(page, "🛠️ Corrections")
        # Expand the first proposals: collapsed rows hide the before/after preview, which
        # is the whole argument that nothing is applied sight unseen.
        _expand(page, limit=2)
        _shot(page, output, "04-corrections.png", scroll_to_tabs=True)

        _open_tab(page, "🤖 AI suggestions")
        if with_ai:
            page.get_by_role("button", name="Generate AI suggestions").click()
            page.wait_for_timeout(45_000)
        else:
            # The payload preview is the honest thing to show: it is what the privacy
            # boundary looks like from the user's side.
            page.get_by_text("Show exactly what would be sent").click()
            page.wait_for_timeout(SETTLE_MS)
        _shot(page, output, "05-ai.png", scroll_to_tabs=True)

        # The report is a standalone file, so it is rendered and shot directly rather
        # than photographed through the download button.
        analysis = analyze_bytes(dataset.read_bytes(), dataset.name)
        review = ReviewSession.from_analysis(analysis)
        report = output.parent / "_report-preview.html"
        report.write_bytes(report_bytes(review, ai_enabled=False))
        page.goto(report.as_uri(), wait_until="networkidle")
        page.wait_for_timeout(1000)
        _shot(page, output, "06-report.png")
        report.unlink(missing_ok=True)

        try:
            browser.close()
        except Exception as error:  # noqa: BLE001 - teardown, every screenshot is written
            # The Windows driver sometimes drops the pipe here. Every file is already on
            # disk at this point, so failing the run would report a problem that is not
            # one - but staying silent about it would be worse.
            print(f"  (browser did not shut down cleanly: {error})", file=sys.stderr)

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=DEFAULT_URL, help=f"Running app (default: {DEFAULT_URL})")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--with-ai",
        action="store_true",
        help="Press 'Generate AI suggestions' first. Makes a billed Anthropic API call.",
    )
    args = parser.parse_args()

    if not args.dataset.is_file():
        print(f"No such dataset: {args.dataset}", file=sys.stderr)
        return 2

    return capture(args.url, args.dataset, args.output, args.with_ai)


if __name__ == "__main__":
    raise SystemExit(main())
