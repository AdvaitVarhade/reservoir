"""Automated Playwright Integration Test for Sediment Hotspot Dashboard."""

import os
import shutil
import sys
import unittest
from pathlib import Path

from playwright.sync_api import sync_playwright

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_playwright_test import (
    DashboardPlaywrightTester,
    StreamlitServerProcess,
    find_free_port,
)


class TestDashboardPlaywright(unittest.TestCase):
    """Playwright integration test suite running against a live Streamlit test server."""

    @classmethod
    def setUpClass(cls):
        cls.screenshot_dir = REPO_ROOT / "data/outputs/playwright_screenshots"
        cls.screenshot_dir.mkdir(parents=True, exist_ok=True)
        cls.port = find_free_port(8515)
        cls.server = StreamlitServerProcess(port=cls.port)
        cls.server.start()

        cls.playwright = sync_playwright().start()
        cls.browser = cls.playwright.chromium.launch(channel="chrome", headless=True)
        cls.context = cls.browser.new_context(viewport={"width": 1440, "height": 900})
        cls.page = cls.context.new_page()
        cls.page.goto(cls.server.url, wait_until="networkidle", timeout=45000)
        cls.page.locator("[role='tab']").first.wait_for(state="visible", timeout=40000)
        cls.page.wait_for_timeout(2000)
        cls.tester = DashboardPlaywrightTester(cls.page, cls.screenshot_dir)

    @classmethod
    def tearDownClass(cls):
        try:
            cls.context.close()
            cls.browser.close()
            cls.playwright.stop()
        except Exception:
            pass
        finally:
            cls.server.stop()

    def _verify_new_results(self, start_idx: int):
        new_results = self.tester.results[start_idx:]
        self.assertGreater(len(new_results), 0, "No assertions were recorded during the test execution.")
        for r in new_results:
            self.assertEqual(r["status"], "PASS", f"Failed sub-test '{r['test_name']}': {r['details']}")

    def test_01_app_header_and_tabs(self):
        start = len(self.tester.results)
        self.tester.test_01_app_header_and_tabs()
        self._verify_new_results(start)

    def test_02_tab_overview(self):
        start = len(self.tester.results)
        self.tester.test_02_tab_overview()
        self._verify_new_results(start)

    def test_03_tab_rainfall_forecast(self):
        start = len(self.tester.results)
        self.tester.test_03_tab_rainfall_forecast()
        self._verify_new_results(start)

    def test_04_tab_hotspot_and_dsdi(self):
        start = len(self.tester.results)
        self.tester.test_04_tab_hotspot_and_dsdi()
        self._verify_new_results(start)

    def test_05_tab_zone_risk(self):
        start = len(self.tester.results)
        self.tester.test_05_tab_zone_risk()
        self._verify_new_results(start)

    def test_06_tab_epistemic_uncertainty(self):
        start = len(self.tester.results)
        self.tester.test_06_tab_epistemic_uncertainty()
        self._verify_new_results(start)

    def test_07_tab_explainability_causal_attention(self):
        start = len(self.tester.results)
        self.tester.test_07_tab_explainability_causal_attention()
        self._verify_new_results(start)

    def test_08_tab_counterfactual_scenarios(self):
        start = len(self.tester.results)
        self.tester.test_08_tab_counterfactual_scenarios()
        self._verify_new_results(start)

    def test_09_tab_multi_reservoir_analysis(self):
        start = len(self.tester.results)
        self.tester.test_09_tab_multi_reservoir_analysis()
        self._verify_new_results(start)

    def test_10_tab_model_benchmark_and_evaluation(self):
        start = len(self.tester.results)
        self.tester.test_10_tab_model_benchmark_and_evaluation()
        self._verify_new_results(start)


if __name__ == "__main__":
    unittest.main()
