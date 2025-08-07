"""Playwright test configuration and fixtures for QCM Generator Pro.

This module provides the test setup for comparing Streamlit and React interfaces.
"""

import pytest
import asyncio
from pathlib import Path
from typing import Generator, Optional
from playwright.async_api import async_playwright, Playwright, Browser, BrowserContext, Page
import subprocess
import time
import logging
import os

logger = logging.getLogger(__name__)

# Test configuration
STREAMLIT_URL = "http://localhost:8501"
REACT_URL = "http://localhost:3000"  # Future React app
API_URL = "http://localhost:8000"

# Timeout settings
STARTUP_TIMEOUT = 30  # seconds
PAGE_LOAD_TIMEOUT = 10  # seconds


# Remove custom event_loop fixture to use pytest-asyncio default


@pytest.fixture(scope="session")
async def playwright_instance() -> Generator[Playwright, None, None]:
    """Initialize Playwright instance."""
    async with async_playwright() as playwright:
        yield playwright


@pytest.fixture(scope="session")
async def browser(playwright_instance: Playwright) -> Generator[Browser, None, None]:
    """Launch browser for testing."""
    browser = await playwright_instance.chromium.launch(
        headless=False,  # Set to True for CI/CD
        args=[
            "--no-sandbox",
            "--disable-setuid-sandbox",
            "--disable-dev-shm-usage",
            "--disable-background-timer-throttling",
            "--disable-backgrounding-occluded-windows",
            "--disable-renderer-backgrounding",
        ]
    )
    yield browser
    await browser.close()


@pytest.fixture(scope="session")
async def context(browser: Browser) -> Generator[BrowserContext, None, None]:
    """Create browser context with optimal settings."""
    context = await browser.new_context(
        viewport={"width": 1920, "height": 1080},
        ignore_https_errors=True,
        record_video_dir="tests/playwright/videos",
        record_video_size={"width": 1920, "height": 1080}
    )
    yield context
    await context.close()


@pytest.fixture
async def streamlit_page(context: BrowserContext) -> Generator[Page, None, None]:
    """Create page for Streamlit interface."""
    page = await context.new_page()
    
    # Navigate to Streamlit app
    try:
        await page.goto(STREAMLIT_URL, timeout=PAGE_LOAD_TIMEOUT * 1000)
        await page.wait_for_load_state("networkidle", timeout=10000)
        logger.info("Successfully connected to Streamlit interface")
    except Exception as e:
        logger.warning(f"Could not connect to Streamlit: {e}")
        pytest.skip("Streamlit interface not available")
    
    yield page
    await page.close()


@pytest.fixture
async def react_page(context: BrowserContext) -> Generator[Page, None, None]:
    """Create page for React interface (future implementation)."""
    page = await context.new_page()
    
    # Try to navigate to React app (will be available later)
    try:
        await page.goto(REACT_URL, timeout=PAGE_LOAD_TIMEOUT * 1000)
        await page.wait_for_load_state("networkidle", timeout=10000)
        logger.info("Successfully connected to React interface")
    except Exception as e:
        logger.warning(f"Could not connect to React interface: {e}")
        pytest.skip("React interface not yet available")
    
    yield page
    await page.close()


@pytest.fixture
async def api_page(context: BrowserContext) -> Generator[Page, None, None]:
    """Create page for API testing."""
    page = await context.new_page()
    
    # Test API health endpoint
    try:
        await page.goto(f"{API_URL}/health", timeout=PAGE_LOAD_TIMEOUT * 1000)
        await page.wait_for_load_state("networkidle")
        logger.info("Successfully connected to FastAPI backend")
    except Exception as e:
        logger.warning(f"Could not connect to API: {e}")
        pytest.skip("FastAPI backend not available")
    
    yield page
    await page.close()


@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Set up test environment before running tests."""
    logger.info("Setting up Playwright test environment")
    
    # Create output directories
    Path("tests/playwright/screenshots").mkdir(parents=True, exist_ok=True)
    Path("tests/playwright/videos").mkdir(parents=True, exist_ok=True)
    Path("tests/playwright/reports").mkdir(parents=True, exist_ok=True)
    
    # Verify test data exists
    test_pdf = Path("tests/fixtures/sample.pdf")
    if not test_pdf.exists():
        logger.warning(f"Test PDF not found at {test_pdf}")
    
    yield
    
    logger.info("Cleaning up Playwright test environment")


def wait_for_service(url: str, timeout: int = STARTUP_TIMEOUT) -> bool:
    """Wait for a service to become available."""
    import requests
    
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                return True
        except requests.RequestException:
            pass
        time.sleep(1)
    
    return False


@pytest.fixture(scope="session")
def ensure_services_running():
    """Ensure required services are running before tests."""
    logger.info("Checking if services are running...")
    
    # Check API
    if not wait_for_service(f"{API_URL}/health"):
        logger.warning("FastAPI service not running")
    
    # Check Streamlit
    if not wait_for_service(STREAMLIT_URL):
        logger.warning("Streamlit service not running")
    
    # React will be checked in fixture when available
    
    yield


class ScreenshotHelper:
    """Helper class for managing screenshots."""
    
    def __init__(self, base_path: str = "tests/playwright/screenshots"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
    
    async def capture_comparison(self, streamlit_page: Page, react_page: Optional[Page], 
                               test_name: str, step: str = "") -> dict:
        """Capture screenshots for comparison between interfaces."""
        timestamp = int(time.time())
        step_suffix = f"_{step}" if step else ""
        
        results = {}
        
        # Capture Streamlit screenshot
        if streamlit_page:
            streamlit_path = self.base_path / f"{test_name}_streamlit{step_suffix}_{timestamp}.png"
            await streamlit_page.screenshot(path=str(streamlit_path), full_page=True)
            results["streamlit"] = streamlit_path
            logger.info(f"Captured Streamlit screenshot: {streamlit_path}")
        
        # Capture React screenshot (when available)
        if react_page:
            react_path = self.base_path / f"{test_name}_react{step_suffix}_{timestamp}.png"
            await react_page.screenshot(path=str(react_path), full_page=True)
            results["react"] = react_path
            logger.info(f"Captured React screenshot: {react_path}")
        
        return results


@pytest.fixture
def screenshot_helper():
    """Provide screenshot helper instance."""
    return ScreenshotHelper()


class InterfaceComparator:
    """Helper class for comparing interface functionality."""
    
    async def compare_upload_workflow(self, streamlit_page: Page, react_page: Optional[Page]) -> dict:
        """Compare document upload workflow between interfaces."""
        results = {"streamlit": {}, "react": {}}
        
        # Test Streamlit upload
        if streamlit_page:
            try:
                # Navigate to upload section
                await streamlit_page.get_by_text("Upload de Documents").click()
                await streamlit_page.wait_for_timeout(1000)
                
                # Check for file upload widget
                upload_widget = streamlit_page.locator('input[type="file"]')
                is_present = await upload_widget.count() > 0
                
                results["streamlit"] = {
                    "upload_widget_present": is_present,
                    "navigation_successful": True
                }
            except Exception as e:
                results["streamlit"] = {"error": str(e), "navigation_successful": False}
        
        # Test React upload (when available)
        if react_page:
            try:
                # Navigate to upload page
                await react_page.goto("http://localhost:3000/upload")
                await react_page.wait_for_timeout(1000)
                
                # Check for file upload input
                upload_input = react_page.locator('input[type="file"]')
                is_present = await upload_input.count() > 0
                
                results["react"] = {
                    "upload_widget_present": is_present,
                    "navigation_successful": True
                }
            except Exception as e:
                results["react"] = {"error": str(e), "navigation_successful": False}
        
        return results
    
    async def compare_generation_workflow(self, streamlit_page: Page, react_page: Optional[Page]) -> dict:
        """Compare QCM generation workflow between interfaces."""
        results = {"streamlit": {}, "react": {}}
        
        # Test Streamlit generation
        if streamlit_page:
            try:
                # Navigate to generation section
                await streamlit_page.get_by_text("Génération QCM").click()
                await streamlit_page.wait_for_timeout(1000)
                
                # Check for generation controls
                generation_button = streamlit_page.get_by_text("Générer QCM")
                is_present = await generation_button.count() > 0
                
                results["streamlit"] = {
                    "generation_button_present": is_present,
                    "navigation_successful": True
                }
            except Exception as e:
                results["streamlit"] = {"error": str(e), "navigation_successful": False}
        
        # Test React generation (when available)
        if react_page:
            try:
                # Navigate to generation page
                await react_page.goto("http://localhost:3000/generate")
                await react_page.wait_for_timeout(1000)
                
                # Check for generation controls
                generate_button = react_page.locator('button:has-text("Générer")')
                is_present = await generate_button.count() > 0
                
                results["react"] = {
                    "generation_button_present": is_present,
                    "navigation_successful": True
                }
            except Exception as e:
                results["react"] = {"error": str(e), "navigation_successful": False}
        
        return results


@pytest.fixture
def interface_comparator():
    """Provide interface comparator instance."""
    return InterfaceComparator()