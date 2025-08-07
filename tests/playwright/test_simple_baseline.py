"""Simple Playwright test to verify setup and capture baseline screenshots."""

import pytest
import logging
import json
import time
from pathlib import Path
from playwright.async_api import async_playwright

logger = logging.getLogger(__name__)

STREAMLIT_URL = "http://localhost:8501"


@pytest.mark.asyncio
async def test_simple_streamlit_capture():
    """Simple test to capture Streamlit interface screenshots."""
    logger.info("Starting simple Streamlit capture test")
    
    async with async_playwright() as p:
        # Launch browser
        browser = await p.chromium.launch(headless=False)
        context = await browser.new_context(
            viewport={"width": 1920, "height": 1080}
        )
        
        try:
            # Create page and navigate
            page = await context.new_page()
            
            logger.info(f"Navigating to {STREAMLIT_URL}")
            await page.goto(STREAMLIT_URL)
            await page.wait_for_timeout(3000)  # Wait for page to load
            
            # Create screenshots directory
            screenshots_dir = Path("tests/playwright/screenshots")
            screenshots_dir.mkdir(parents=True, exist_ok=True)
            
            # Capture main page screenshot
            screenshot_path = screenshots_dir / "streamlit_main_page.png"
            await page.screenshot(path=str(screenshot_path), full_page=True)
            logger.info(f"Screenshot captured: {screenshot_path}")
            
            # Check for key elements
            elements_found = {}
            
            # Check sidebar
            try:
                sidebar = page.locator('[data-testid="stSidebar"]')
                elements_found["sidebar"] = await sidebar.count() > 0
                logger.info(f"Sidebar found: {elements_found['sidebar']}")
            except Exception as e:
                elements_found["sidebar"] = False
                logger.warning(f"Sidebar check failed: {e}")
            
            # Check main content
            try:
                main_content = page.locator('[data-testid="stMain"]')
                elements_found["main_content"] = await main_content.count() > 0
                logger.info(f"Main content found: {elements_found['main_content']}")
            except Exception as e:
                elements_found["main_content"] = False
                logger.warning(f"Main content check failed: {e}")
            
            # Check for navigation elements
            nav_elements = [
                "Upload de Documents",
                "Génération QCM", 
                "Export",
                "Système"
            ]
            
            for nav_text in nav_elements:
                try:
                    element = page.get_by_text(nav_text)
                    found = await element.count() > 0
                    elements_found[f"nav_{nav_text}"] = found
                    logger.info(f"Navigation '{nav_text}': {found}")
                except Exception as e:
                    elements_found[f"nav_{nav_text}"] = False
                    logger.warning(f"Navigation '{nav_text}' check failed: {e}")
            
            # Save results
            results_dir = Path("tests/playwright/reports")
            results_dir.mkdir(parents=True, exist_ok=True)
            
            results = {
                "test": "simple_streamlit_capture",
                "timestamp": int(time.time()),
                "url": STREAMLIT_URL,
                "screenshot": str(screenshot_path),
                "elements_found": elements_found,
                "total_elements": sum(1 for found in elements_found.values() if found)
            }
            
            results_file = results_dir / "simple_baseline_results.json"
            with open(results_file, "w") as f:
                json.dump(results, f, indent=2)
            
            logger.info(f"Results saved to {results_file}")
            logger.info(f"Elements found: {results['total_elements']}/{len(elements_found)}")
            
            # Basic assertions
            assert elements_found["sidebar"] or elements_found["main_content"], \
                "Should find either sidebar or main content"
            
            assert results["total_elements"] > 0, \
                "Should find at least one UI element"
            
            logger.info("✅ Simple baseline test completed successfully!")
            
        finally:
            await context.close()
            await browser.close()


@pytest.mark.asyncio 
async def test_streamlit_navigation():
    """Test navigation between different sections."""
    logger.info("Testing Streamlit navigation")
    
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        context = await browser.new_context(viewport={"width": 1920, "height": 1080})
        
        try:
            page = await context.new_page()
            await page.goto(STREAMLIT_URL)
            await page.wait_for_timeout(3000)
            
            screenshots_dir = Path("tests/playwright/screenshots")
            screenshots_dir.mkdir(parents=True, exist_ok=True)
            
            # Test navigation to different sections
            sections = ["Upload de Documents", "Génération QCM", "Export", "Système"]
            navigation_results = {}
            
            for section in sections:
                try:
                    logger.info(f"Testing navigation to: {section}")
                    
                    # Try to click on section
                    element = page.get_by_text(section)
                    if await element.count() > 0:
                        await element.click()
                        await page.wait_for_timeout(2000)
                        
                        # Capture screenshot
                        screenshot_name = f"nav_{section.lower().replace(' ', '_')}.png"
                        screenshot_path = screenshots_dir / screenshot_name
                        await page.screenshot(path=str(screenshot_path))
                        
                        navigation_results[section] = {
                            "clickable": True,
                            "screenshot": str(screenshot_path)
                        }
                        logger.info(f"✅ Successfully navigated to {section}")
                    else:
                        navigation_results[section] = {
                            "clickable": False,
                            "error": "Element not found"
                        }
                        logger.warning(f"❌ Could not find {section}")
                        
                except Exception as e:
                    navigation_results[section] = {
                        "clickable": False,
                        "error": str(e)
                    }
                    logger.error(f"❌ Navigation to {section} failed: {e}")
            
            # Save navigation results
            results_dir = Path("tests/playwright/reports")
            results_dir.mkdir(parents=True, exist_ok=True)
            
            nav_results = {
                "test": "streamlit_navigation",
                "timestamp": int(time.time()),
                "navigation_results": navigation_results,
                "successful_navigations": sum(1 for r in navigation_results.values() if r.get("clickable", False))
            }
            
            nav_file = results_dir / "navigation_results.json"
            with open(nav_file, "w") as f:
                json.dump(nav_results, f, indent=2)
            
            logger.info(f"Navigation results: {nav_results['successful_navigations']}/{len(sections)} sections")
            logger.info(f"Results saved to {nav_file}")
            
            # At least one section should be navigable
            assert nav_results["successful_navigations"] > 0, \
                "Should be able to navigate to at least one section"
            
        finally:
            await context.close()
            await browser.close()