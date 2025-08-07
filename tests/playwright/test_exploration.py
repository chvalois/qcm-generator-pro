"""Exploratory test to understand Streamlit interface structure."""

import pytest
import logging
import json
import time
from pathlib import Path
from playwright.async_api import async_playwright

logger = logging.getLogger(__name__)

STREAMLIT_URL = "http://localhost:8501"


@pytest.mark.asyncio
async def test_explore_streamlit_structure():
    """Explore Streamlit interface to understand its structure."""
    logger.info("Exploring Streamlit interface structure")
    
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        context = await browser.new_context(viewport={"width": 1920, "height": 1080})
        
        try:
            page = await context.new_page()
            await page.goto(STREAMLIT_URL)
            await page.wait_for_timeout(5000)  # Wait longer for full load
            
            # Create output directories
            screenshots_dir = Path("tests/playwright/screenshots")
            reports_dir = Path("tests/playwright/reports")
            screenshots_dir.mkdir(parents=True, exist_ok=True)
            reports_dir.mkdir(parents=True, exist_ok=True)
            
            # Take full page screenshot
            await page.screenshot(path="tests/playwright/screenshots/exploration_full.png", full_page=True)
            
            exploration_results = {
                "timestamp": int(time.time()),
                "url": STREAMLIT_URL,
                "page_title": await page.title(),
                "elements_found": {}
            }
            
            logger.info(f"Page title: {exploration_results['page_title']}")
            
            # Test various Streamlit selectors
            streamlit_selectors = [
                '[data-testid="stSidebar"]',
                '[data-testid="stMain"]', 
                '[data-testid="stHeader"]',
                '.main',
                '.sidebar',
                'div[data-testid]',
                '.stSelectbox',
                '.stButton',
                'button',
                'select',
                'input'
            ]
            
            for selector in streamlit_selectors:
                try:
                    elements = page.locator(selector)
                    count = await elements.count()
                    exploration_results["elements_found"][selector] = count
                    logger.info(f"Selector '{selector}': {count} elements")
                    
                    if count > 0:
                        # Get text content of first few elements
                        for i in range(min(3, count)):
                            try:
                                text_content = await elements.nth(i).text_content()
                                if text_content and text_content.strip():
                                    exploration_results["elements_found"][f"{selector}_text_{i}"] = text_content.strip()[:100]
                            except Exception:
                                pass
                                
                except Exception as e:
                    exploration_results["elements_found"][selector] = f"Error: {str(e)}"
                    logger.warning(f"Error with selector '{selector}': {e}")
            
            # Look for all clickable elements
            try:
                clickable_elements = page.locator('button, a, [role="button"], [onclick]')
                clickable_count = await clickable_elements.count()
                exploration_results["clickable_elements_count"] = clickable_count
                logger.info(f"Found {clickable_count} potentially clickable elements")
                
                # Get text of clickable elements
                clickable_texts = []
                for i in range(min(10, clickable_count)):
                    try:
                        text = await clickable_elements.nth(i).text_content()
                        if text and text.strip():
                            clickable_texts.append(text.strip())
                    except Exception:
                        pass
                
                exploration_results["clickable_texts"] = clickable_texts
                logger.info(f"Clickable element texts: {clickable_texts}")
                
            except Exception as e:
                exploration_results["clickable_error"] = str(e)
            
            # Look for text containing navigation keywords
            nav_keywords = ["upload", "génération", "export", "système", "documents", "qcm"]
            for keyword in nav_keywords:
                try:
                    # Case-insensitive search
                    elements_with_keyword = page.get_by_text(keyword, exact=False)
                    count = await elements_with_keyword.count()
                    exploration_results[f"keyword_{keyword}_count"] = count
                    
                    if count > 0:
                        # Get surrounding context
                        for i in range(min(3, count)):
                            try:
                                element = elements_with_keyword.nth(i)
                                text = await element.text_content()
                                parent_text = await element.locator('..').text_content()
                                exploration_results[f"keyword_{keyword}_{i}"] = {
                                    "text": text[:50] if text else None,
                                    "parent": parent_text[:100] if parent_text else None
                                }
                            except Exception:
                                pass
                                
                except Exception as e:
                    exploration_results[f"keyword_{keyword}_error"] = str(e)
            
            # Get page HTML structure (truncated)
            try:
                html_content = await page.content()
                exploration_results["html_length"] = len(html_content)
                exploration_results["html_sample"] = html_content[:1000]  # First 1000 chars
            except Exception as e:
                exploration_results["html_error"] = str(e)
            
            # Save exploration results
            with open(reports_dir / "exploration_results.json", "w", encoding="utf-8") as f:
                json.dump(exploration_results, f, indent=2, ensure_ascii=False)
            
            logger.info("Exploration completed - results saved to exploration_results.json")
            
            # Basic assertions
            assert exploration_results["page_title"] is not None, "Should have page title"
            assert exploration_results["elements_found"].get('[data-testid="stMain"]', 0) > 0, \
                "Should find main Streamlit content"
                
        finally:
            await context.close()
            await browser.close()


@pytest.mark.asyncio
async def test_streamlit_wait_and_interact():
    """Test waiting for Streamlit to fully load and then interact."""
    logger.info("Testing Streamlit interaction after full load")
    
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        context = await browser.new_context(viewport={"width": 1920, "height": 1080})
        
        try:
            page = await context.new_page()
            await page.goto(STREAMLIT_URL)
            
            # Wait for Streamlit to fully initialize
            logger.info("Waiting for Streamlit to fully load...")
            
            # Wait for main content
            await page.wait_for_selector('[data-testid="stMain"]', timeout=10000)
            
            # Wait additional time for dynamic content
            await page.wait_for_timeout(3000)
            
            # Take screenshot after full load
            await page.screenshot(path="tests/playwright/screenshots/after_full_load.png", full_page=True)
            
            # Try to find any form elements
            form_elements = []
            
            # Look for file upload
            file_inputs = page.locator('input[type="file"]')
            file_count = await file_inputs.count()
            if file_count > 0:
                form_elements.append(f"File inputs: {file_count}")
                logger.info(f"Found {file_count} file input elements")
            
            # Look for buttons
            buttons = page.locator('button')
            button_count = await buttons.count()
            if button_count > 0:
                form_elements.append(f"Buttons: {button_count}")
                logger.info(f"Found {button_count} button elements")
                
                # Get button texts
                for i in range(min(5, button_count)):
                    try:
                        btn_text = await buttons.nth(i).text_content()
                        if btn_text and btn_text.strip():
                            logger.info(f"Button {i}: '{btn_text.strip()}'")
                    except Exception:
                        pass
            
            # Look for selectboxes
            selects = page.locator('select, [data-testid*="stSelectbox"]')
            select_count = await selects.count()
            if select_count > 0:
                form_elements.append(f"Select elements: {select_count}")
                logger.info(f"Found {select_count} select elements")
            
            # Look for text inputs
            text_inputs = page.locator('input[type="text"], input[type="number"], textarea')
            input_count = await text_inputs.count()
            if input_count > 0:
                form_elements.append(f"Text inputs: {input_count}")
                logger.info(f"Found {input_count} text input elements")
            
            # Save interaction results
            interaction_results = {
                "timestamp": int(time.time()),
                "form_elements_found": form_elements,
                "total_interactive_elements": file_count + button_count + select_count + input_count
            }
            
            reports_dir = Path("tests/playwright/reports")
            with open(reports_dir / "interaction_results.json", "w") as f:
                json.dump(interaction_results, f, indent=2)
            
            logger.info(f"Interactive elements found: {interaction_results['total_interactive_elements']}")
            logger.info("Interaction test completed")
            
            # Basic assertion
            assert interaction_results["total_interactive_elements"] >= 0, \
                "Should complete interaction analysis"
                
        finally:
            await context.close()
            await browser.close()