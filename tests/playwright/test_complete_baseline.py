"""Complete Streamlit baseline capture with navigation screenshots."""

import pytest
import logging
import json
import time
from pathlib import Path
from playwright.async_api import async_playwright

logger = logging.getLogger(__name__)

STREAMLIT_URL = "http://localhost:8501"


@pytest.mark.asyncio
async def test_complete_streamlit_baseline():
    """Complete baseline test with navigation through all sections."""
    logger.info("Starting complete Streamlit baseline test")
    
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        context = await browser.new_context(
            viewport={"width": 1920, "height": 1080},
            record_video_dir="tests/playwright/videos"
        )
        
        try:
            page = await context.new_page()
            await page.goto(STREAMLIT_URL)
            await page.wait_for_selector('[data-testid="stMain"]', timeout=10000)
            await page.wait_for_timeout(3000)  # Wait for full load
            
            # Create output directories
            screenshots_dir = Path("tests/playwright/screenshots")
            reports_dir = Path("tests/playwright/reports")
            screenshots_dir.mkdir(parents=True, exist_ok=True)
            reports_dir.mkdir(parents=True, exist_ok=True)
            
            baseline_results = {
                "timestamp": int(time.time()),
                "test_type": "complete_baseline",
                "url": STREAMLIT_URL,
                "page_title": await page.title(),
                "navigation_sections": {},
                "performance_metrics": {},
                "screenshots": {}
            }
            
            # Capture main page
            main_screenshot = screenshots_dir / "01_main_page.png"
            await page.screenshot(path=str(main_screenshot), full_page=True)
            baseline_results["screenshots"]["main_page"] = str(main_screenshot)
            logger.info("📸 Captured main page screenshot")
            
            # Define navigation sections to test
            nav_sections = [
                {"name": "Upload de Documents", "emoji": "📄", "key": "upload"},
                {"name": "Gestion Documents", "emoji": "📚", "key": "management"}, 
                {"name": "Génération QCM", "emoji": "🎯", "key": "generation"},
                {"name": "Génération par Titre", "emoji": "🏷️", "key": "title_generation"},
                {"name": "Export", "emoji": "📤", "key": "export"},
                {"name": "Système", "emoji": "⚙️", "key": "system"}
            ]
            
            nav_success_count = 0
            
            for i, section in enumerate(nav_sections, 1):
                section_key = section["key"]
                section_name = section["name"]
                
                try:
                    logger.info(f"🧭 Navigating to: {section_name}")
                    
                    # Try to find and click navigation element
                    nav_element = page.get_by_text(section_name, exact=False)
                    element_count = await nav_element.count()
                    
                    if element_count > 0:
                        # Click on the first matching element
                        await nav_element.first.click()
                        await page.wait_for_timeout(2000)  # Wait for navigation
                        
                        # Capture screenshot
                        screenshot_filename = f"{i:02d}_{section_key}.png"
                        screenshot_path = screenshots_dir / screenshot_filename
                        await page.screenshot(path=str(screenshot_path), full_page=True)
                        
                        # Record navigation success
                        baseline_results["navigation_sections"][section_key] = {
                            "name": section_name,
                            "navigable": True,
                            "screenshot": str(screenshot_path),
                            "elements_found": element_count
                        }
                        baseline_results["screenshots"][section_key] = str(screenshot_path)
                        
                        nav_success_count += 1
                        logger.info(f"✅ Successfully navigated to {section_name}")
                        
                        # Check for specific elements in each section
                        await _analyze_section_content(page, section_key, baseline_results)
                        
                    else:
                        baseline_results["navigation_sections"][section_key] = {
                            "name": section_name,
                            "navigable": False,
                            "error": "Navigation element not found",
                            "elements_found": 0
                        }
                        logger.warning(f"❌ Could not find navigation for {section_name}")
                        
                except Exception as e:
                    baseline_results["navigation_sections"][section_key] = {
                        "name": section_name,
                        "navigable": False,
                        "error": str(e),
                        "elements_found": 0
                    }
                    logger.error(f"❌ Navigation failed for {section_name}: {e}")
            
            # Test responsive behavior
            await _test_responsive_viewports(page, screenshots_dir, baseline_results)
            
            # Measure performance
            await _measure_performance(page, baseline_results)
            
            # Final summary
            baseline_results["summary"] = {
                "total_sections": len(nav_sections),
                "successful_navigations": nav_success_count,
                "navigation_success_rate": nav_success_count / len(nav_sections),
                "total_screenshots": len(baseline_results["screenshots"]),
                "test_duration_seconds": int(time.time()) - baseline_results["timestamp"]
            }
            
            # Save comprehensive results
            results_file = reports_dir / "complete_baseline_results.json"
            with open(results_file, "w", encoding="utf-8") as f:
                json.dump(baseline_results, f, indent=2, ensure_ascii=False)
            
            logger.info(f"📊 Baseline test completed!")
            logger.info(f"Navigation success: {nav_success_count}/{len(nav_sections)} sections")
            logger.info(f"Screenshots captured: {len(baseline_results['screenshots'])}")
            logger.info(f"Results saved to: {results_file}")
            
            # Assertions for test validation
            assert nav_success_count > 0, "Should successfully navigate to at least one section"
            assert baseline_results["summary"]["navigation_success_rate"] >= 0.5, \
                "Should successfully navigate to at least 50% of sections"
            assert len(baseline_results["screenshots"]) >= 3, \
                "Should capture at least 3 screenshots"
            
        finally:
            await context.close()
            await browser.close()


async def _analyze_section_content(page, section_key, results):
    """Analyze content specific to each section."""
    try:
        section_analysis = {"elements": {}}
        
        if section_key == "upload":
            # Check for file upload elements
            file_inputs = await page.locator('input[type="file"]').count()
            config_inputs = await page.locator('input[type="number"]').count()
            process_buttons = await page.get_by_text("Traiter").count()
            
            section_analysis["elements"].update({
                "file_inputs": file_inputs,
                "config_inputs": config_inputs,
                "process_buttons": process_buttons
            })
            
        elif section_key == "generation":
            # Check for generation controls
            num_inputs = await page.locator('input[type="number"]').count()
            select_boxes = await page.locator('select, [data-testid*="stSelectbox"]').count()
            generate_buttons = await page.get_by_text("Générer").count()
            
            section_analysis["elements"].update({
                "number_inputs": num_inputs,
                "select_boxes": select_boxes,
                "generate_buttons": generate_buttons
            })
            
        elif section_key == "export":
            # Check for export options
            download_buttons = await page.get_by_text("Télécharger").count()
            format_options = await page.get_by_text("CSV").count() + await page.get_by_text("JSON").count()
            
            section_analysis["elements"].update({
                "download_buttons": download_buttons,
                "format_options": format_options
            })
            
        elif section_key == "system":
            # Check for system status elements
            status_elements = await page.get_by_text("Statut").count()
            model_elements = await page.get_by_text("Ollama").count()
            config_elements = await page.get_by_text("Configuration").count()
            
            section_analysis["elements"].update({
                "status_elements": status_elements,
                "model_elements": model_elements,
                "config_elements": config_elements
            })
        
        # Store analysis in results
        results["navigation_sections"][section_key]["content_analysis"] = section_analysis
        
    except Exception as e:
        logger.warning(f"Content analysis failed for {section_key}: {e}")


async def _test_responsive_viewports(page, screenshots_dir, results):
    """Test responsive behavior across different viewports."""
    viewports = [
        {"width": 1366, "height": 768, "name": "desktop_medium"},
        {"width": 768, "height": 1024, "name": "tablet"},
        {"width": 375, "height": 667, "name": "mobile"}
    ]
    
    responsive_results = {}
    
    for viewport in viewports:
        try:
            await page.set_viewport_size({"width": viewport["width"], "height": viewport["height"]})
            await page.wait_for_timeout(1000)
            
            screenshot_path = screenshots_dir / f"responsive_{viewport['name']}.png"
            await page.screenshot(path=str(screenshot_path))
            
            # Check if key elements are still visible
            sidebar_visible = await page.locator('[data-testid="stSidebar"]').is_visible()
            main_visible = await page.locator('[data-testid="stMain"]').is_visible()
            
            responsive_results[viewport["name"]] = {
                "viewport": viewport,
                "screenshot": str(screenshot_path),
                "sidebar_visible": sidebar_visible,
                "main_visible": main_visible,
                "layout_preserved": sidebar_visible and main_visible
            }
            
            logger.info(f"📱 Responsive test {viewport['name']}: layout_preserved={sidebar_visible and main_visible}")
            
        except Exception as e:
            responsive_results[viewport["name"]] = {"error": str(e)}
            logger.error(f"Responsive test failed for {viewport['name']}: {e}")
    
    results["responsive_behavior"] = responsive_results
    
    # Reset to desktop viewport
    await page.set_viewport_size({"width": 1920, "height": 1080})


async def _measure_performance(page, results):
    """Measure basic performance metrics."""
    try:
        # Measure page reload time
        start_time = time.time()
        await page.reload()
        await page.wait_for_selector('[data-testid="stMain"]', timeout=15000)
        reload_time = time.time() - start_time
        
        # Try to get browser performance metrics
        try:
            perf_metrics = await page.evaluate("""
                () => {
                    if (window.performance && window.performance.getEntriesByType) {
                        const navigation = window.performance.getEntriesByType('navigation')[0];
                        return {
                            loadTime: navigation ? navigation.loadEventEnd - navigation.fetchStart : null,
                            domContentLoaded: navigation ? navigation.domContentLoadedEventEnd - navigation.fetchStart : null,
                            timeToFirstByte: navigation ? navigation.responseStart - navigation.requestStart : null
                        };
                    }
                    return null;
                }
            """)
        except Exception:
            perf_metrics = None
        
        results["performance_metrics"] = {
            "page_reload_time_seconds": reload_time,
            "browser_metrics": perf_metrics
        }
        
        logger.info(f"⚡ Performance: reload={reload_time:.2f}s")
        
    except Exception as e:
        results["performance_metrics"] = {"error": str(e)}
        logger.error(f"Performance measurement failed: {e}")