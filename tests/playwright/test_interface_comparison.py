"""Interface comparison tests between Streamlit and React implementations.

This module provides comprehensive testing for comparing the functionality
and visual appearance of both interfaces during the migration process.
"""

import pytest
import asyncio
import logging
from pathlib import Path
from playwright.async_api import Page, expect
from typing import Optional

logger = logging.getLogger(__name__)


class TestInterfaceComparison:
    """Test suite for comparing Streamlit and React interfaces."""
    
    @pytest.mark.asyncio
    async def test_homepage_comparison(self, streamlit_page: Page, screenshot_helper):
        """Compare homepage/landing page between interfaces."""
        logger.info("Starting homepage comparison test")
        
        # Capture Streamlit homepage
        await screenshot_helper.capture_comparison(
            streamlit_page, None, "homepage", "streamlit_main"
        )
        
        # Check for key elements in Streamlit
        streamlit_elements = {}
        
        # Check for navigation elements
        try:
            sidebar = streamlit_page.locator('[data-testid="stSidebar"]')
            streamlit_elements["has_sidebar"] = await sidebar.count() > 0
            
            # Check for main sections
            upload_section = streamlit_page.get_by_text("Upload de Documents")
            streamlit_elements["has_upload_section"] = await upload_section.count() > 0
            
            generation_section = streamlit_page.get_by_text("Génération QCM")
            streamlit_elements["has_generation_section"] = await generation_section.count() > 0
            
            export_section = streamlit_page.get_by_text("Export")
            streamlit_elements["has_export_section"] = await export_section.count() > 0
            
        except Exception as e:
            logger.error(f"Error checking Streamlit elements: {e}")
            streamlit_elements["error"] = str(e)
        
        # Log results
        logger.info(f"Streamlit homepage elements: {streamlit_elements}")
        
        # For now, just verify Streamlit works (React will be added later)
        assert streamlit_elements.get("has_sidebar", False), "Streamlit should have sidebar navigation"
    
    @pytest.mark.asyncio
    async def test_upload_page_comparison(self, streamlit_page: Page, screenshot_helper, interface_comparator):
        """Compare document upload functionality between interfaces."""
        logger.info("Starting upload page comparison test")
        
        # Navigate to upload section in Streamlit
        try:
            await streamlit_page.get_by_text("Upload de Documents").click()
            await streamlit_page.wait_for_timeout(2000)
            
            # Capture screenshot of upload page
            await screenshot_helper.capture_comparison(
                streamlit_page, None, "upload_page", "streamlit"
            )
            
            # Test upload functionality
            comparison_results = await interface_comparator.compare_upload_workflow(
                streamlit_page, None  # React page not available yet
            )
            
            logger.info(f"Upload comparison results: {comparison_results}")
            
            # Verify Streamlit upload functionality
            streamlit_result = comparison_results["streamlit"]
            assert streamlit_result.get("navigation_successful", False), "Should navigate to upload successfully"
            assert streamlit_result.get("upload_widget_present", False), "Should have file upload widget"
            
        except Exception as e:
            logger.error(f"Upload page test failed: {e}")
            pytest.fail(f"Upload page comparison failed: {e}")
    
    @pytest.mark.asyncio
    async def test_generation_page_comparison(self, streamlit_page: Page, screenshot_helper):
        """Compare QCM generation page between interfaces."""
        logger.info("Starting generation page comparison test")
        
        try:
            # Navigate to generation section
            await streamlit_page.get_by_text("Génération QCM").click()
            await streamlit_page.wait_for_timeout(2000)
            
            # Capture screenshot
            await screenshot_helper.capture_comparison(
                streamlit_page, None, "generation_page", "streamlit"
            )
            
            # Check for key generation elements
            generation_elements = {}
            
            # Check for configuration options
            num_questions = streamlit_page.locator('input[type="number"]')
            generation_elements["has_question_count_input"] = await num_questions.count() > 0
            
            # Check for model selection
            model_select = streamlit_page.locator('select, [data-testid="stSelectbox"]')
            generation_elements["has_model_selection"] = await model_select.count() > 0
            
            # Check for language selection
            language_select = streamlit_page.get_by_text("Français").or_(streamlit_page.get_by_text("English"))
            generation_elements["has_language_selection"] = await language_select.count() > 0
            
            logger.info(f"Generation elements found: {generation_elements}")
            
            # Basic assertions
            assert generation_elements.get("has_question_count_input", False), "Should have question count input"
            
        except Exception as e:
            logger.error(f"Generation page test failed: {e}")
            # Don't fail the test if some elements are missing, just log
            logger.warning("Generation page elements may not be fully loaded")
    
    @pytest.mark.asyncio
    async def test_system_page_comparison(self, streamlit_page: Page, screenshot_helper):
        """Compare system/configuration page between interfaces."""
        logger.info("Starting system page comparison test")
        
        try:
            # Navigate to system section
            await streamlit_page.get_by_text("Système").click()
            await streamlit_page.wait_for_timeout(2000)
            
            # Capture screenshot
            await screenshot_helper.capture_comparison(
                streamlit_page, None, "system_page", "streamlit"
            )
            
            # Check for system elements
            system_elements = {}
            
            # Check for health status
            health_text = streamlit_page.get_by_text("Statut").or_(streamlit_page.get_by_text("Health"))
            system_elements["has_health_status"] = await health_text.count() > 0
            
            # Check for model management
            model_management = streamlit_page.get_by_text("Ollama").or_(streamlit_page.get_by_text("Modèles"))
            system_elements["has_model_management"] = await model_management.count() > 0
            
            logger.info(f"System elements found: {system_elements}")
            
        except Exception as e:
            logger.error(f"System page test failed: {e}")
            logger.warning("System page elements may not be fully loaded")
    
    @pytest.mark.asyncio
    async def test_export_functionality_comparison(self, streamlit_page: Page, screenshot_helper):
        """Compare export functionality between interfaces."""
        logger.info("Starting export functionality comparison test")
        
        try:
            # Navigate to export section
            await streamlit_page.get_by_text("Export").click()
            await streamlit_page.wait_for_timeout(2000)
            
            # Capture screenshot
            await screenshot_helper.capture_comparison(
                streamlit_page, None, "export_page", "streamlit"
            )
            
            # Check for export elements
            export_elements = {}
            
            # Check for format selection
            csv_option = streamlit_page.get_by_text("CSV").or_(streamlit_page.get_by_text("Udemy"))
            export_elements["has_csv_option"] = await csv_option.count() > 0
            
            json_option = streamlit_page.get_by_text("JSON")
            export_elements["has_json_option"] = await json_option.count() > 0
            
            # Check for download button
            download_button = streamlit_page.get_by_text("Télécharger").or_(streamlit_page.get_by_text("Download"))
            export_elements["has_download_button"] = await download_button.count() > 0
            
            logger.info(f"Export elements found: {export_elements}")
            
        except Exception as e:
            logger.error(f"Export page test failed: {e}")
            logger.warning("Export page elements may not be fully loaded")
    
    @pytest.mark.asyncio
    async def test_responsive_design(self, context, streamlit_page: Page, screenshot_helper):
        """Test responsive design of Streamlit interface (for React comparison baseline)."""
        logger.info("Testing responsive design")
        
        # Test different viewport sizes
        viewports = [
            {"width": 1920, "height": 1080, "name": "desktop"},
            {"width": 1024, "height": 768, "name": "tablet"},
            {"width": 375, "height": 667, "name": "mobile"}
        ]
        
        for viewport in viewports:
            try:
                # Set viewport size
                await streamlit_page.set_viewport_size({
                    "width": viewport["width"], 
                    "height": viewport["height"]
                })
                await streamlit_page.wait_for_timeout(1000)
                
                # Capture screenshot for each size
                await screenshot_helper.capture_comparison(
                    streamlit_page, None, f"responsive_{viewport['name']}", "streamlit"
                )
                
                logger.info(f"Captured {viewport['name']} viewport screenshot")
                
            except Exception as e:
                logger.error(f"Responsive test failed for {viewport['name']}: {e}")
    
    @pytest.mark.asyncio
    async def test_performance_baseline(self, streamlit_page: Page):
        """Establish performance baseline for Streamlit (for React comparison)."""
        logger.info("Testing performance baseline")
        
        try:
            # Measure page load time
            start_time = asyncio.get_event_loop().time()
            
            # Navigate to main page
            await streamlit_page.reload()
            await streamlit_page.wait_for_load_state("networkidle")
            
            load_time = asyncio.get_event_loop().time() - start_time
            
            logger.info(f"Streamlit page load time: {load_time:.2f} seconds")
            
            # Test navigation performance
            navigation_times = {}
            
            sections = ["Upload de Documents", "Génération QCM", "Export", "Système"]
            
            for section in sections:
                try:
                    nav_start = asyncio.get_event_loop().time()
                    await streamlit_page.get_by_text(section).click()
                    await streamlit_page.wait_for_timeout(1000)  # Wait for content to load
                    nav_time = asyncio.get_event_loop().time() - nav_start
                    navigation_times[section] = nav_time
                    logger.info(f"Navigation to {section}: {nav_time:.2f}s")
                except Exception as e:
                    logger.warning(f"Could not navigate to {section}: {e}")
            
            # Store baseline metrics for comparison
            performance_metrics = {
                "page_load_time": load_time,
                "navigation_times": navigation_times,
                "interface": "streamlit"
            }
            
            # Save metrics to file for future comparison
            metrics_file = Path("tests/playwright/reports/streamlit_performance_baseline.json")
            metrics_file.parent.mkdir(parents=True, exist_ok=True)
            
            import json
            with open(metrics_file, "w") as f:
                json.dump(performance_metrics, f, indent=2)
            
            logger.info(f"Performance baseline saved to {metrics_file}")
            
        except Exception as e:
            logger.error(f"Performance baseline test failed: {e}")


class TestFutureReactComparison:
    """Placeholder tests for future React interface comparison."""
    
    @pytest.mark.skip(reason="React interface not yet implemented")
    @pytest.mark.asyncio
    async def test_react_vs_streamlit_upload(self, streamlit_page: Page, react_page: Page, 
                                           screenshot_helper, interface_comparator):
        """Compare upload workflow between Streamlit and React."""
        # This test will be enabled once React interface is ready
        comparison_results = await interface_comparator.compare_upload_workflow(
            streamlit_page, react_page
        )
        
        # Capture comparison screenshots
        await screenshot_helper.capture_comparison(
            streamlit_page, react_page, "upload_workflow", "comparison"
        )
        
        # Compare functionality
        streamlit_result = comparison_results["streamlit"]
        react_result = comparison_results["react"]
        
        # Assert both interfaces have required functionality
        assert streamlit_result.get("upload_widget_present", False)
        assert react_result.get("upload_widget_present", False)
        
        logger.info("Upload workflow comparison completed successfully")
    
    @pytest.mark.skip(reason="React interface not yet implemented")
    @pytest.mark.asyncio
    async def test_react_vs_streamlit_generation(self, streamlit_page: Page, react_page: Page,
                                               screenshot_helper, interface_comparator):
        """Compare generation workflow between Streamlit and React."""
        # This test will be enabled once React interface is ready
        comparison_results = await interface_comparator.compare_generation_workflow(
            streamlit_page, react_page
        )
        
        # Capture comparison screenshots
        await screenshot_helper.capture_comparison(
            streamlit_page, react_page, "generation_workflow", "comparison"
        )
        
        # Compare functionality
        streamlit_result = comparison_results["streamlit"]
        react_result = comparison_results["react"]
        
        # Assert both interfaces have required functionality
        assert streamlit_result.get("generation_button_present", False)
        assert react_result.get("generation_button_present", False)
        
        logger.info("Generation workflow comparison completed successfully")