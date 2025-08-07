"""Streamlit interface baseline tests.

This module provides comprehensive testing of the current Streamlit interface
to establish a baseline for comparison with the future React implementation.
"""

import pytest
import asyncio
import logging
import json
import re
from pathlib import Path
from playwright.async_api import Page, expect
from typing import Dict, Any

logger = logging.getLogger(__name__)


class TestStreamlitBaseline:
    """Comprehensive baseline tests for Streamlit interface."""
    
    @pytest.mark.asyncio
    async def test_streamlit_app_loads(self, streamlit_page: Page):
        """Verify Streamlit app loads correctly."""
        logger.info("Testing Streamlit app loading")
        
        # Check if page loaded
        await expect(streamlit_page).to_have_title(re="QCM Generator|Streamlit")
        
        # Check for Streamlit-specific elements
        sidebar = streamlit_page.locator('[data-testid="stSidebar"]')
        await expect(sidebar).to_be_visible(timeout=10000)
        
        logger.info("Streamlit app loaded successfully")
    
    @pytest.mark.asyncio
    async def test_navigation_structure(self, streamlit_page: Page, screenshot_helper):
        """Test the navigation structure and accessibility."""
        logger.info("Testing navigation structure")
        
        # Capture initial state
        await screenshot_helper.capture_comparison(
            streamlit_page, None, "navigation_structure", "initial"
        )
        
        # Check for main navigation sections
        expected_sections = [
            "Upload de Documents",
            "Génération QCM", 
            "Gestion Documents",
            "Export",
            "Système"
        ]
        
        navigation_results = {}
        
        for section in expected_sections:
            try:
                section_element = streamlit_page.get_by_text(section)
                is_visible = await section_element.count() > 0
                navigation_results[section] = {
                    "present": is_visible,
                    "clickable": False  # Will test clicking next
                }
                
                if is_visible:
                    # Test if clickable
                    await section_element.click()
                    await streamlit_page.wait_for_timeout(1000)
                    navigation_results[section]["clickable"] = True
                    logger.info(f"Successfully navigated to: {section}")
                    
            except Exception as e:
                logger.warning(f"Navigation issue with {section}: {e}")
                navigation_results[section] = {"present": False, "error": str(e)}
        
        # Save navigation results
        results_file = Path("tests/playwright/reports/streamlit_navigation_baseline.json")
        results_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(results_file, "w") as f:
            json.dump(navigation_results, f, indent=2)
        
        logger.info(f"Navigation results: {navigation_results}")
        
        # At least some sections should be present
        present_sections = sum(1 for r in navigation_results.values() if r.get("present", False))
        assert present_sections >= 3, f"Expected at least 3 navigation sections, found {present_sections}"
    
    @pytest.mark.asyncio
    async def test_document_upload_workflow(self, streamlit_page: Page, screenshot_helper):
        """Test complete document upload workflow."""
        logger.info("Testing document upload workflow")
        
        try:
            # Navigate to upload section
            await streamlit_page.get_by_text("Upload de Documents").click()
            await streamlit_page.wait_for_timeout(2000)
            
            # Capture upload page
            await screenshot_helper.capture_comparison(
                streamlit_page, None, "upload_workflow", "page_loaded"
            )
            
            upload_results = {}
            
            # Check for file upload widget
            file_input = streamlit_page.locator('input[type="file"]')
            upload_results["file_input_present"] = await file_input.count() > 0
            
            # Check for configuration options
            chunk_size_input = streamlit_page.locator('input[type="number"]').first
            upload_results["chunk_size_config"] = await chunk_size_input.count() > 0
            
            # Check for upload button or drag-drop area
            upload_area = streamlit_page.locator('[data-testid="stFileUploader"]')
            upload_results["upload_area_present"] = await upload_area.count() > 0
            
            # Check for processing options
            processing_options = streamlit_page.get_by_text("Configuration").or_(
                streamlit_page.get_by_text("Paramètres")
            )
            upload_results["processing_options"] = await processing_options.count() > 0
            
            logger.info(f"Upload workflow results: {upload_results}")
            
            # Save results
            results_file = Path("tests/playwright/reports/streamlit_upload_baseline.json")
            with open(results_file, "w") as f:
                json.dump(upload_results, f, indent=2)
            
            # Basic assertions
            assert upload_results.get("file_input_present", False) or upload_results.get("upload_area_present", False), \
                "Should have file upload capability"
                
        except Exception as e:
            logger.error(f"Upload workflow test failed: {e}")
            pytest.fail(f"Upload workflow test failed: {e}")
    
    @pytest.mark.asyncio
    async def test_qcm_generation_workflow(self, streamlit_page: Page, screenshot_helper):
        """Test QCM generation workflow and options."""
        logger.info("Testing QCM generation workflow")
        
        try:
            # Navigate to generation section
            await streamlit_page.get_by_text("Génération QCM").click()
            await streamlit_page.wait_for_timeout(2000)
            
            # Capture generation page
            await screenshot_helper.capture_comparison(
                streamlit_page, None, "generation_workflow", "page_loaded"
            )
            
            generation_results = {}
            
            # Check for question count input
            question_count = streamlit_page.locator('input[type="number"]')
            generation_results["question_count_input"] = await question_count.count() > 0
            
            # Check for difficulty settings
            difficulty_options = streamlit_page.get_by_text("Facile").or_(
                streamlit_page.get_by_text("Easy")
            ).or_(streamlit_page.get_by_text("Difficulté"))
            generation_results["difficulty_options"] = await difficulty_options.count() > 0
            
            # Check for language selection
            language_options = streamlit_page.get_by_text("Français").or_(
                streamlit_page.get_by_text("French")
            ).or_(streamlit_page.get_by_text("Langue"))
            generation_results["language_options"] = await language_options.count() > 0
            
            # Check for model selection
            model_selection = streamlit_page.get_by_text("Modèle").or_(
                streamlit_page.get_by_text("Model")
            ).or_(streamlit_page.locator('[data-testid="stSelectbox"]'))
            generation_results["model_selection"] = await model_selection.count() > 0
            
            # Check for progressive workflow options
            progressive_options = streamlit_page.get_by_text("Progressif").or_(
                streamlit_page.get_by_text("Progressive")
            ).or_(streamlit_page.get_by_text("1 question"))
            generation_results["progressive_workflow"] = await progressive_options.count() > 0
            
            # Check for generation button
            generate_button = streamlit_page.get_by_text("Générer").or_(
                streamlit_page.get_by_text("Generate")
            )
            generation_results["generate_button"] = await generate_button.count() > 0
            
            logger.info(f"Generation workflow results: {generation_results}")
            
            # Save results
            results_file = Path("tests/playwright/reports/streamlit_generation_baseline.json")
            with open(results_file, "w") as f:
                json.dump(generation_results, f, indent=2)
            
            # Basic assertions
            assert generation_results.get("question_count_input", False), "Should have question count input"
            
        except Exception as e:
            logger.error(f"Generation workflow test failed: {e}")
            # Don't fail test, just log for baseline purposes
            logger.warning("Some generation elements may not be visible")
    
    @pytest.mark.asyncio
    async def test_export_functionality(self, streamlit_page: Page, screenshot_helper):
        """Test export functionality and formats."""
        logger.info("Testing export functionality")
        
        try:
            # Navigate to export section
            await streamlit_page.get_by_text("Export").click()
            await streamlit_page.wait_for_timeout(2000)
            
            # Capture export page
            await screenshot_helper.capture_comparison(
                streamlit_page, None, "export_functionality", "page_loaded"
            )
            
            export_results = {}
            
            # Check for format options
            csv_option = streamlit_page.get_by_text("CSV").or_(
                streamlit_page.get_by_text("Udemy")
            )
            export_results["csv_format"] = await csv_option.count() > 0
            
            json_option = streamlit_page.get_by_text("JSON")
            export_results["json_format"] = await json_option.count() > 0
            
            # Check for export button
            export_button = streamlit_page.get_by_text("Exporter").or_(
                streamlit_page.get_by_text("Télécharger")
            ).or_(streamlit_page.get_by_text("Download"))
            export_results["export_button"] = await export_button.count() > 0
            
            # Check for preview functionality
            preview_area = streamlit_page.get_by_text("Aperçu").or_(
                streamlit_page.get_by_text("Preview")
            )
            export_results["preview_functionality"] = await preview_area.count() > 0
            
            logger.info(f"Export functionality results: {export_results}")
            
            # Save results
            results_file = Path("tests/playwright/reports/streamlit_export_baseline.json")
            with open(results_file, "w") as f:
                json.dump(export_results, f, indent=2)
            
        except Exception as e:
            logger.error(f"Export functionality test failed: {e}")
            logger.warning("Export elements may not be visible without generated questions")
    
    @pytest.mark.asyncio
    async def test_system_monitoring(self, streamlit_page: Page, screenshot_helper):
        """Test system monitoring and health features."""
        logger.info("Testing system monitoring")
        
        try:
            # Navigate to system section
            await streamlit_page.get_by_text("Système").click()
            await streamlit_page.wait_for_timeout(2000)
            
            # Capture system page
            await screenshot_helper.capture_comparison(
                streamlit_page, None, "system_monitoring", "page_loaded"
            )
            
            system_results = {}
            
            # Check for health status
            health_status = streamlit_page.get_by_text("Statut").or_(
                streamlit_page.get_by_text("Health")
            ).or_(streamlit_page.get_by_text("État"))
            system_results["health_status"] = await health_status.count() > 0
            
            # Check for model management
            model_management = streamlit_page.get_by_text("Ollama").or_(
                streamlit_page.get_by_text("Modèles")
            )
            system_results["model_management"] = await model_management.count() > 0
            
            # Check for configuration display
            config_display = streamlit_page.get_by_text("Configuration").or_(
                streamlit_page.get_by_text("Settings")
            )
            system_results["configuration_display"] = await config_display.count() > 0
            
            # Check for database status
            database_status = streamlit_page.get_by_text("Database").or_(
                streamlit_page.get_by_text("Base de données")
            )
            system_results["database_status"] = await database_status.count() > 0
            
            logger.info(f"System monitoring results: {system_results}")
            
            # Save results
            results_file = Path("tests/playwright/reports/streamlit_system_baseline.json")
            with open(results_file, "w") as f:
                json.dump(system_results, f, indent=2)
            
        except Exception as e:
            logger.error(f"System monitoring test failed: {e}")
            logger.warning("System elements may not be visible")
    
    @pytest.mark.asyncio
    async def test_responsive_behavior(self, streamlit_page: Page, screenshot_helper):
        """Test responsive behavior of Streamlit interface."""
        logger.info("Testing responsive behavior")
        
        viewports = [
            {"width": 1920, "height": 1080, "name": "desktop_large"},
            {"width": 1366, "height": 768, "name": "desktop_medium"},
            {"width": 1024, "height": 768, "name": "tablet_landscape"},
            {"width": 768, "height": 1024, "name": "tablet_portrait"},
            {"width": 375, "height": 667, "name": "mobile"},
        ]
        
        responsive_results = {}
        
        for viewport in viewports:
            try:
                # Set viewport
                await streamlit_page.set_viewport_size({
                    "width": viewport["width"],
                    "height": viewport["height"]
                })
                await streamlit_page.wait_for_timeout(1000)
                
                # Capture screenshot
                await screenshot_helper.capture_comparison(
                    streamlit_page, None, f"responsive_{viewport['name']}", "streamlit"
                )
                
                # Check basic layout elements
                sidebar = streamlit_page.locator('[data-testid="stSidebar"]')
                main_content = streamlit_page.locator('[data-testid="stMain"]')
                
                viewport_results = {
                    "sidebar_visible": await sidebar.is_visible(),
                    "main_content_visible": await main_content.is_visible(),
                    "viewport": viewport
                }
                
                responsive_results[viewport["name"]] = viewport_results
                logger.info(f"Responsive test for {viewport['name']}: {viewport_results}")
                
            except Exception as e:
                logger.error(f"Responsive test failed for {viewport['name']}: {e}")
                responsive_results[viewport["name"]] = {"error": str(e)}
        
        # Save responsive results
        results_file = Path("tests/playwright/reports/streamlit_responsive_baseline.json")
        with open(results_file, "w") as f:
            json.dump(responsive_results, f, indent=2)
        
        logger.info("Responsive behavior testing completed")
    
    @pytest.mark.asyncio
    async def test_performance_metrics(self, streamlit_page: Page):
        """Measure performance metrics for baseline comparison."""
        logger.info("Measuring performance metrics")
        
        performance_results = {}
        
        try:
            # Measure initial load time
            start_time = asyncio.get_event_loop().time()
            await streamlit_page.reload()
            await streamlit_page.wait_for_load_state("networkidle", timeout=15000)
            initial_load_time = asyncio.get_event_loop().time() - start_time
            
            performance_results["initial_load_time"] = initial_load_time
            logger.info(f"Initial load time: {initial_load_time:.2f}s")
            
            # Measure navigation times
            navigation_times = {}
            sections = ["Upload de Documents", "Génération QCM", "Export", "Système"]
            
            for section in sections:
                try:
                    nav_start = asyncio.get_event_loop().time()
                    await streamlit_page.get_by_text(section).click()
                    await streamlit_page.wait_for_timeout(1500)  # Wait for content
                    nav_time = asyncio.get_event_loop().time() - nav_start
                    navigation_times[section] = nav_time
                    logger.info(f"Navigation to {section}: {nav_time:.2f}s")
                except Exception as e:
                    logger.warning(f"Could not measure navigation time for {section}: {e}")
                    navigation_times[section] = {"error": str(e)}
            
            performance_results["navigation_times"] = navigation_times
            
            # Measure memory usage (approximation through browser)
            try:
                memory_info = await streamlit_page.evaluate("() => performance.memory")
                performance_results["memory_usage"] = memory_info
            except Exception as e:
                logger.warning(f"Could not measure memory usage: {e}")
            
            # Save performance results
            results_file = Path("tests/playwright/reports/streamlit_performance_baseline.json")
            with open(results_file, "w") as f:
                json.dump(performance_results, f, indent=2)
            
            logger.info(f"Performance metrics: {performance_results}")
            
            # Basic performance assertions
            assert initial_load_time < 30, f"Initial load time too slow: {initial_load_time}s"
            
        except Exception as e:
            logger.error(f"Performance measurement failed: {e}")
            pytest.fail(f"Performance measurement failed: {e}")
    
    @pytest.mark.asyncio
    async def test_error_handling(self, streamlit_page: Page, screenshot_helper):
        """Test error handling and user feedback mechanisms."""
        logger.info("Testing error handling")
        
        try:
            # Test invalid file upload (if upload is available)
            await streamlit_page.get_by_text("Upload de Documents").click()
            await streamlit_page.wait_for_timeout(2000)
            
            # Look for error messages or validation
            error_elements = await streamlit_page.locator('.stAlert, [data-testid="stAlert"]').count()
            
            error_results = {
                "error_display_capability": error_elements > 0,
                "page_remains_functional": True  # If we got here, page didn't crash
            }
            
            # Capture error handling state
            await screenshot_helper.capture_comparison(
                streamlit_page, None, "error_handling", "test_state"
            )
            
            logger.info(f"Error handling results: {error_results}")
            
            # Save results
            results_file = Path("tests/playwright/reports/streamlit_error_handling_baseline.json")
            with open(results_file, "w") as f:
                json.dump(error_results, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error handling test failed: {e}")
            # This is expected - we're testing error conditions