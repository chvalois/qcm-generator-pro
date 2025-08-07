#!/usr/bin/env python3
"""Test runner script for Playwright interface comparison tests.

This script provides convenient ways to run various test suites and
generate comprehensive reports for the Streamlit vs React migration.
"""

import subprocess
import sys
import time
import requests
import argparse
import logging
from pathlib import Path
from typing import List, Optional
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Service configuration
SERVICES = {
    "api": "http://localhost:8000/health",
    "streamlit": "http://localhost:8501",
    "react": "http://localhost:3000"  # Future
}

TIMEOUT = 30  # seconds


def check_service(name: str, url: str, timeout: int = 5) -> bool:
    """Check if a service is responding."""
    try:
        response = requests.get(url, timeout=timeout)
        if response.status_code == 200:
            logger.info(f"✅ {name.title()} service is running")
            return True
        else:
            logger.warning(f"⚠️  {name.title()} service returned {response.status_code}")
            return False
    except requests.RequestException as e:
        logger.warning(f"❌ {name.title()} service not available: {e}")
        return False


def wait_for_services(required_services: List[str], timeout: int = TIMEOUT) -> bool:
    """Wait for required services to become available."""
    logger.info(f"Waiting for services: {', '.join(required_services)}")
    
    start_time = time.time()
    while time.time() - start_time < timeout:
        all_ready = True
        for service_name in required_services:
            if service_name not in SERVICES:
                logger.error(f"Unknown service: {service_name}")
                return False
            
            if not check_service(service_name, SERVICES[service_name]):
                all_ready = False
                break
        
        if all_ready:
            logger.info("🎉 All required services are ready!")
            return True
        
        logger.info("⏳ Waiting for services...")
        time.sleep(2)
    
    logger.error(f"❌ Services not ready after {timeout} seconds")
    return False


def run_pytest(test_path: str, extra_args: List[str] = None) -> int:
    """Run pytest with specified arguments."""
    if extra_args is None:
        extra_args = []
    
    # Base pytest arguments
    base_args = [
        "python", "-m", "pytest",
        test_path,
        "-v",
        "--tb=short",
        "--asyncio-mode=auto"
    ]
    
    # Add HTML report
    reports_dir = Path("tests/playwright/reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    html_report = reports_dir / f"report_{int(time.time())}.html"
    base_args.extend(["--html", str(html_report), "--self-contained-html"])
    
    # Add extra arguments
    base_args.extend(extra_args)
    
    logger.info(f"Running: {' '.join(base_args)}")
    
    try:
        result = subprocess.run(base_args, cwd=Path.cwd())
        
        if result.returncode == 0:
            logger.info(f"✅ Tests completed successfully! Report: {html_report}")
        else:
            logger.error(f"❌ Tests failed with return code {result.returncode}")
            
        return result.returncode
        
    except subprocess.SubprocessError as e:
        logger.error(f"Error running tests: {e}")
        return 1


def run_baseline_tests(args):
    """Run Streamlit baseline tests."""
    logger.info("🚀 Running Streamlit baseline tests")
    
    # For baseline tests, we only need Streamlit (API not required)
    if not wait_for_services(["streamlit"]):
        logger.error("Required services not available for baseline tests")
        return 1
    
    extra_args = []
    if args.headed:
        extra_args.append("--headed")
    if args.slow:
        extra_args.extend(["-m", "not slow"])
    
    return run_pytest("tests/playwright/test_streamlit_baseline.py", extra_args)


def run_comparison_tests(args):
    """Run interface comparison tests."""
    logger.info("🚀 Running interface comparison tests")
    
    required_services = ["api", "streamlit"]
    if args.include_react:
        required_services.append("react")
    
    if not wait_for_services(required_services):
        logger.error("Required services not available for comparison tests")
        return 1
    
    extra_args = []
    if args.headed:
        extra_args.append("--headed")
    if not args.include_react:
        extra_args.extend(["-m", "not react"])
    
    return run_pytest("tests/playwright/test_interface_comparison.py", extra_args)


def run_all_tests(args):
    """Run all available tests."""
    logger.info("🚀 Running all Playwright tests")
    
    required_services = ["api", "streamlit"]
    if args.include_react:
        required_services.append("react")
    
    if not wait_for_services(required_services):
        logger.error("Required services not available")
        return 1
    
    extra_args = []
    if args.headed:
        extra_args.append("--headed")
    if args.slow:
        extra_args.extend(["-m", "not slow"])
    if not args.include_react:
        extra_args.extend(["-k", "not react"])
    
    return run_pytest("tests/playwright/", extra_args)


def generate_comparison_report():
    """Generate a comparison report from test results."""
    logger.info("📊 Generating comparison report")
    
    reports_dir = Path("tests/playwright/reports")
    
    # Collect all baseline JSON files
    baseline_files = list(reports_dir.glob("*_baseline.json"))
    
    if not baseline_files:
        logger.warning("No baseline files found for report generation")
        return
    
    report = {
        "timestamp": int(time.time()),
        "baseline_data": {},
        "summary": {}
    }
    
    for file in baseline_files:
        try:
            with open(file, 'r') as f:
                data = json.load(f)
                report["baseline_data"][file.stem] = data
        except Exception as e:
            logger.error(f"Error reading {file}: {e}")
    
    # Generate summary
    navigation_data = report["baseline_data"].get("streamlit_navigation_baseline", {})
    performance_data = report["baseline_data"].get("streamlit_performance_baseline", {})
    
    report["summary"] = {
        "navigation_sections_working": sum(
            1 for section_data in navigation_data.values() 
            if isinstance(section_data, dict) and section_data.get("present", False)
        ),
        "average_navigation_time": (
            sum(performance_data.get("navigation_times", {}).values()) / 
            len(performance_data.get("navigation_times", {}))
            if performance_data.get("navigation_times") else 0
        ),
        "initial_load_time": performance_data.get("initial_load_time", 0)
    }
    
    # Save comprehensive report
    report_file = reports_dir / f"comparison_report_{int(time.time())}.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"📊 Comparison report saved: {report_file}")
    
    # Print summary
    print("\n" + "="*50)
    print("🎯 BASELINE TEST SUMMARY")
    print("="*50)
    print(f"Navigation sections working: {report['summary']['navigation_sections_working']}")
    print(f"Average navigation time: {report['summary']['average_navigation_time']:.2f}s")
    print(f"Initial load time: {report['summary']['initial_load_time']:.2f}s")
    print(f"Full report: {report_file}")
    print("="*50)


def clean_test_artifacts():
    """Clean old test artifacts."""
    logger.info("🧹 Cleaning test artifacts")
    
    directories = [
        "tests/playwright/screenshots",
        "tests/playwright/videos", 
        "tests/playwright/reports"
    ]
    
    for dir_path in directories:
        dir_obj = Path(dir_path)
        if dir_obj.exists():
            for file in dir_obj.iterdir():
                if file.is_file():
                    file.unlink()
                    logger.info(f"Deleted: {file}")
    
    logger.info("✅ Cleanup completed")


def main():
    """Main test runner."""
    parser = argparse.ArgumentParser(
        description="Playwright test runner for QCM Generator Pro",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Baseline tests
    baseline_parser = subparsers.add_parser('baseline', help='Run Streamlit baseline tests')
    baseline_parser.add_argument('--headed', action='store_true', help='Run with browser visible')
    baseline_parser.add_argument('--slow', action='store_true', help='Skip slow tests')
    
    # Comparison tests
    comparison_parser = subparsers.add_parser('comparison', help='Run interface comparison tests')
    comparison_parser.add_argument('--headed', action='store_true', help='Run with browser visible')
    comparison_parser.add_argument('--include-react', action='store_true', help='Include React tests (if available)')
    
    # All tests
    all_parser = subparsers.add_parser('all', help='Run all tests')
    all_parser.add_argument('--headed', action='store_true', help='Run with browser visible')
    all_parser.add_argument('--slow', action='store_true', help='Skip slow tests')
    all_parser.add_argument('--include-react', action='store_true', help='Include React tests (if available)')
    
    # Utilities
    subparsers.add_parser('report', help='Generate comparison report from existing results')
    subparsers.add_parser('clean', help='Clean test artifacts')
    subparsers.add_parser('check-services', help='Check if services are running')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    try:
        if args.command == 'baseline':
            return run_baseline_tests(args)
        elif args.command == 'comparison':
            return run_comparison_tests(args)
        elif args.command == 'all':
            return run_all_tests(args)
        elif args.command == 'report':
            generate_comparison_report()
            return 0
        elif args.command == 'clean':
            clean_test_artifacts()
            return 0
        elif args.command == 'check-services':
            all_available = True
            for name, url in SERVICES.items():
                if not check_service(name, url):
                    all_available = False
            return 0 if all_available else 1
        else:
            logger.error(f"Unknown command: {args.command}")
            return 1
            
    except KeyboardInterrupt:
        logger.info("\n⏹️  Tests interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())