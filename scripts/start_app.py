#!/usr/bin/env python3
"""
QCM Generator Pro - Application Startup Script

This script starts both the FastAPI backend and Streamlit frontend
for local development and production use.
"""

import argparse
import asyncio
import logging
import multiprocessing
import signal
import sys
import time
from pathlib import Path
from typing import Optional

# Add both src and root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

# Verify the path is correct
src_path = project_root / "src"
if not src_path.exists():
    print(f"ERROR: src directory not found at {src_path}")
    sys.exit(1)

logger = logging.getLogger(__name__)


def setup_logging(debug: bool = False) -> None:
    """Setup logging configuration."""
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )


def start_fastapi_server(
    host: str = "127.0.0.1",
    port: int = 8000,
    debug: bool = False
) -> None:
    """
    Start FastAPI server in a subprocess.
    
    Args:
        host: Server host
        port: Server port
        debug: Debug mode
    """
    import uvicorn
    
    logger.info(f"Starting FastAPI server on {host}:{port}")
    
    try:
        uvicorn.run(
            "src.api.main:app",
            host=host,
            port=port,
            reload=debug,
            log_level="debug" if debug else "info",
            workers=1
        )
    except Exception as e:
        logger.error(f"FastAPI server failed to start: {e}")
        raise


def start_streamlit_app(
    host: str = "127.0.0.1",
    port: int = 8501,
    debug: bool = False
) -> None:
    """
    Start Streamlit application.
    
    Args:
        host: Server host
        port: Server port
        debug: Debug mode
    """
    import subprocess
    import os
    
    logger.info(f"Starting Streamlit interface on {host}:{port}")
    
    try:
        # Wait for FastAPI to be ready
        time.sleep(3)
        
        # Set environment variables for Streamlit
        env = os.environ.copy()
        env["STREAMLIT_SERVER_ADDRESS"] = host
        env["STREAMLIT_SERVER_PORT"] = str(port)
        
        # Add Python path for src module imports
        current_pythonpath = env.get("PYTHONPATH", "")
        new_paths = [str(project_root), str(project_root / "src")]
        if current_pythonpath:
            new_paths.append(current_pythonpath)
        env["PYTHONPATH"] = os.pathsep.join(new_paths)
        
        # Build Streamlit command
        cmd = [
            "streamlit",
            "run",
            "main_app.py",
            "--server.address", host,
            "--server.port", str(port),
            "--server.headless", "true",
            "--browser.gatherUsageStats", "false"
        ]
        
        if debug:
            cmd.extend(["--logger.level", "debug"])
        
        # Start Streamlit process
        subprocess.run(cmd, env=env, cwd=project_root)
        
    except Exception as e:
        logger.error(f"Streamlit interface failed to start: {e}")
        raise


def check_dependencies() -> bool:
    """
    Check if required dependencies are installed.
    
    Returns:
        True if all dependencies are available
    """
    required_packages = [
        ("fastapi", "FastAPI"),
        ("uvicorn", "Uvicorn"),
        ("streamlit", "Streamlit"),
        ("sqlalchemy", "SQLAlchemy"),
        ("pydantic", "Pydantic")
    ]
    
    missing_packages = []
    
    for package, name in required_packages:
        try:
            __import__(package)
            logger.debug(f"✓ {name} is available")
        except ImportError:
            missing_packages.append(name)
            logger.error(f"✗ {name} is not installed")
    
    if missing_packages:
        logger.error(f"Missing required packages: {', '.join(missing_packages)}")
        logger.info("Install with: pip install -e .[ui,full]")
        return False
    
    return True


def signal_handler(signum: int, frame) -> None:
    """Handle shutdown signals."""
    logger.info(f"Received signal {signum}, shutting down...")
    sys.exit(0)


def main() -> None:
    """Main application entry point."""
    parser = argparse.ArgumentParser(
        description="QCM Generator Pro - Start both API and UI"
    )
    
    parser.add_argument(
        "--api-host",
        default="127.0.0.1",
        help="FastAPI server host (default: 127.0.0.1)"
    )
    parser.add_argument(
        "--api-port",
        type=int,
        default=8001,
        help="FastAPI server port (default: 8001)"
    )
    parser.add_argument(
        "--ui-host",
        default="127.0.0.1",
        help="Streamlit server host (default: 127.0.0.1)"
    )
    parser.add_argument(
        "--ui-port",
        type=int,
        default=8501,
        help="Streamlit server port (default: 8501)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode"
    )
    parser.add_argument(
        "--api-only",
        action="store_true",
        help="Start only the FastAPI backend"
    )
    parser.add_argument(
        "--ui-only",
        action="store_true",
        help="Start only the Streamlit frontend"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.debug)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    logger.info("🎯 Starting QCM Generator Pro")
    logger.info(f"Debug mode: {args.debug}")
    
    try:
        if args.api_only:
            # Start only FastAPI
            start_fastapi_server(
                host=args.api_host,
                port=args.api_port,
                debug=args.debug
            )
        elif args.ui_only:
            # Start only Streamlit
            start_streamlit_app(
                host=args.ui_host,
                port=args.ui_port,
                debug=args.debug
            )
        else:
            # Start both services
            logger.info("Starting both FastAPI backend and Streamlit frontend...")
            
            # Start FastAPI in a separate process
            api_process = multiprocessing.Process(
                target=start_fastapi_server,
                args=(args.api_host, args.api_port, args.debug)
            )
            api_process.start()
            
            logger.info(f"FastAPI started (PID: {api_process.pid})")
            
            try:
                # Start Streamlit in main process
                start_streamlit_app(
                    host=args.ui_host,
                    port=args.ui_port,
                    debug=args.debug
                )
            finally:
                # Cleanup API process
                if api_process.is_alive():
                    logger.info("Terminating FastAPI process...")
                    api_process.terminate()
                    api_process.join(timeout=5)
                    if api_process.is_alive():
                        api_process.kill()
                    
    except KeyboardInterrupt:
        logger.info("Shutdown requested by user")
    except Exception as e:
        logger.error(f"Application error: {e}")
        sys.exit(1)
    
    logger.info("QCM Generator Pro stopped")


if __name__ == "__main__":
    main()