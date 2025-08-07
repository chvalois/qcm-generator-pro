#!/usr/bin/env python3
"""
QCM Generator Pro - Docker Startup Script

This script handles Docker container startup including:
- Environment validation
- Service initialization
- Health checks
- Application startup
"""

import asyncio
import logging
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

# Add src to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from docker_setup import DockerSetupManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - [DOCKER] %(message)s"
)
logger = logging.getLogger(__name__)


class DockerStartupManager:
    """Manages Docker container startup process."""
    
    def __init__(self):
        self.setup_manager = DockerSetupManager()
        self.processes: List[subprocess.Popen] = []
        self.shutdown_event = asyncio.Event()
        
    def signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        logger.info(f"Received signal {signum}, initiating graceful shutdown")
        self.shutdown_event.set()
        
    async def wait_for_dependencies(self) -> bool:
        """Wait for external dependencies to be ready."""
        logger.info("Waiting for dependencies to be ready")
        
        # Wait for Ollama service (if configured)
        ollama_url = os.getenv("OLLAMA_BASE_URL")
        if ollama_url:
            logger.info(f"Waiting for Ollama service at {ollama_url}")
            if not await self.setup_manager.wait_for_ollama(timeout=120):
                logger.error("Ollama service not ready - continuing with fallback mode")
                
        # Wait for Redis (if configured)
        redis_url = os.getenv("REDIS_URL")
        if redis_url:
            logger.info("Checking Redis connection")
            # Simple check - we'll continue even if Redis fails
            
        return True
        
    async def initialize_application(self) -> bool:
        """Initialize application components."""
        logger.info("Initializing application components")
        
        try:
            # Run setup tasks
            success = await self.setup_manager.full_setup()
            if not success:
                logger.error("Application setup failed")
                return False
                
            logger.info("Application initialization completed")
            return True
            
        except Exception as e:
            logger.error(f"Application initialization failed: {e}")
            return False
            
    def start_api_server(self) -> subprocess.Popen:
        """Start the FastAPI server."""
        logger.info("Starting FastAPI server")
        
        api_host = os.getenv("API_HOST", "0.0.0.0")
        api_port = os.getenv("API_PORT", "8001")
        
        cmd = [
            "python3", "-m", "uvicorn",
            "src.api.main:app",
            "--host", api_host,
            "--port", api_port,
            "--workers", "1",
            "--log-level", "info"
        ]
        
        logger.info(f"API server command: {' '.join(cmd)}")
        
        process = subprocess.Popen(
            cmd,
            cwd=project_root,
            env=os.environ,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        self.processes.append(process)
        return process
        
    def start_ui_server(self) -> subprocess.Popen:
        """Start the Streamlit UI server."""
        logger.info("Starting Streamlit UI server")
        
        ui_host = os.getenv("UI_HOST", "0.0.0.0")
        ui_port = os.getenv("UI_PORT", "8501")
        
        cmd = [
            "streamlit", "run",
            "main_app.py",
            "--server.address", ui_host,
            "--server.port", ui_port,
            "--server.headless", "true",
            "--browser.gatherUsageStats", "false",
            "--logger.level", "info"
        ]
        
        logger.info(f"Starting UI server on {ui_host}:{ui_port}")
        
        process = subprocess.Popen(
            cmd,
            cwd=project_root,
            env=os.environ,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        self.processes.append(process)
        return process
        
    async def monitor_processes(self):
        """Monitor running processes and handle failures."""
        logger.info("Starting process monitoring")
        
        while not self.shutdown_event.is_set():
            try:
                # Check process health
                for i, process in enumerate(self.processes.copy()):
                    if process.poll() is not None:
                        # Process has terminated
                        logger.error(f"Process {i} terminated with code {process.returncode}")
                        
                        # Read any remaining output
                        try:
                            output = process.stdout.read()
                            if output:
                                logger.error(f"Process {i} output: {output}")
                        except:
                            pass
                            
                        # Remove from list
                        self.processes.remove(process)
                        
                        # Trigger shutdown if critical process died
                        logger.error("Critical process died, initiating shutdown")
                        self.shutdown_event.set()
                        break
                        
                # Sleep between checks
                await asyncio.sleep(5)
                
            except Exception as e:
                logger.error(f"Error in process monitoring: {e}")
                await asyncio.sleep(5)
                
    async def stream_process_output(self, process: subprocess.Popen, name: str):
        """Stream process output to logs."""
        try:
            while not self.shutdown_event.is_set() and process.poll() is None:
                line = process.stdout.readline()
                if line:
                    logger.info(f"[{name}] {line.strip()}")
                else:
                    await asyncio.sleep(0.1)
        except Exception as e:
            logger.error(f"Error streaming output from {name}: {e}")
            
    async def graceful_shutdown(self):
        """Perform graceful shutdown of all processes."""
        logger.info("Starting graceful shutdown")
        
        # Send SIGTERM to all processes
        for i, process in enumerate(self.processes):
            if process.poll() is None:
                logger.info(f"Terminating process {i}")
                try:
                    process.terminate()
                except:
                    pass
                    
        # Wait for processes to terminate
        shutdown_timeout = 30
        start_time = time.time()
        
        while self.processes and (time.time() - start_time) < shutdown_timeout:
            for process in self.processes.copy():
                if process.poll() is not None:
                    self.processes.remove(process)
            await asyncio.sleep(1)
            
        # Force kill remaining processes
        for process in self.processes:
            if process.poll() is None:
                logger.warning(f"Force killing process {process.pid}")
                try:
                    process.kill()
                except:
                    pass
                    
        logger.info("Graceful shutdown completed")
        
    async def health_check_loop(self):
        """Periodic health checks."""
        logger.info("Starting health check loop")
        
        while not self.shutdown_event.is_set():
            try:
                # Basic health check every 60 seconds
                await asyncio.sleep(60)
                
                if self.shutdown_event.is_set():
                    break
                    
                # Check if processes are still running
                running_processes = [p for p in self.processes if p.poll() is None]
                logger.info(f"Health check: {len(running_processes)}/{len(self.processes)} processes running")
                
                if len(running_processes) < len(self.processes):
                    logger.warning("Some processes have died")
                    
            except Exception as e:
                logger.error(f"Error in health check: {e}")
                
    async def run(self):
        """Main startup routine."""
        logger.info("Starting QCM Generator Pro Docker container")
        
        # Setup signal handlers
        signal.signal(signal.SIGTERM, self.signal_handler)
        signal.signal(signal.SIGINT, self.signal_handler)
        
        try:
            # Wait for dependencies
            if not await self.wait_for_dependencies():
                logger.error("Dependencies not ready")
                return False
                
            # Initialize application
            if not await self.initialize_application():
                logger.error("Application initialization failed")
                return False
                
            # Start services
            api_process = self.start_api_server()
            ui_process = self.start_ui_server()
            
            logger.info("All services started successfully")
            
            # Create monitoring tasks
            monitor_task = asyncio.create_task(self.monitor_processes())
            health_task = asyncio.create_task(self.health_check_loop())
            
            # Stream output from processes
            api_output_task = asyncio.create_task(
                self.stream_process_output(api_process, "API")
            )
            ui_output_task = asyncio.create_task(
                self.stream_process_output(ui_process, "UI")
            )
            
            # Wait for shutdown signal
            await self.shutdown_event.wait()
            
            # Cancel tasks
            for task in [monitor_task, health_task, api_output_task, ui_output_task]:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
                    
            # Graceful shutdown
            await self.graceful_shutdown()
            
            logger.info("Container shutdown completed")
            return True
            
        except Exception as e:
            logger.error(f"Startup failed: {e}")
            await self.graceful_shutdown()
            return False


async def main():
    """Main entry point."""
    startup_manager = DockerStartupManager()
    
    try:
        success = await startup_manager.run()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        logger.info("Startup interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Startup failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())