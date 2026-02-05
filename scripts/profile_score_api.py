import argparse
import asyncio
import logging
import multiprocessing
import os
import subprocess
import sys
import time
import requests
import json
from sgl_jax.srt.utils import kill_process_tree

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def wait_for_server(base_url, timeout=600):
    """Wait for the server to be ready."""
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = requests.get(f"{base_url}/health")
            if response.status_code == 200:
                logger.info("Server is ready!")
                return True
        except requests.RequestException:
            pass
        time.sleep(5)
    logger.error("Server failed to start within timeout.")
    return False

def run_workload(base_url, model_name):
    """Send a variety of score requests."""
    logger.info("Starting workload...")
    
    url = f"{base_url}/v1/score"
    headers = {"Content-Type": "application/json"}
    
    # 1. Single Item Requests
    logger.info("Running single-item requests...")
    for i in range(10):
        data = {
            "model": model_name,
            "text": "The capital of France is",
            "score_text": [" Paris", " London"]
        }
        try:
            requests.post(url, json=data, headers=headers)
        except Exception as e:
            logger.error(f"Request failed: {e}")

    # 2. Multi-item Batch (Small)
    logger.info("Running small batch requests (4 items)...")
    for i in range(5):
        data = {
            "model": model_name,
            "text": "Which is a fruit?",
            "score_text": [" Apple", " Car", " Bus", " Dog"]
        }
        requests.post(url, json=data, headers=headers)

    # 3. Multi-item Batch (Large)
    logger.info("Running large batch requests (32 items)...")
    large_batch = [f" Item {i}" for i in range(32)]
    for i in range(3):
        data = {
            "model": model_name,
            "text": "Pick the best item.",
            "score_text": large_batch
        }
        requests.post(url, json=data, headers=headers)
        
    logger.info("Workload completed.")

def main():
    parser = argparse.ArgumentParser(description="Profile SGLang Score API")
    parser.add_argument("--model-path", type=str, required=True, help="Path to the model")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=str, default="30000")
    parser.add_argument("--output-dir", type=str, default="/outputs", help="Directory to save profile")
    parser.add_argument("--duration", type=int, default=60, help="Duration to run py-spy")
    args = parser.parse_args()

    base_url = f"http://{args.host}:{args.port}"
    
    # Launch Server
    logger.info(f"Launching server with model: {args.model_path}")
    server_cmd = [
        "python", "-m", "sgl_jax.launch_server",
        "--model-path", args.model_path,
        "--host", args.host,
        "--port", args.port,
        "--trust-remote-code",
        "--mem-fraction-static", "0.8",
        "--dtype", "bfloat16"
    ]
    
    # Check for TPU
    if os.getenv("TPU_NAME"):
        server_cmd.extend(["--device", "tpu"])
    
    server_process = subprocess.Popen(server_cmd, stdout=sys.stdout, stderr=sys.stderr)
    
    try:
        if not wait_for_server(base_url):
            raise RuntimeError("Server start failed")

        # Start Profiler (py-spy)
        # We attach to the main server process. Note: sglang-jax uses multiprocessing.
        # py-spy's --subprocess flag is crucial here to capture child processes.
        profile_file = os.path.join(args.output_dir, "score_api_profile.svg")
        logger.info(f"Starting py-spy on PID {server_process.pid}...")
        
        profiler_cmd = [
            "py-spy", "record",
            "--pid", str(server_process.pid),
            "--output", profile_file,
            "--subprocesses",  # Profile child processes too
            "--rate", "100",   # Sampling rate
            "--format", "speedscope" # compatible format
        ]
        
        # We run py-spy as a background process so we can run workload simultaneously
        profiler_process = subprocess.Popen(profiler_cmd)
        
        # Give profiler a moment to attach
        time.sleep(2)
        
        # Run Workload
        run_workload(base_url, args.model_path)
        
        # Let it run for a bit more if needed or until profiler is done
        # Usually py-spy runs until we kill it (Ctrl-C) or for a set duration if --duration is used (not used here)
        # Here we let it run for the duration of the workload + buffer, then terminate.
        
        time.sleep(5) 
        logger.info("Stopping profiler...")
        profiler_process.terminate()
        try:
            profiler_process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            profiler_process.kill()
            
        logger.info(f"Profile saved to {profile_file}")

    finally:
        logger.info("Stopping server...")
        kill_process_tree(server_process.pid)

if __name__ == "__main__":
    main()
