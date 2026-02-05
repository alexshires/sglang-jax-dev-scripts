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

def run_workload(base_url, model_name, duration=60):
    """Send a variety of score requests for a fixed duration."""
    logger.info(f"Starting workload for {duration} seconds...")
    
    url = f"{base_url}/v1/score"
    headers = {"Content-Type": "application/json"}
    
    # We need some valid label_token_ids for the Qwen tokenizer.
    label_token_ids = [1284, 1342] # " Yes", " No" approx
    
    start_time = time.time()
    count = 0
    
    while time.time() - start_time < duration:
        count += 1
        
        # 1. Single Item Request
        data = {
            "model": model_name,
            "query": "The capital of France is",
            "items": [" Paris", " London"],
            "label_token_ids": label_token_ids
        }
        try:
            requests.post(url, json=data, headers=headers)
        except Exception:
            pass

        # 2. Multi-item Batch (Small)
        data = {
            "model": model_name,
            "query": "Which is a fruit?",
            "items": [" Apple", " Car", " Bus", " Dog"],
            "label_token_ids": label_token_ids
        }
        try:
            requests.post(url, json=data, headers=headers)
        except Exception:
            pass

        # 3. Multi-item Batch (Large)
        large_batch = [f" Item {i}" for i in range(32)]
        data = {
            "model": model_name,
            "query": "Pick the best item.",
            "items": large_batch,
            "label_token_ids": label_token_ids
        }
        try:
            requests.post(url, json=data, headers=headers)
        except Exception:
            pass
            
        if count % 10 == 0:
            logger.info(f"Completed {count} workload iterations...")
            
    logger.info(f"Workload completed. Total iterations: {count}")

def main():
    parser = argparse.ArgumentParser(description="Profile SGLang Score API")
    parser.add_argument("--model-path", type=str, required=True, help="Path to the model")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=str, default="30000")
    parser.add_argument("--output-dir", type=str, default="/outputs", help="Directory to save profile")
    parser.add_argument("--duration", type=int, default=60, help="Duration to run py-spy")
    parser.add_argument("--profiler", type=str, default="pyspy", choices=["pyspy", "cprofile"], help="Profiler to use")
    args = parser.parse_args()

    base_url = f"http://{args.host}:{args.port}"
    
    # Launch Server
    logger.info(f"Launching server with model: {args.model_path}")
    
    server_cmd = ["python"]
    if args.profiler == "cprofile":
        profile_file = os.path.join(args.output_dir, "server.prof")
        server_cmd.extend(["-m", "cProfile", "-o", profile_file])
    
    server_cmd.extend([
        "-m", "sgl_jax.launch_server",
        "--model-path", args.model_path,
        "--host", args.host,
        "--port", args.port,
        "--trust-remote-code",
        "--mem-fraction-static", "0.8",
        "--dtype", "bfloat16"
    ])
    
    # Check for TPU
    if os.getenv("TPU_NAME") or "tpu" in os.getenv("JAX_PLATFORMS", "").lower():
        server_cmd.extend(["--device", "tpu"])
    
    server_process = subprocess.Popen(server_cmd, stdout=sys.stdout, stderr=sys.stderr)
    
    profiler_process = None
    
    try:
        if not wait_for_server(base_url):
            raise RuntimeError("Server start failed")

        if args.profiler == "pyspy":
            # Start Profiler (py-spy)
            profile_file = os.path.join(args.output_dir, "score_api_profile.svg")
            logger.info(f"Starting py-spy on PID {server_process.pid}...")
            
            # Note: speedscope format is good for multi-process
            profiler_cmd = [
                "py-spy", "record",
                "--pid", str(server_process.pid),
                "--output", profile_file,
                "--subprocesses",
                "--rate", "10",
                "--format", "speedscope"
            ]
            
            profiler_process = subprocess.Popen(profiler_cmd)
            # Give profiler a moment to attach
            time.sleep(5)
        
        # Run Workload
        run_workload(base_url, args.model_path, duration=args.duration)
        
        time.sleep(5) 
        
        if profiler_process:
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
        # For cProfile, normal termination should save the file. 
        # kill_process_tree sends SIGTERM which Python handles gracefully usually.

if __name__ == "__main__":
    main()
