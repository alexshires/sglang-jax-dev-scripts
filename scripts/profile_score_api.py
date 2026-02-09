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
import signal
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
    
    label_token_ids = [1284, 1342] # " Yes", " No" approx
    
    start_time = time.time()
    count = 0
    
    while time.time() - start_time < duration:
        count += 1
        
        # Mixed requests
        requests_to_run = [
            # 1. Single Item
            {
                "model": model_name,
                "query": "The capital of France is",
                "items": [" Paris", " London"],
                "label_token_ids": label_token_ids
            },
            # 2. Small Batch
            {
                "model": model_name,
                "query": "Which is a fruit?",
                "items": [" Apple", " Car", " Bus", " Dog"],
                "label_token_ids": label_token_ids
            },
            # 3. Large Batch
            {
                "model": model_name,
                "query": "Pick the best item.",
                "items": [f" Item {i}" for i in range(32)],
                "label_token_ids": label_token_ids
            }
        ]
        
        for data in requests_to_run:
            try:
                requests.post(url, json=data, headers=headers, timeout=30)
            except Exception:
                pass

        if count % 5 == 0:
            logger.info(f"Completed {count} workload iterations...")
            
    logger.info(f"Workload completed. Total iterations: {count}")

def main():
    parser = argparse.ArgumentParser(description="Profile SGLang Score API")
    parser.add_argument("--model-path", type=str, required=True, help="Path to the model")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=str, default="30000")
    parser.add_argument("--output-dir", type=str, default="/outputs", help="Directory to save profile")
    parser.add_argument("--duration", type=int, default=60, help="Duration to run workload")
    parser.add_argument("--profiler", type=str, default="pyspy", choices=["pyspy", "cprofile", "jax"], help="Profiler to use")
    args = parser.parse_args()

    base_url = f"http://{args.host}:{args.port}"
    
    # Verify Write Permission
    heartbeat_file = os.path.join(args.output_dir, "heartbeat.txt")
    try:
        os.makedirs(args.output_dir, exist_ok=True)
        with open(heartbeat_file, "w") as f:
            f.write(f"Profiler started at {time.ctime()}\n")
        logger.info(f"Verified write access to {args.output_dir}")
    except Exception as e:
        logger.error(f"Failed write access to {args.output_dir}: {e}")

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
    
    if os.getenv("TPU_NAME") or "tpu" in os.getenv("JAX_PLATFORMS", "").lower():
        server_cmd.extend(["--device", "tpu"])
    
    server_process = subprocess.Popen(server_cmd, stdout=sys.stdout, stderr=sys.stderr)
    
    profiler_process = None
    
    # Signal handling for clean exit
    def signal_handler(sig, frame):
        logger.info("Caught signal, shutting down...")
        if server_process:
            kill_process_tree(server_process.pid)
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        if not wait_for_server(base_url):
            raise RuntimeError("Server start failed")

        if args.profiler == "pyspy":
            profile_file = os.path.join(args.output_dir, "score_api_profile.svg")
            logger.info(f"Starting py-spy on PID {server_process.pid}...")
            profiler_cmd = [
                "py-spy", "record",
                "--pid", str(server_process.pid),
                "--output", profile_file,
                "--subprocesses",
                "--rate", "10",
                "--format", "speedscope"
            ]
            profiler_process = subprocess.Popen(profiler_cmd)
            time.sleep(5)
        
        elif args.profiler == "jax":
            logger.info("Enabling JAX profiling via /start_profile...")
            trace_dir = os.path.join(args.output_dir, "jax_trace")
            os.makedirs(trace_dir, exist_ok=True)
            try:
                resp = requests.post(f"{base_url}/start_profile", json={"output_dir": trace_dir})
                logger.info(f"JAX Start Profile response: {resp.status_code} {resp.text}")
            except Exception as e:
                logger.error(f"Failed to start JAX profile: {e}")

        # Run Workload
        run_workload(base_url, args.model_path, duration=args.duration)
        
        # Stop Profiler
        if args.profiler == "pyspy" and profiler_process:
            logger.info("Stopping py-spy...")
            profiler_process.terminate()
            try:
                profiler_process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                profiler_process.kill()
            logger.info(f"Profile saved to {profile_file}")
            
        elif args.profiler == "jax":
            logger.info("Stopping JAX profiling via /stop_profile...")
            try:
                resp = requests.post(f"{base_url}/stop_profile")
                logger.info(f"JAX Stop Profile response: {resp.status_code} {resp.text}")
                logger.info("Waiting for JAX to flush traces (this can take 2-3 minutes)...")
                time.sleep(120)
            except Exception as e:
                logger.error(f"Failed to stop JAX profile: {e}")

    finally:
        logger.info("Stopping server...")
        kill_process_tree(server_process.pid)
        time.sleep(10)
        logger.info("Exit.")

if __name__ == "__main__":
    main()
