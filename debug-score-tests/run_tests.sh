#!/bin/bash
set -e

echo "Starting Score API Core Tests..."
cd /app/sglang-jax
export PYTHONPATH=$PYTHONPATH:$(pwd)/python

# Run the core tests
python3 test/srt/test_score_api_core.py

echo "Tests completed successfully."
