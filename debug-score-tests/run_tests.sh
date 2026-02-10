#!/bin/bash
set -e

echo "Starting Score API Multi-Item Scoring Tests..."
cd /app/sglang-jax
export PYTHONPATH=$PYTHONPATH:$(pwd)/python

# Run the multi-item scoring suite
python3 test/srt/run_suite.py --suite multi-item-scoring-test-tpu-v6e-1 --auto-partition-id 0 --auto-partition-size 1

echo "Tests completed successfully."
