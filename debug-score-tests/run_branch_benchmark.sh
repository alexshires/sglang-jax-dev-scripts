#!/bin/bash
# Usage: ./run_branch_benchmark.sh <pod_name>

POD_NAME=$1
NAMESPACE="eval-serving"
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../" && pwd)"

if [ -z "$POD_NAME" ]; then
    echo "Usage: $0 <pod_name>"
    exit 1
fi

# Determine remote directory (image layout varies)
REMOTE_WORK_DIR="/app"
REMOTE_TEST_DIR="$REMOTE_WORK_DIR/test/srt"

echo "Ensuring remote test directory exists: $REMOTE_TEST_DIR"
kubectl exec -n $NAMESPACE $POD_NAME -c inference-server -- mkdir -p $REMOTE_TEST_DIR

echo "Syncing canonical TPU benchmark script to $POD_NAME..."
kubectl cp "$SCRIPT_DIR/test_bench_multi_item_score_tpu.py" $NAMESPACE/$POD_NAME:$REMOTE_TEST_DIR/test_bench_multi_item_score_tpu.py -c inference-server

echo "Running multi-item scoring benchmarks on $POD_NAME..."
kubectl exec -n $NAMESPACE $POD_NAME -c inference-server -- /bin/bash -c "
    cd $REMOTE_WORK_DIR && \
    export PYTHONPATH=\$PYTHONPATH:\$(pwd)/python && \
    python3 -m unittest test/srt/test_bench_multi_item_score_tpu.py
"
