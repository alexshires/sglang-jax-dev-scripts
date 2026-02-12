#!/bin/bash
# Usage: ./run_remote_tests.sh <pod_name>

POD_NAME=$1
NAMESPACE="eval-serving"
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../" && pwd)"

if [ -z "$POD_NAME" ]; then
    echo "Usage: $0 <pod_name>"
    exit 1
fi

echo "Syncing latest benchmark script to $POD_NAME..."
kubectl cp "$PROJECT_ROOT/sglang-jax/test/srt/test_bench_multi_item_score.py" $NAMESPACE/$POD_NAME:/app/sglang-jax/test/srt/test_bench_multi_item_score.py -c inference-server

echo "Running multi-item scoring benchmarks on $POD_NAME..."
kubectl exec -n $NAMESPACE $POD_NAME -c inference-server -- /bin/bash -c "
    cd /app/sglang-jax && \
    python3 -m unittest test/srt/test_bench_multi_item_score.py
"
