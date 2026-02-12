#!/bin/bash
# Usage: ./run_gpu_tests.sh <pod_name> <tp_size>

POD_NAME=$1
TP_SIZE=$2
NAMESPACE="eval-serving"

if [ -z "$POD_NAME" ] || [ -z "$TP_SIZE" ]; then
    echo "Usage: $0 <pod_name> <tp_size>"
    exit 1
fi

echo "Running GPU benchmarks on $POD_NAME (TP=$TP_SIZE)..."

# Copy the benchmark script
kubectl cp test_bench_multi_item_score_gpu.py $NAMESPACE/$POD_NAME:/tmp/test_bench.py

# Run the benchmark
kubectl exec -n $NAMESPACE $POD_NAME -- /bin/bash -c "
    export TP_SIZE=$TP_SIZE && 
    export HF_HUB_OFFLINE=1 && 
    export HF_HOME=/models && 
    python3 -m pytest /tmp/test_bench.py -s
"
