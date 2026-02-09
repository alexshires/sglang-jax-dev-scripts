#!/bin/bash
set -e

# Default values
ACCELERATOR="tpu-v6e-slice"
TOPOLOGY="1x1"
ACCELERATOR_COUNT="1" # Only for GPU
IMAGE="europe-docker.pkg.dev/ashires-e7aaot/container/vllm-tpu:pr"
JOB_NAME_PREFIX="sglang-benchmark-"

usage() {
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  --accelerator TYPE   Accelerator type (default: tpu-v6e-slice)"
    echo "  --topology TOPOLOGY  TPU topology (default: 1x1)"
    echo "  --count COUNT        Accelerator count (for GPU) (default: 1)"
    echo "  --image IMAGE        Docker image (default: vllm-tpu:pr)"
    echo "  --help               Show this help message"
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --accelerator)
            ACCELERATOR="$2"
            shift 2
            ;;
        --topology)
            TOPOLOGY="$2"
            shift 2
            ;;
        --count)
            ACCELERATOR_COUNT="$2"
            shift 2
            ;;
        --image)
            IMAGE="$2"
            shift 2
            ;;
        --help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# Render the job YAML
# We use sed to replace placeholders. Ideally we should use envsubst but we need to match specific keys.
# Actually, the template currently has hardcoded values. Let's make sure the template uses env vars or we replace them.

TEMPLATE_FILE="sglang-scripts/benchmark/benchmark-job.yaml"
OUTPUT_FILE="sglang-scripts/benchmark/benchmark-job-rendered.yaml"

echo "Rendering job for accelerator: $ACCELERATOR..."

# Simple replacement for now. 
# WARNING: This assumes specific formatting in the YAML.
cp $TEMPLATE_FILE $OUTPUT_FILE

# accelerators
# Detect if TPU or GPU based on string
if [[ "$ACCELERATOR" == *"tpu"* ]]; then
    # TPU
    sed -i "s|cloud.google.com/gke-tpu-accelerator: .*|cloud.google.com/gke-tpu-accelerator: \"$ACCELERATOR\"|g" $OUTPUT_FILE
    sed -i "s|cloud.google.com/gke-tpu-topology: .*|cloud.google.com/gke-tpu-topology: \"$TOPOLOGY\"|g" $OUTPUT_FILE
    # Update resource limits for TPU
    # We assume 1 TPU slice implies google.com/tpu: 1 in K8s (which is typical for GKE TPU)
    sed -i "s|google.com/tpu: .*|google.com/tpu: \"1\"|g" $OUTPUT_FILE
    # Remove nvidia.com/gpu line if it exists (not in current template but good for future)
    # The current template has hardcoded google.com/tpu. 
    # If we want to support GPU, we need to swap the resource lines completely.
else
    # GPU
    # Replace TPU specific lines with GPU specific lines
    sed -i "s|cloud.google.com/gke-tpu-accelerator: .*|cloud.google.com/gke-accelerator: \"$ACCELERATOR\"|g" $OUTPUT_FILE
    sed -i "/cloud.google.com\/gke-tpu-topology:/d" $OUTPUT_FILE
    
    # Replace resources
    # This is tricky with sed. Ideally use yq or similar.
    # Hack: Replace google.com/tpu with nvidia.com/gpu
    sed -i "s|google.com/tpu: .*|nvidia.com/gpu: \"$ACCELERATOR_COUNT\"|g" $OUTPUT_FILE
fi

# Image
sed -i "s|image: .*|image: $IMAGE|g" $OUTPUT_FILE

echo "Job rendered to $OUTPUT_FILE"
echo "Submitting job..."

kubectl create -f $OUTPUT_FILE

echo "Job submitted. Monitor with:"
echo "kubectl get jobs -n eval-serving"
