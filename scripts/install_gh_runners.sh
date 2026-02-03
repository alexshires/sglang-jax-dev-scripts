#!/bin/bash
set -e

# Ensure common binary paths are included
export PATH=$PATH:/usr/local/bin

# Configuration
PROJECT_ID="ashires-e7aaot"
CLUSTER_NAME="ray-tpu-test-cluster"
REGION="europe-west4"
GITHUB_CONFIG_URL="${1:-}" # Pass URL as first arg or set env var

# Load GitHub Token from file or argument
GITHUB_TOKEN_FILE="$HOME/sglang_gh_pat.txt"
if [[ -f "$GITHUB_TOKEN_FILE" ]]; then
  GITHUB_TOKEN=$(cat "$GITHUB_TOKEN_FILE")
fi
GITHUB_TOKEN="${2:-$GITHUB_TOKEN}" # Argument takes precedence if provided

if [[ -z "$GITHUB_CONFIG_URL" || -z "$GITHUB_TOKEN" ]]; then
  echo "Usage: $0 <GITHUB_CONFIG_URL> [GITHUB_TOKEN]"
  echo "Example: $0 https://github.com/my-org"
  echo "Note: GITHUB_TOKEN is read from $GITHUB_TOKEN_FILE if not provided."
  exit 1
fi

echo "Connecting to GKE cluster..."
gcloud container clusters get-credentials "$CLUSTER_NAME" --region "$REGION" --project "$PROJECT_ID"

echo "Creating namespaces..."
kubectl create namespace arc-systems --dry-run=client -o yaml | kubectl apply -f -
kubectl create namespace arc-runners --dry-run=client -o yaml | kubectl apply -f -

echo "Creating GitHub Token Secret..."
kubectl create secret generic gh-token-secret \
  --namespace=arc-runners \
  --from-literal=github_token="$GITHUB_TOKEN" \
  --dry-run=client -o yaml | kubectl apply -f -

echo "Installing ARC Controller..."
helm upgrade --install arc-controller oci://ghcr.io/actions/actions-runner-controller-charts/gha-runner-scale-set-controller \
  --namespace arc-systems \
  --version 0.9.3

echo "Installing GPU Runner Scale Set..."
helm upgrade --install arc-runner-gpu oci://ghcr.io/actions/actions-runner-controller-charts/gha-runner-scale-set \
  --namespace arc-runners \
  --version 0.9.3 \
  --set githubConfigUrl="$GITHUB_CONFIG_URL" \
  --set githubConfigSecret="gh-token-secret" \
  -f sglang-scripts/k8s/gpu-runner-values.yaml

echo "Installing CPU Runner Scale Set (arc-runner-cpu)..."
helm upgrade --install arc-runner-cpu oci://ghcr.io/actions/actions-runner-controller-charts/gha-runner-scale-set \
  --namespace arc-runners \
  --version 0.9.3 \
  --set githubConfigUrl="$GITHUB_CONFIG_URL" \
  --set githubConfigSecret="gh-token-secret" \
  -f sglang-scripts/k8s/cpu-runner-values.yaml

echo "Installing TPU Runner Scale Set (arc-runner-v6e-1)..."
helm upgrade --install arc-runner-tpu oci://ghcr.io/actions/actions-runner-controller-charts/gha-runner-scale-set \
  --namespace arc-runners \
  --version 0.9.3 \
  --set githubConfigUrl="$GITHUB_CONFIG_URL" \
  --set githubConfigSecret="gh-token-secret" \
  -f sglang-scripts/k8s/tpu-v6e-1-runner-values.yaml

echo "Installing TPU Runner 2 (Standard) Scale Set (arc-runner-v6e-1-standard)..."
helm upgrade --install arc-runner-tpu-2 oci://ghcr.io/actions/actions-runner-controller-charts/gha-runner-scale-set \
  --namespace arc-runners \
  --version 0.9.3 \
  --set githubConfigUrl="$GITHUB_CONFIG_URL" \
  --set githubConfigSecret="gh-token-secret" \
  -f sglang-scripts/k8s/tpu-v6e-1-standard-runner-values.yaml

echo "Installing TPU Runner v6e-4 Scale Set (arc-runner-v6e-4)..."
helm upgrade --install arc-runner-tpu-v6e-4 oci://ghcr.io/actions/actions-runner-controller-charts/gha-runner-scale-set \
  --namespace arc-runners \
  --version 0.9.3 \
  --set githubConfigUrl="$GITHUB_CONFIG_URL" \
  --set githubConfigSecret="gh-token-secret" \
  -f sglang-scripts/k8s/tpu-v6e-4-runner-values.yaml

echo "Installing TPU Runner v5e Scale Set (arc-runner-v5e-4)..."
helm upgrade --install arc-runner-tpu-v5e oci://ghcr.io/actions/actions-runner-controller-charts/gha-runner-scale-set \
  --namespace arc-runners \
  --version 0.9.3 \
  --set githubConfigUrl="$GITHUB_CONFIG_URL" \
  --set githubConfigSecret="gh-token-secret" \
  -f sglang-scripts/k8s/tpu-v5e-4-runner-values.yaml

echo "Done! Runners deployed."
