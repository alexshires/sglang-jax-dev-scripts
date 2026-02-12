#!/bin/bash
# Usage: ./restart_pod.sh <pod_name> <manifest_file>

POD_NAME=$1
MANIFEST=$2
NAMESPACE="eval-serving"

if [ -z "$POD_NAME" ] || [ -z "$MANIFEST" ]; then
    echo "Usage: $0 <pod_name> <manifest_file>"
    exit 1
fi

echo "Deleting existing pod: $POD_NAME..."
kubectl delete pod -n $NAMESPACE $POD_NAME --ignore-not-found

echo "Applying manifest: $MANIFEST..."
kubectl apply -f $MANIFEST

echo "Waiting for pod $POD_NAME to be ready..."
kubectl wait --for=condition=Ready pod/$POD_NAME -n $NAMESPACE --timeout=900s

echo "Pod $POD_NAME is ready."
