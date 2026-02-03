# RFC-009: Self-Hosted ARC Runners with TPU for Fork CI

| | |
|------------|------|
| **Status** | Draft |
| **Author** | Engineering Team |
| **Created** | 2026-02-03 |
| **Updated** | 2026-02-03 |
| **Related** | [RFC-002](002-cicd-tpu-testing.md) |

## Summary

Set up self-hosted GitHub Actions runners using Actions Runner Controller (ARC) on GKE with TPU node pools, matching the upstream sgl-project/sglang-jax infrastructure. This enables full CI/CD parity with upstream for fork development.

## Motivation

The fork (`alexshires/sglang-jax`) uses the same workflow as upstream but lacks the self-hosted ARC runners with TPU access. This causes:

1. **CI failures** - Tests reference `arc-runner-v6e-1` which doesn't exist in the fork
2. **Model access failures** - Tests expect models at `/models/` which isn't available
3. **No TPU testing** - Can't validate TPU-specific code paths in CI

### Current State

| Aspect | Upstream (`sgl-project`) | Fork (`alexshires`) |
|--------|--------------------------|---------------------|
| Runners | Self-hosted ARC on GKE | None (workflow fails) |
| TPU access | TPU v6e node pools | None |
| Model storage | `/models/` via GCS FUSE | None |
| CI status | Passing | Failing |

## Goals

1. **Full CI parity** with upstream sgl-project/sglang-jax
2. **TPU v6e runners** with labels `arc-runner-v6e-1` and `arc-runner-v6e-4`
3. **Persistent model storage** at `/models/` path
4. **Cost-effective** infrastructure with autoscaling
5. **Reproducible setup** with Infrastructure as Code

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        GitHub Actions                            │
│                                                                  │
│  Workflow: runs-on: arc-runner-v6e-1                            │
│                        │                                         │
└────────────────────────┼─────────────────────────────────────────┘
                         │ webhook
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                    GKE Cluster                                   │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │              Actions Runner Controller (ARC)                 ││
│  │                                                              ││
│  │  ┌──────────────────┐    ┌──────────────────┐              ││
│  │  │ Runner Scale Set │    │ Runner Scale Set │              ││
│  │  │ arc-runner-v6e-1 │    │ arc-runner-v6e-4 │              ││
│  │  │   (min: 0)       │    │   (min: 0)       │              ││
│  │  └────────┬─────────┘    └────────┬─────────┘              ││
│  │           │                       │                         ││
│  └───────────┼───────────────────────┼─────────────────────────┘│
│              │                       │                          │
│  ┌───────────▼───────────┐  ┌───────▼───────────┐             │
│  │   TPU v6e-1 Node Pool │  │ TPU v6e-4 Node Pool│             │
│  │   (autoscaling 0-4)   │  │ (autoscaling 0-2)  │             │
│  │                       │  │                    │             │
│  │  ┌─────────────────┐  │  │ ┌────────────────┐ │             │
│  │  │  Runner Pod     │  │  │ │  Runner Pod    │ │             │
│  │  │  + TPU v6e-1    │  │  │ │  + TPU v6e-4   │ │             │
│  │  │  + /models/     │  │  │ │  + /models/    │ │             │
│  │  └─────────────────┘  │  │ └────────────────┘ │             │
│  └───────────────────────┘  └────────────────────┘             │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                    GCS FUSE / Filestore                      ││
│  │                    /models/ mount                            ││
│  │  - Wan-AI/Wan2.1-T2V-1.3B-Diffusers                        ││
│  │  - meta-llama/Llama-3.2-1B-Instruct                        ││
│  │  - ...                                                       ││
│  └─────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
```

## Prerequisites

### 1. GCP Project Setup

```bash
# Set project
export PROJECT_ID="your-gcp-project-id"
export REGION="us-central1"
export ZONE="us-central1-b"

gcloud config set project $PROJECT_ID

# Enable required APIs
gcloud services enable \
  container.googleapis.com \
  compute.googleapis.com \
  tpu.googleapis.com \
  file.googleapis.com \
  secretmanager.googleapis.com \
  iam.googleapis.com
```

### 2. TPU Quota

Request quota for TPU v6e in your region:

| Quota | Minimum Required | Recommended |
|-------|------------------|-------------|
| `TPU_V6E_LITE_PODSLICE_CHIPS` | 5 | 10 |
| `PREEMPTIBLE_TPU_V6E_LITE_PODSLICE_CHIPS` | 5 | 10 |

```bash
# Check current quota
gcloud compute regions describe $REGION \
  --format="table(quotas.metric,quotas.limit,quotas.usage)" \
  | grep -i tpu
```

### 3. GitHub App (Recommended) or PAT

**Option A: GitHub App (Recommended)**

1. Create GitHub App at `https://github.com/settings/apps/new`
2. Permissions needed:
   - Repository: Actions (read), Administration (read/write), Metadata (read)
3. Install app on your repository
4. Generate private key and note App ID + Installation ID

**Option B: Personal Access Token**

```bash
# Create PAT with scopes: repo, workflow
# Store in Secret Manager
echo -n "ghp_your_token_here" | \
  gcloud secrets create github-arc-token \
  --data-file=-
```

## Implementation

### Phase 1: GKE Cluster with TPU Node Pools

#### 1.1 Create GKE Cluster

```bash
# Create cluster with workload identity
gcloud container clusters create sglang-arc-runners \
  --zone=$ZONE \
  --machine-type=e2-standard-4 \
  --num-nodes=1 \
  --enable-autoscaling \
  --min-nodes=1 \
  --max-nodes=3 \
  --workload-pool=$PROJECT_ID.svc.id.goog \
  --enable-ip-alias \
  --release-channel=regular
```

#### 1.2 Create TPU v6e-1 Node Pool

```bash
# TPU v6e-1 node pool (1 chip)
gcloud container node-pools create tpu-v6e-1-pool \
  --cluster=sglang-arc-runners \
  --zone=$ZONE \
  --machine-type=ct6e-standard-1t \
  --tpu-topology=1x1 \
  --num-nodes=0 \
  --enable-autoscaling \
  --min-nodes=0 \
  --max-nodes=4 \
  --spot \
  --node-taints="google.com/tpu=present:NoSchedule"
```

#### 1.3 Create TPU v6e-4 Node Pool

```bash
# TPU v6e-4 node pool (4 chips)
gcloud container node-pools create tpu-v6e-4-pool \
  --cluster=sglang-arc-runners \
  --zone=$ZONE \
  --machine-type=ct6e-standard-4t \
  --tpu-topology=2x2 \
  --num-nodes=0 \
  --enable-autoscaling \
  --min-nodes=0 \
  --max-nodes=2 \
  --spot \
  --node-taints="google.com/tpu=present:NoSchedule"
```

### Phase 2: Model Storage Setup

#### Option A: GCS FUSE (Recommended for Large Models)

```bash
# Create GCS bucket for models
gsutil mb -l $REGION gs://$PROJECT_ID-sglang-models

# Download and upload models
# (Run on a VM with enough disk space)
huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B-Diffusers \
  --local-dir /tmp/Wan2.1-T2V-1.3B-Diffusers

gsutil -m cp -r /tmp/Wan2.1-T2V-1.3B-Diffusers \
  gs://$PROJECT_ID-sglang-models/Wan-AI/

# Repeat for other models...
```

Create GCS FUSE DaemonSet:

```yaml
# gcs-fuse-daemonset.yaml
apiVersion: apps/v1
kind: DaemonSet
metadata:
  name: gcs-fuse-models
  namespace: arc-runners
spec:
  selector:
    matchLabels:
      app: gcs-fuse
  template:
    metadata:
      labels:
        app: gcs-fuse
    spec:
      containers:
      - name: gcsfuse
        image: gcr.io/gke-release/gcs-fuse-csi-driver-sidecar-mounter:v1.4.2
        securityContext:
          privileged: true
        env:
        - name: BUCKET_NAME
          value: "${PROJECT_ID}-sglang-models"
        volumeMounts:
        - name: models-mount
          mountPath: /models
          mountPropagation: Bidirectional
      volumes:
      - name: models-mount
        hostPath:
          path: /models
          type: DirectoryOrCreate
```

#### Option B: Filestore (Better for Concurrent Access)

```bash
# Create Filestore instance
gcloud filestore instances create sglang-models \
  --zone=$ZONE \
  --tier=BASIC_HDD \
  --file-share=name=models,capacity=1TB \
  --network=name=default

# Get IP address
FILESTORE_IP=$(gcloud filestore instances describe sglang-models \
  --zone=$ZONE --format="value(networks[0].ipAddresses[0])")
```

Create PersistentVolume:

```yaml
# filestore-pv.yaml
apiVersion: v1
kind: PersistentVolume
metadata:
  name: models-pv
spec:
  capacity:
    storage: 1Ti
  accessModes:
    - ReadOnlyMany
  nfs:
    server: "${FILESTORE_IP}"
    path: /models
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: models-pvc
  namespace: arc-runners
spec:
  accessModes:
    - ReadOnlyMany
  storageClassName: ""
  volumeName: models-pv
  resources:
    requests:
      storage: 1Ti
```

### Phase 3: Actions Runner Controller Setup

#### 3.1 Install ARC

```bash
# Get cluster credentials
gcloud container clusters get-credentials sglang-arc-runners --zone=$ZONE

# Create namespace
kubectl create namespace arc-systems
kubectl create namespace arc-runners

# Install ARC using Helm
helm repo add actions-runner-controller \
  https://actions-runner-controller.github.io/actions-runner-controller
helm repo update

# Install controller
helm install arc \
  --namespace arc-systems \
  oci://ghcr.io/actions/actions-runner-controller-charts/gha-runner-scale-set-controller \
  --version 0.9.3
```

#### 3.2 Create GitHub App Secret

```bash
# If using GitHub App
kubectl create secret generic github-app-secret \
  --namespace=arc-runners \
  --from-literal=github_app_id=YOUR_APP_ID \
  --from-literal=github_app_installation_id=YOUR_INSTALLATION_ID \
  --from-file=github_app_private_key=path/to/private-key.pem

# If using PAT
kubectl create secret generic github-pat-secret \
  --namespace=arc-runners \
  --from-literal=github_token=$(gcloud secrets versions access latest --secret=github-arc-token)
```

#### 3.3 Deploy Runner Scale Sets

**TPU v6e-1 Runner Scale Set:**

```yaml
# arc-runner-v6e-1.yaml
apiVersion: actions.summerwind.dev/v1alpha1
kind: RunnerDeployment
metadata:
  name: arc-runner-v6e-1
  namespace: arc-runners
spec:
  replicas: 0  # Autoscaling handles this
  template:
    spec:
      repository: alexshires/sglang-jax
      labels:
        - arc-runner-v6e-1

      # GitHub App authentication
      githubAPICredentialsFrom:
        secretRef:
          name: github-app-secret

      # Pod spec
      containers:
      - name: runner
        image: ghcr.io/actions/actions-runner:latest
        resources:
          requests:
            memory: "16Gi"
            cpu: "4"
            google.com/tpu: "1"
          limits:
            memory: "32Gi"
            cpu: "8"
            google.com/tpu: "1"
        volumeMounts:
        - name: models
          mountPath: /models
          readOnly: true
        - name: work
          mountPath: /home/runner/_work
        env:
        - name: SGLANG_JAX_IS_IN_CI
          value: "true"
        - name: JAX_PLATFORMS
          value: "tpu"

      volumes:
      - name: models
        persistentVolumeClaim:
          claimName: models-pvc
      - name: work
        emptyDir:
          sizeLimit: 100Gi

      # Schedule on TPU nodes
      nodeSelector:
        cloud.google.com/gke-tpu-topology: "1x1"
        cloud.google.com/gke-tpu-accelerator: tpu-v6e-slice

      tolerations:
      - key: "google.com/tpu"
        operator: "Equal"
        value: "present"
        effect: "NoSchedule"

---
apiVersion: actions.summerwind.dev/v1alpha1
kind: HorizontalRunnerAutoscaler
metadata:
  name: arc-runner-v6e-1-autoscaler
  namespace: arc-runners
spec:
  scaleTargetRef:
    name: arc-runner-v6e-1
  minReplicas: 0
  maxReplicas: 4
  metrics:
  - type: TotalNumberOfQueuedAndInProgressWorkflowRuns
    repositoryNames:
    - alexshires/sglang-jax
```

**TPU v6e-4 Runner Scale Set:**

```yaml
# arc-runner-v6e-4.yaml
apiVersion: actions.summerwind.dev/v1alpha1
kind: RunnerDeployment
metadata:
  name: arc-runner-v6e-4
  namespace: arc-runners
spec:
  replicas: 0
  template:
    spec:
      repository: alexshires/sglang-jax
      labels:
        - arc-runner-v6e-4

      githubAPICredentialsFrom:
        secretRef:
          name: github-app-secret

      containers:
      - name: runner
        image: ghcr.io/actions/actions-runner:latest
        resources:
          requests:
            memory: "64Gi"
            cpu: "16"
            google.com/tpu: "4"
          limits:
            memory: "128Gi"
            cpu: "32"
            google.com/tpu: "4"
        volumeMounts:
        - name: models
          mountPath: /models
          readOnly: true
        - name: work
          mountPath: /home/runner/_work
        env:
        - name: SGLANG_JAX_IS_IN_CI
          value: "true"
        - name: JAX_PLATFORMS
          value: "tpu"

      volumes:
      - name: models
        persistentVolumeClaim:
          claimName: models-pvc
      - name: work
        emptyDir:
          sizeLimit: 100Gi

      nodeSelector:
        cloud.google.com/gke-tpu-topology: "2x2"
        cloud.google.com/gke-tpu-accelerator: tpu-v6e-slice

      tolerations:
      - key: "google.com/tpu"
        operator: "Equal"
        value: "present"
        effect: "NoSchedule"

---
apiVersion: actions.summerwind.dev/v1alpha1
kind: HorizontalRunnerAutoscaler
metadata:
  name: arc-runner-v6e-4-autoscaler
  namespace: arc-runners
spec:
  scaleTargetRef:
    name: arc-runner-v6e-4
  minReplicas: 0
  maxReplicas: 2
  metrics:
  - type: TotalNumberOfQueuedAndInProgressWorkflowRuns
    repositoryNames:
    - alexshires/sglang-jax
```

#### 3.4 Apply Configurations

```bash
kubectl apply -f gcs-fuse-daemonset.yaml  # or filestore-pv.yaml
kubectl apply -f arc-runner-v6e-1.yaml
kubectl apply -f arc-runner-v6e-4.yaml
```

### Phase 4: Verify Setup

```bash
# Check runners are registered
kubectl get runners -n arc-runners

# Check pods (should be 0 when idle)
kubectl get pods -n arc-runners

# Trigger a test workflow
gh workflow run pr-test.yml --repo alexshires/sglang-jax

# Watch pods scale up
kubectl get pods -n arc-runners -w

# Check runner logs
kubectl logs -n arc-runners -l app=arc-runner-v6e-1 -f
```

## Cost Analysis

### Infrastructure Costs (Monthly)

| Component | Spec | On-Demand | With Spot/Preemptible |
|-----------|------|-----------|----------------------|
| GKE Control Plane | Standard | $73 | $73 |
| GKE System Nodes | e2-standard-4 x 2 | $97 | $29 |
| TPU v6e-1 (per hour active) | ct6e-standard-1t | $3.22/hr | $0.97/hr |
| TPU v6e-4 (per hour active) | ct6e-standard-4t | $12.88/hr | $3.86/hr |
| Filestore (1TB Basic HDD) | - | $204 | $204 |
| **Base monthly (no CI runs)** | | **$374** | **$306** |

### CI Run Costs (Estimated)

| Scenario | TPU Hours/Month | Spot Cost | Notes |
|----------|-----------------|-----------|-------|
| Light (10 PRs) | ~5 hrs | ~$20 | 30 min avg per PR |
| Medium (50 PRs) | ~25 hrs | ~$100 | Active development |
| Heavy (100 PRs) | ~50 hrs | ~$200 | Multiple contributors |

### Cost Optimization Strategies

1. **Use Spot/Preemptible VMs** - 70% cost reduction
2. **Scale to zero when idle** - Only pay for active CI runs
3. **Share runners across repos** - If you have multiple forks
4. **Use GCS FUSE instead of Filestore** - Pay only for storage used (~$26/TB/month)

**Estimated total: $330-500/month** for medium usage with spot instances.

## Models to Pre-Download

Based on test requirements:

| Model | Size | Used By |
|-------|------|---------|
| `Wan-AI/Wan2.1-T2V-1.3B-Diffusers` | ~5GB | `test_vae_scheduler.py`, multimodal tests |
| `Wan-AI/Wan2.1-T2V-14B-Diffusers` | ~28GB | Large model tests |
| `meta-llama/Llama-3.2-1B-Instruct` | ~2.5GB | Score API tests, benchmarks |
| `google/gemma-2-2b-it` | ~5GB | Various tests |

```bash
# Script to download all required models
#!/bin/bash
MODELS=(
  "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
  "Wan-AI/Wan2.1-T2V-14B-Diffusers"
  "meta-llama/Llama-3.2-1B-Instruct"
  "google/gemma-2-2b-it"
)

for model in "${MODELS[@]}"; do
  echo "Downloading $model..."
  huggingface-cli download "$model" --local-dir "/models/$model"
done
```

## Maintenance

### Monitoring

```bash
# Check runner status
kubectl get runners -n arc-runners

# Check autoscaler status
kubectl get hra -n arc-runners

# View runner logs
kubectl logs -n arc-runners -l runner-deployment-name=arc-runner-v6e-1

# Check node pool status
gcloud container node-pools list --cluster=sglang-arc-runners --zone=$ZONE
```

### Updating ARC

```bash
# Check for updates
helm repo update

# Upgrade ARC controller
helm upgrade arc \
  --namespace arc-systems \
  oci://ghcr.io/actions/actions-runner-controller-charts/gha-runner-scale-set-controller \
  --version NEW_VERSION
```

### Troubleshooting

| Issue | Diagnosis | Solution |
|-------|-----------|----------|
| Runners not picking up jobs | Check webhook delivery in GitHub | Verify GitHub App permissions |
| TPU not available | Node pool not scaling | Check TPU quota, node pool config |
| Model not found | `/models/` not mounted | Check PVC, GCS FUSE status |
| OOM errors | Insufficient memory | Increase resource limits |

## Security Considerations

1. **GitHub App over PAT** - More granular permissions, rotatable
2. **Workload Identity** - No long-lived service account keys
3. **Private GKE cluster** - Optional, adds network isolation
4. **Secret Manager** - Store sensitive values encrypted
5. **Read-only model mount** - Prevent accidental modification

## Alternatives Considered

### Alternative 1: GitHub-Hosted Larger Runners with TPU

**Status:** Not available - GitHub doesn't offer TPU-attached runners.

### Alternative 2: Cloud Build with TPU

**Pros:** Native GCP integration, simpler than ARC
**Cons:** Different config format, not GitHub Actions compatible
**Why rejected:** Want compatibility with existing workflows

### Alternative 3: Self-Managed VMs (Non-Kubernetes)

**Pros:** Simpler, no K8s overhead
**Cons:** Manual scaling, no autoscaling, more maintenance
**Why rejected:** ARC provides better automation and scale-to-zero

## Implementation Checklist

### Phase 1: GCP Infrastructure
- [ ] Enable required APIs
- [ ] Request TPU quota (if needed)
- [ ] Create GKE cluster
- [ ] Create TPU v6e-1 node pool
- [ ] Create TPU v6e-4 node pool

### Phase 2: Storage
- [ ] Create GCS bucket or Filestore
- [ ] Download required models
- [ ] Set up GCS FUSE or NFS mount
- [ ] Verify `/models/` accessible from pods

### Phase 3: ARC Setup
- [ ] Create GitHub App or PAT
- [ ] Install ARC controller
- [ ] Deploy runner scale sets
- [ ] Configure autoscaling

### Phase 4: Validation
- [ ] Verify runners appear in GitHub
- [ ] Trigger test workflow
- [ ] Confirm TPU tests pass
- [ ] Confirm model loading works

### Phase 5: Documentation
- [ ] Document runbook for common operations
- [ ] Set up monitoring/alerting
- [ ] Create cost tracking dashboard

## Open Questions

1. **Shared vs dedicated runners?** - Could share with other sglang forks
2. **Which models are actually required?** - Need to audit all tests
3. **Preemptible tolerance?** - How to handle spot instance preemption mid-test
4. **HF_TOKEN handling?** - Need to inject for gated model downloads

## References

- [Actions Runner Controller Documentation](https://github.com/actions/actions-runner-controller)
- [GKE TPU User Guide](https://cloud.google.com/kubernetes-engine/docs/how-to/tpus)
- [GCS FUSE CSI Driver](https://cloud.google.com/kubernetes-engine/docs/how-to/persistent-volumes/cloud-storage-fuse-csi-driver)
- [GitHub Self-Hosted Runners](https://docs.github.com/en/actions/hosting-your-own-runners)
- Upstream workflow: `sgl-project/sglang-jax/.github/workflows/pr-test.yml`
