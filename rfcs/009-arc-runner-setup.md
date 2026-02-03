# RFC-009: Self-Hosted ARC Runners with TPU for Fork CI

| | |
|------------|------|
| **Status** | Accepted |
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
| Runners | Self-hosted ARC on GKE | Self-hosted ARC on GKE (This RFC) |
| TPU access | TPU v6e node pools | TPU v5e/v6e node pools |
| Model storage | `/models/` via GCS FUSE | `/models/` via GCS FUSE |
| CI status | Passing | Passing |

## Goals

1. **Full CI parity** with upstream sgl-project/sglang-jax
2. **TPU v6e runners** with labels `arc-runner-v6e-1`, `arc-runner-v6e-1-standard`, `arc-runner-v6e-4`, and `arc-runner-v5e-4`
3. **GPU runners** with label `gpu-runner`
4. **Persistent model storage** at `/models/` path via GCS FUSE
5. **Reproducible setup** with automated installation scripts

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
│  │  │ arc-runner-v6e-1 │    │ arc-runner-v5e-4 │              ││
│  │  │   (min: 0)       │    │   (min: 0)       │              ││
│  │  └────────┬─────────┘    └────────┬─────────┘              ││
│  │           │                       │                         ││
│  └───────────┼───────────────────────┼─────────────────────────┘│
│              │                       │                          │
│  ┌───────────▼───────────┐  ┌───────▼───────────┐             │
│  │   TPU v6e-1 Node Pool │  │ TPU v5e-4 Node Pool│             │
│  │   (autoscaling 0-4)   │  │ (autoscaling 0-2)  │             │
│  │                       │  │                    │             │
│  │  ┌─────────────────┐  │  │ ┌────────────────┐ │             │
│  │  │  Runner Pod     │  │  │ │  Runner Pod    │ │             │
│  │  │  + TPU v6e-1    │  │  │ │  + TPU v5e-4   │ │             │
│  │  │  + /models/     │  │  │ │  + /models/    │ │             │
│  │  └─────────────────┘  │  │ └────────────────┘ │             │
│  └───────────────────────┘  └────────────────────┘             │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                    GCS FUSE Mount                            ││
│  │                    /models/ mount                            ││
│  │  Bucket: ashires-e7aaot-model-download-europe-west4          ││
│  │  - Wan-AI/Wan2.1-T2V-1.3B-Diffusers                        ││
│  │  - meta-llama/Llama-3.2-1B-Instruct                        ││
│  │  - ...                                                       ││
│  └─────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
```

## Implementation

### Phase 1: Automation Script

The entire setup is automated via `sglang-scripts/scripts/install_gh_runners.sh`.

```bash
# Usage:
# ./sglang-scripts/scripts/install_gh_runners.sh <GITHUB_CONFIG_URL> [GITHUB_TOKEN]

# Example:
# ./sglang-scripts/scripts/install_gh_runners.sh https://github.com/alexshires/sglang-jax
```

This script:
1. Connects to the GKE cluster `ray-tpu-test-cluster` in `europe-west4`.
2. Creates namespaces `arc-systems` and `arc-runners`.
3. Sets up GitHub PAT secrets.
4. Installs the ARC Controller.
5. Installs multiple Runner Scale Sets (GPU, TPU v6e, TPU v5e).

### Phase 2: Runner Resource Configurations

The configurations are stored in `sglang-scripts/k8s/` and include GCS FUSE sidecar support and TPU/GPU resource requests.

#### 2.1 TPU v6e-1 Spot Runner (`arc-runner-v6e-1`)
Used for standard PR unit and E2E tests. Uses spot instances to reduce cost.
- **Node Selector:** `pool_type: "tpu-standard"`, `cloud.google.com/gke-spot: "true"`
- **Resources:** 1 TPU chip, 40 CPUs, 150Gi RAM.

#### 2.2 TPU v6e-1 Standard Runner (`arc-runner-v6e-1-standard`)
Used for non-preemptible workloads or where spot availability is low.
- **Node Selector:** `pool_type: "tpu-standard"`
- **Resources:** 1 TPU chip, 40 CPUs, 150Gi RAM.

#### 2.3 TPU v6e-4 Runner (`arc-runner-v6e-4`)
Used for multi-chip tests requiring a 2x2 topology on v6e chips. Uses spot instances.
- **Node Selector:** `pool_type: "tpu-standard"`, `cloud.google.com/gke-tpu-topology: "2x2"`, `cloud.google.com/gke-spot: "true"`
- **Resources:** 4 TPU chips, 100 CPUs, 150Gi RAM.

#### 2.4 TPU v5e-4 Runner (`arc-runner-v5e-4`)
Used for multi-chip tests requiring a 2x2 topology.
- **Node Selector:** `pool_type: "tpu-v5e-spot"`, `cloud.google.com/gke-tpu-topology: "2x2"`
- **Resources:** 4 TPU chips, 100 CPUs, 150Gi RAM.

#### 2.4 GPU Runner (`gpu-runner`)
Used for CUDA-based testing.
- **Node Selector:** `pool_type: "gpu-spot-l4"`
- **Resources:** 1 L4 GPU.

### Phase 3: Model Storage (GCS FUSE)

All runners mount the GCS bucket `ashires-e7aaot-model-download-europe-west4` to `/models` using the GKE GCS FUSE CSI driver.

**Key Mount Option:** `only-dir=huggingface_models`

The models must be organized hierarchically to avoid `HFValidationError`:
```
/models/
├── Wan-AI/
│   └── Wan2.1-T2V-1.3B-Diffusers/
├── meta-llama/
│   └── Llama-3.2-1B-Instruct/
└── Qwen/
    └── Qwen3-0.6B/
```

## Maintenance & Troubleshooting

### Monitoring Pods
```bash
kubectl get pods -n arc-runners
```

### Viewing Runner Logs
```bash
kubectl logs -n arc-runners -c runner <pod-name>
```

### Common Issues
1. **TPU Not Scaling:** Check quota in `europe-west4` for `TPU_V6E_LITE_PODSLICE_CHIPS`.
2. **Model Found but Error:** Ensure the directory structure in GCS matches `Namespace/Repo` exactly (no flattened underscores).
3. **OOM Score Adj:** `ACTIONS_RUNNER_DISABLE_OOM_SCORE_ADJ: "true"` is set to prevent the runner from being killed by the Linux OOM killer too aggressively.

## References

- Deployment Script: `sglang-scripts/scripts/install_gh_runners.sh`
- Values Folder: `sglang-scripts/k8s/`
- GKE GCS FUSE Documentation: [Cloud Storage FUSE CSI driver](https://cloud.google.com/kubernetes-engine/docs/how-to/persistent-volumes/cloud-storage-fuse-csi-driver)
