# Multi-Item Scoring v1 Blocker Fix Artifacts (2026-02-14)

Artifacts for PR #24 blocker-fix validation.

## Directory map

- `phase1_repro_clean/`: deterministic clean reproduction for B2/C3
- `phase2_b2_fix/`: B2 fail-fast/recovery validation
- `phase3_c3_tuning/`: stable-knob tuning sweep matrix
- `phase4_gates_rerun_v3/`: final required gates used for GO/NO-GO

Additional historical reruns kept for traceability:

- `phase4_gates/`
- `phase4_gates_rerun/`
- `phase4_gates_rerun_v2/`

## Final gate bundle (authoritative)

Use `phase4_gates_rerun_v3/` for final decision evidence:

- `phase4_non_soak_summary.json`
- `gate_c2_nlte100_10m_summary.json`
- `gate_recovery_*_summary.json`
- `gate_soak_2h_c2_mixed_summary.json`
- `gate_bench_large_5m_summary.json`
