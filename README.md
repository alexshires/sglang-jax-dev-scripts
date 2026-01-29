# SGLang-JAX Development Scripts & Documentation

This repository contains design documents, investigations, test plans, and utility scripts for developing SGLang-JAX.

**→ New here? Start with [QUICK_START.md](QUICK_START.md) for a guided tour.**

## Purpose

**Design before code.** This repo helps us:
- Think through problems before implementing
- Document decisions for future reference
- Share designs for review and feedback
- Track the evolution of our thinking
- Onboard new contributors

## Repository Structure

```
sglang-jax-dev-scripts/
├── rfcs/              # Request for Comments (design proposals)
├── decisions/         # Architecture Decision Records (ADRs)
├── investigations/    # Deep dives & research
├── scripts/           # Utility scripts
├── test-plans/        # Test strategies
└── runbooks/          # Operational guides
```

## Workflow

### 1. Before Coding: Create an RFC

For any non-trivial feature or change:

```bash
cp rfcs/template.md rfcs/NNN-my-feature.md
# Fill in the RFC
# Get feedback from team
# Iterate on design
```

### 2. Document Key Decisions

Create ADRs for important architectural choices:

```bash
cp decisions/template.md decisions/NNN-key-decision.md
# Document what, why, and alternatives
```

### 3. During Coding

Reference RFCs in commits:
```bash
git commit -m "feat: implement X (RFC-003)"
```

Update RFC status as you progress:
- Draft → Review → Accepted → Implementing → Implemented

### 4. After Coding

- Update RFC with actual results vs plan
- Create runbook if needed for operations
- Archive or supersede outdated decisions

## Document Types

### RFCs (Request for Comments)
**What:** Design proposals for features, changes, or improvements
**When:** Before implementing anything non-trivial
**Example:** "RFC-001: Comprehensive Testing for /v1/score API"

### ADRs (Architecture Decision Records)
**What:** Record of a specific decision and its rationale
**When:** After making an important architectural choice
**Example:** "ADR-001: Use Pure Python Softmax in tokenizer_manager"

### Investigations
**What:** Deep analysis, comparisons, or research
**When:** Exploring a problem space or comparing options
**Example:** "Comparison: JAX vs PyTorch tokenizer_manager"

### Test Plans
**What:** Strategy for testing a feature or component
**When:** Before writing tests for new functionality
**Example:** "Score API Test Plan"

### Runbooks
**What:** Step-by-step operational guides
**When:** For recurring operational tasks
**Example:** "Debugging TPU Test Failures"

## Best Practices

1. **Write RFCs early** - Before you're committed to an approach
2. **Keep ADRs immutable** - Don't edit, supersede if needed
3. **Be concise** - Clear writing forces clear thinking
4. **Include code examples** - Show, don't just tell
5. **Update status** - Keep documents current

## Contributing

1. Use templates for consistency
2. Get at least one review before "Accepted" status
3. Link related documents together
4. Keep a consistent numbering scheme

## Integration with Main Repo

The main sglang-jax repository references these docs:
- `CONTRIBUTING.md` links to RFC process
- PR descriptions reference relevant RFCs
- Design decisions documented in ADRs

## Examples

See existing documents for examples:
- **RFC-001:** Score API comprehensive tests
- **RFC-002:** CI/CD for TPU testing
- **ADR-001:** Pure Python softmax decision

---

*This repository follows the RFC/ADR pattern used by successful open-source projects like Rust, Python (PEPs), and Kubernetes (KEPs).*
