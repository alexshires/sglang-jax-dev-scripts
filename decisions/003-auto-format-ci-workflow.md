# ADR-003: Add Auto-Format CI Workflow for PRs

| | |
|------------|------|
| **Date** | 2026-02-04 |
| **Status** | Proposed |
| **Deciders** | Engineering Team |
| **Related** | [Upstream auto-format.yml](https://github.com/sgl-project/sglang/blob/main/.github/workflows/auto-format.yml) |

## Context

When PRs have minor formatting issues (whitespace, import order, line length), the lint CI blocks the PR and requires the contributor to manually fix and push again. This creates unnecessary friction, especially for external contributors or quick fixes.

### Current State Across Repos

| Repo | Lint Check | Auto-Fix | Trigger |
|------|-----------|----------|---------|
| `sgl-project/sglang` (PyTorch) | `lint.yml` — blocks on failure | `auto-format.yml` — commits fixes | `format` label added to PR |
| `sgl-project/sglang-jax` (upstream JAX) | `lint.yml` — blocks on failure | **None** | — |
| `alexshires/sglang-jax` (fork) | `lint.yml` — blocks on failure | **None** | — |

The PyTorch repo already solved this with a label-triggered `auto-format.yml` workflow. Neither the upstream nor fork sglang-jax repos have this capability.

### Existing Toolchain

Both sglang-jax repos already have:
- `.pre-commit-config.yaml` with isort, black, ruff, codespell, clang-format, nbstripout, mypy
- `lint.yml` workflow that runs `pre-commit run --all-files --show-diff-on-failure`

The auto-fix hooks (`--fix` for ruff, auto-formatting by black/isort) already exist in the pre-commit config — they just aren't used in CI because the workflow doesn't commit results back.

## Decision

Adopt the **same label-triggered auto-format pattern** used by `sgl-project/sglang` (PyTorch).

Add `.github/workflows/auto-format.yml` to `sglang-jax` with:
1. **Trigger:** PR labeled with `format`
2. **Action:** Run pre-commit with `--all-files`, commit and push fixes
3. **Cleanup:** Remove the `format` label after completion

### Workflow

```yaml
name: Auto Format Code

on:
  pull_request:
    types: [labeled]

permissions:
  contents: write
  pull-requests: write

jobs:
  auto-format:
    if: github.event.label.name == 'format'
    runs-on: ubuntu-latest
    steps:
      - name: Checkout PR branch
        uses: actions/checkout@v4
        with:
          ref: ${{ github.event.pull_request.head.ref }}
          repository: ${{ github.event.pull_request.head.repo.full_name }}
          token: ${{ secrets.GITHUB_TOKEN }}
          fetch-depth: 0

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.12"

      - name: Install pre-commit
        run: |
          python -m pip install pre-commit
          pre-commit install

      - name: Run pre-commit to format code
        run: SKIP=no-commit-to-branch pre-commit run --all-files
        continue-on-error: true

      - name: Check for changes
        id: check_changes
        run: |
          if [[ -n $(git status -s) ]]; then
            echo "has_changes=true" >> $GITHUB_OUTPUT
          else
            echo "has_changes=false" >> $GITHUB_OUTPUT
          fi

      - name: Commit and push changes
        if: steps.check_changes.outputs.has_changes == 'true'
        run: |
          git config --local user.email "github-actions[bot]@users.noreply.github.com"
          git config --local user.name "github-actions[bot]"
          git add .
          git commit -m "Auto-format code with isort, black, ruff, and clang-format"
          git push

      - name: Remove format label
        if: always()
        uses: actions/github-script@v7
        with:
          github-token: ${{ secrets.GITHUB_TOKEN }}
          script: |
            try {
              await github.rest.issues.removeLabel({
                owner: context.repo.owner,
                repo: context.repo.repo,
                issue_number: context.issue.number,
                name: 'format'
              });
            } catch (error) {
              console.log('Label may have already been removed');
            }
```

### User Flow

1. Contributor opens PR
2. `lint.yml` fails due to formatting issues
3. Maintainer (or contributor) adds the `format` label to the PR
4. `auto-format.yml` triggers, fixes formatting, pushes commit
5. Label is removed automatically
6. `lint.yml` re-runs on the new commit and passes

## Consequences

### Positive
- **Reduces friction** for contributors — no manual reformat-and-push cycle
- **Consistent with upstream** — same pattern as `sgl-project/sglang`
- **Opt-in, not automatic** — only runs when explicitly requested via label
- **No new tooling** — reuses the existing `.pre-commit-config.yaml`
- **Safe** — maintainers control when auto-format runs

### Negative
- **Requires manual label step** — someone must add the `format` label
- **Does not work on forked PRs** — `pull_request` events from forks receive a read-only `GITHUB_TOKEN`, so `git push` will fail. This means auto-format only works for PRs from branches within the same repo (collaborator PRs). This is the same limitation the upstream `sgl-project/sglang` workflow has and accepts. For forked PRs, contributors must fix formatting locally.

### Neutral
- **Does not replace local pre-commit** — developers should still run `pre-commit install` locally
- **Commit history** — adds a formatting commit to the PR (acceptable trade-off)

## Alternatives Considered

### Alternative 1: Auto-fix on every PR (no label trigger)
**Description:** Run auto-format on every `pull_request` event, not just when labeled.

**Pros:**
- Fully automatic, zero manual intervention

**Cons:**
- Noisy — pushes formatting commits even when not needed
- Can surprise contributors with unexpected commits
- Risk of infinite loops if not carefully guarded
- Runs on every PR update (push, reopen, etc.)

**Why rejected:** Too aggressive. The label-based approach gives maintainers control and matches upstream.

### Alternative 2: pre-commit.ci (third-party service)
**Description:** Use [pre-commit.ci](https://pre-commit.ci) GitHub App for automatic fixes.

**Pros:**
- Zero workflow config needed
- Free for open-source
- Built-in auto-fix and auto-update of hook versions

**Cons:**
- External dependency / third-party service
- Less control over behavior
- Not used by upstream `sgl-project/sglang`
- Requires GitHub App installation (org-level approval)

**Why rejected:** Adds external dependency. Upstream doesn't use it. The in-repo workflow approach is simpler and self-contained.

### Alternative 3: Auto-fix only on lint failure
**Description:** Chain workflows — if `lint.yml` fails, automatically trigger auto-format.

**Pros:**
- Only runs when needed
- No manual label step

**Cons:**
- Complex workflow chaining (`workflow_run` triggers)
- Harder to debug
- Can still surprise contributors

**Why rejected:** Over-engineered for the use case. Label trigger is simpler and proven in upstream.

## Implementation Notes

- **Create the `format` label** in the GitHub repo (can be done via UI or `gh label create format`)
- **Runner:** Uses `ubuntu-latest` (not `arc-runner-cpu`) since formatting doesn't need special hardware
- **SKIP=no-commit-to-branch:** Required to bypass the `no-commit-to-branch` pre-commit hook that prevents commits to `main`
- **Deploy to fork first** (`alexshires/sglang-jax`), then propose for upstream (`sgl-project/sglang-jax`)

## Validation

After implementation, verify:
1. Adding `format` label triggers the workflow
2. Formatting fixes are committed and pushed to the PR branch
3. Label is removed after workflow completes
4. `lint.yml` passes on the subsequent commit
5. Workflow is a no-op when there are no formatting changes

## References

- [Upstream auto-format.yml](https://github.com/sgl-project/sglang/blob/main/.github/workflows/auto-format.yml)
- [sglang-jax .pre-commit-config.yaml](https://github.com/alexshires/sglang-jax/blob/main/.pre-commit-config.yaml)
- [GitHub Actions: Triggering workflows with labels](https://docs.github.com/en/actions/writing-workflows/choosing-when-your-workflow-runs/events-that-trigger-workflows#pull_request)
