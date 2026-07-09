# Progress: hermes-mneme Current Hermes Compatibility

## 2026-07-08

- Created and approved `.planning/spec.md`.
- User confirmed final publication should go to GitHub `main`.
- User confirmed plugin version should be bumped to the next version.
- Created roadmap and task hierarchy for planning gate review.
- No production code has been changed yet.

## 2026-07-08 Execution Start

- User approved the plan structure.
- Planning gate cleared.
- Active phase: `01-compatibility-fix`.
- Active task: `01-regression-tests`.

## 2026-07-08 Regression Tests

- Added `tests/unit/test_current_hermes_compat.py`.
- Verified red baseline: targeted pytest produced 3 expected failures.
- Failure evidence: `copy.deepcopy()` hits `RecursionError`; `compress(..., force=True)` hits `TypeError`.
- Active task advanced to `02-implementation`.

## 2026-07-08 Implementation

- Added explicit `CustomRouterContextEngine.__deepcopy__`.
- Updated `compress()` to accept `force=False` and future kwargs.
- Bumped plugin metadata to `0.2.1`.
- Targeted compatibility tests passed: `3 passed`.
- Active task advanced to `03-local-verification`.

## 2026-07-08 Local Verification

- Full local pytest passed: `34 passed`.
- Diagnostic lineage reproducer passed all invariants.
- Diagnostic state-bugs reproducer passed all invariants.
- `py_compile` and `git diff --check` passed.
- Phase `01-compatibility-fix` marked complete.
- Active phase advanced to `02-hermes-smoke`.

## 2026-07-08 Hermes Smoke

- Prepared temporary home at `/private/tmp/hermes-mneme-compat-smoke-20260708`.
- Actual `hermes -z` with temp home registered `hermes-mneme` and reached the local API boundary.
- Programmatic Hermes plugin-selection smoke from `/private/tmp` returned `hermes-mneme`, deep-copied it, and ran `update_model()`.
- No `could not be safely copied` fallback warning appeared in temp logs.
- Phase `02-hermes-smoke` marked complete.
- Active phase advanced to `03-publish-main`.

## 2026-07-09 CI Failure Follow-up

- GitHub Actions run `28973003465` failed in pytest collection on Python 3.11 and 3.12; Python 3.10 was cancelled after fail-fast.
- Root cause: CI lacks Hermes Agent's `agent.context_engine`, while the new compatibility test imports `hermes_mneme.engine`.
- Added conditional Hermes-side stubs to `tests/conftest.py` for standalone pytest environments.
- Reproduced the first boundary locally with system Python: `agent` import no longer fails; the next missing dependency is `tiktoken`, which CI installs.
- Verified local workflow command with Hermes venv: `pytest tests/ -v` -> `39 passed`.
