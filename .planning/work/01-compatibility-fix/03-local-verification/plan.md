# Plan: Local Verification

## Level
task

## Parent
`.planning/work/01-compatibility-fix/plan.md`

## Status
complete

## Goal

Verify the compatibility fix locally before any Hermes smoke or GitHub push.

---

## Spec Coverage

| Req ID | Requirement Summary |
|--------|---------------------|
| SC-001 | Direct deepcopy reproduction succeeds |
| SC-003 | `compress()` accepts `force` |
| SC-004 | Baseline tests pass |
| SC-005 | Diagnostic modules pass |
| SC-006 | Git hygiene checks pass |
| NFR-005 | Existing Hermes venv can run tests |
| NFR-006 | Runtime artifacts excluded |

---

## Active Task / Step

Active Task: none (leaf)

---

## Steps

- [x] **Step 1:** Run unit and integration regression suite.
  - *Verification:* `/Users/openclaw/.hermes/hermes-agent/venv/bin/python3 -m pytest tests/unit tests/integration/test_store.py` -> `34 passed`.
- [x] **Step 2:** Run diagnostic invariants.
  - *Verification:* Both diagnostic modules finished with `ALL INVARIANTS PASSED`.
- [x] **Step 3:** Run syntax and diff hygiene checks.
  - *Verification:* `py_compile` and `git diff --check` passed; status shows intentional tracked changes plus pre-existing unrelated untracked docs.

---

## Spec Compliance

| Req ID | Status | Verification Evidence |
|--------|--------|-----------------------|
| SC-001 | ✓ met | `tests/unit/test_current_hermes_compat.py` verifies direct deepcopy succeeds. |
| SC-003 | ✓ met | `test_compress_accepts_current_hermes_force_keyword` passed. |
| SC-004 | ✓ met | Full local pytest command -> `34 passed`. |
| SC-005 | ✓ met | Both diagnostic modules -> `ALL INVARIANTS PASSED`. |
| SC-006 | ✓ met | `git diff --check` passed; status reviewed. |
| NFR-005 | ✓ met | All tests ran using `/Users/openclaw/.hermes/hermes-agent/venv/bin/python3`. |
| NFR-006 | ✓ met | Runtime artifacts remain untracked and are not part of intended commit. |

**Compliance Status: VERIFIED - 2026-07-08**

---

## Decisions Made

| Decision | Rationale |
|----------|-----------|
| Treat sandbox embedding warnings as non-fatal | Diagnostics are designed to pass without the embedding service |

---

## Errors Encountered

| Error | Attempt | Resolution |
|-------|---------|------------|

---

## Notes

- Do not proceed to Hermes smoke until this task passes.
