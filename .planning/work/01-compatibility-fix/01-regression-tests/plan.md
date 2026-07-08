# Plan: Regression Tests

## Level
task

## Parent
`.planning/work/01-compatibility-fix/plan.md`

## Status
complete

## Goal

Capture the current Hermes compatibility failures before changing plugin behavior.

---

## Spec Coverage

| Req ID | Requirement Summary |
|--------|---------------------|
| FR-001 | Engine deepcopy succeeds |
| FR-002 | Copied engine has fresh runtime resources |
| FR-003 | Copied engine preserves model and budget state |
| FR-004 | Session state does not leak |
| FR-005 | `compress()` accepts `force` |

---

## Active Task / Step

Active Task: none (leaf)

---

## Steps

- [x] **Step 1:** Inspect existing pytest fixtures and instantiate `CustomRouterContextEngine` with a temporary DB and low-side-effect config.
  - *Verification:* `tests/unit/test_current_hermes_compat.py` defines `_build_engine()` with temp DB and low-side-effect config.
- [x] **Step 2:** Add a regression test for `copy.deepcopy()` success, runtime-resource isolation, and preserved model/budget fields.
  - *Verification:* Targeted pytest failed on baseline with `RecursionError`.
- [x] **Step 3:** Add a regression test proving copied engines do not inherit active session state from the source singleton.
  - *Verification:* Targeted pytest failed on baseline with `RecursionError` before clone assertions.
- [x] **Step 4:** Add a regression test proving `compress(..., force=True)` is accepted.
  - *Verification:* Targeted pytest failed on baseline with `TypeError: unexpected keyword argument 'force'`.

---

## Spec Compliance

| Req ID | Status | Verification Evidence |
|--------|--------|-----------------------|
| FR-001 | ✓ met | Regression test fails on baseline with `RecursionError`. |
| FR-002 | ✓ met | Regression test asserts distinct store/index/tools/executors. |
| FR-003 | ✓ met | Regression test asserts preserved budget/model/token state. |
| FR-004 | ✓ met | Regression test asserts copied engine resets session state. |
| FR-005 | ✓ met | Regression test fails on baseline with `force` TypeError. |

**Compliance Status: VERIFIED - 2026-07-08**

---

## Decisions Made

| Decision | Rationale |
|----------|-----------|
| Use temp DBs for compatibility tests | Avoid live Mneme state and keep tests deterministic |

---

## Errors Encountered

| Error | Attempt | Resolution |
|-------|---------|------------|

---

## Notes

- This task may intentionally produce failing tests before implementation.
