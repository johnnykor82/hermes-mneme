# Plan: Compatibility Fix

## Level
phase

## Parent
`.planning/roadmap.md`

## Status
complete

## Goal

Make the published legacy plugin compatible with current Hermes context-engine initialization and compression calls.

---

## Spec Coverage

| Req ID | Requirement Summary |
|--------|---------------------|
| FR-001 | Engine deepcopy succeeds |
| FR-002 | Copied engine has fresh runtime resources |
| FR-003 | Copied engine preserves model and budget state |
| FR-004 | Session state does not leak from singleton |
| FR-005 | `compress()` accepts `force` |
| FR-006 | Legacy compression-hook behavior remains active |
| FR-007 | Native-hooks code stays out of this release |
| NFR-001 | Narrow compatibility scope |
| NFR-003 | No new dependencies |
| NFR-005 | Tests run in existing Hermes venv |

---

## Active Task / Step

Active Task: none

---

## Tasks

### 01-regression-tests
- **Status:** complete
- **Goal:** Add focused tests that capture the current incompatibilities.
- **Spec coverage:** FR-001, FR-002, FR-003, FR-004, FR-005
- **Plan:** `./01-regression-tests/plan.md`
- **Acceptance criteria:**
  - [x] Tests fail on the current baseline for the known compatibility gap.
  - [x] Tests are narrow and avoid live DB/config writes.

### 02-implementation
- **Status:** complete
- **Goal:** Implement copy safety, compression-call compatibility, and version bump.
- **Spec coverage:** FR-001, FR-002, FR-003, FR-004, FR-005, FR-006, FR-007, NFR-001, NFR-003
- **Plan:** `./02-implementation/plan.md`
- **Acceptance criteria:**
  - [x] `copy.deepcopy()` succeeds and creates isolated runtime resources.
  - [x] `compress(..., force=True)` is accepted.
  - [x] Version metadata moves from `0.2.0` to `0.2.1`.

### 03-local-verification
- **Status:** complete
- **Goal:** Run local regression, integration, diagnostic, syntax, and git hygiene checks.
- **Spec coverage:** SC-001, SC-003, SC-004, SC-005, SC-006, NFR-005, NFR-006
- **Plan:** `./03-local-verification/plan.md`
- **Acceptance criteria:**
  - [x] All listed local tests pass.
  - [x] `git diff --check` passes.
  - [x] Only intentional files are changed; pre-existing unrelated untracked docs are left untouched.

---

## Spec Compliance

| Req ID | Status | Verification Evidence |
|--------|--------|-----------------------|
| FR-001 | ✓ met | Direct deepcopy regression test passed. |
| FR-002 | ✓ met | Test verifies distinct runtime resources. |
| FR-003 | ✓ met | Test verifies preserved model/budget/token fields. |
| FR-004 | ✓ met | Test verifies copied engine resets active session fields. |
| FR-005 | ✓ met | Force-keyword regression test passed. |
| FR-006 | ✓ met | `should_compress()` remains unchanged and local diagnostics passed. |
| FR-007 | ✓ met | No native-hooks implementation included. |
| NFR-001 | ✓ met | Narrow production diff: `engine.py`, metadata files, tests. |
| NFR-003 | ✓ met | No dependencies added. |
| NFR-005 | ✓ met | Full test suite and diagnostics ran in existing Hermes venv. |

**Compliance Status: VERIFIED - 2026-07-08**

---

## Decisions Made

| Decision | Rationale |
|----------|-----------|
| Use explicit `CustomRouterContextEngine.__deepcopy__` | Avoid copying unsafe runtime graph and match current Hermes expectations |
| Keep native-hooks code out | It depends on unmerged Hermes API |

---

## Errors Encountered

| Error | Attempt | Resolution |
|-------|---------|------------|

---

## Notes

- This phase does not modify live Hermes or live Mneme runtime data.
