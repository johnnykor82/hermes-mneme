# Plan: Implementation

## Level
task

## Parent
`.planning/work/01-compatibility-fix/plan.md`

## Status
complete

## Goal

Implement the smallest compatibility change that satisfies the approved spec and regression tests.

---

## Spec Coverage

| Req ID | Requirement Summary |
|--------|---------------------|
| FR-001 | Engine deepcopy succeeds |
| FR-002 | Copied engine has fresh runtime resources |
| FR-003 | Copied engine preserves model and budget state |
| FR-004 | Session state does not leak |
| FR-005 | `compress()` accepts `force` |
| FR-006 | Legacy compression path remains active |
| FR-007 | Native hooks excluded |
| NFR-001 | Narrow scope |
| NFR-003 | No new dependencies |

---

## Active Task / Step

Active Task: none (leaf)

---

## Steps

- [x] **Step 1:** Add copy-safety support for plugin config if needed by regression tests.
  - *Verification:* Engine-level `__deepcopy__` bypasses recursive config copying; no separate `PluginConfig.__deepcopy__` was needed.
- [x] **Step 2:** Implement explicit `CustomRouterContextEngine.__deepcopy__` with fresh runtime resources and copied model/budget accounting state.
  - *Verification:* `pytest tests/unit/test_current_hermes_compat.py` passed.
- [x] **Step 3:** Reset per-session mutable fields on the copied engine while preserving global configuration/model state.
  - *Verification:* `test_deepcopy_does_not_inherit_active_session_state` passed.
- [x] **Step 4:** Update `compress()` signature to accept `force=False` and ignored future kwargs without changing current legacy behavior.
  - *Verification:* `test_compress_accepts_current_hermes_force_keyword` passed.
- [x] **Step 5:** Bump metadata from `0.2.0` to `0.2.1`.
  - *Verification:* `pyproject.toml` and `plugin.yaml` show `0.2.1`.

---

## Spec Compliance

| Req ID | Status | Verification Evidence |
|--------|--------|-----------------------|
| FR-001 | ✓ met | Targeted deepcopy regression test passed. |
| FR-002 | ✓ met | Targeted test asserts distinct store/index/tools/executors. |
| FR-003 | ✓ met | Targeted test asserts preserved model/budget/token fields. |
| FR-004 | ✓ met | Targeted test asserts session state reset on clone. |
| FR-005 | ✓ met | Targeted force-keyword regression test passed. |
| FR-006 | ✓ met | `should_compress()` remains unchanged and returns `True`. |
| FR-007 | ✓ met | No native-hooks files or APIs were imported into this change. |
| NFR-001 | ✓ met | Production edits limited to `engine.py`, `pyproject.toml`, `plugin.yaml`. |
| NFR-003 | ✓ met | No dependencies added. |

**Compliance Status: VERIFIED - 2026-07-08**

---

## Decisions Made

| Decision | Rationale |
|----------|-----------|
| Bump to `0.2.1` | Current version is `0.2.0`; compatibility fix is a patch update |

---

## Errors Encountered

| Error | Attempt | Resolution |
|-------|---------|------------|

---

## Notes

- Do not import native-hooks code from `_hermes-mneme-native`.
