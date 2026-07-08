# Roadmap: hermes-mneme Current Hermes Compatibility

## Project Goal

Restore and publish `hermes-mneme` compatibility with the current Hermes legacy context engine API from baseline `1e306f6`.

---

## Planning Gate

- [x] `spec.md` created and **approved** by user
- [x] All phases listed below with estimated effort (even future stubs)
- [x] `plan.md` exists for every phase
- [x] `plan.md` exists for every task in the **current** phase (detailed, not stubs)
- [x] Stub `plan.md` exists for tasks in future phases
- [x] If using a non-template spec: not applicable; spec was created from the approved template shape
- [x] User has reviewed and approved the plan structure

**Gate Status: CLEARED - 2026-07-08**

---

## Active Phase

Active Phase: 03-publish-main

---

## Phases

### 01-compatibility-fix
- **Status:** complete
- **Goal:** Add regression coverage and a narrow compatibility fix for current Hermes context-engine copying and compression calls.
- **Spec coverage:** FR-001, FR-002, FR-003, FR-004, FR-005, FR-006, FR-007, NFR-001, NFR-003, NFR-005
- **Estimated effort:** 1 focused session
- **Plan:** `.planning/work/01-compatibility-fix/plan.md`
- **Tasks:** 3

### 02-hermes-smoke
- **Status:** complete
- **Goal:** Prove Hermes selects `hermes-mneme` in a temporary home and does not fall back to the built-in compressor.
- **Spec coverage:** SC-002, NFR-002, NFR-005
- **Estimated effort:** 20-40 minutes
- **Plan:** `.planning/work/02-hermes-smoke/plan.md`
- **Tasks:** 1 stub

### 03-publish-main
- **Status:** in_progress
- **Goal:** Commit the tested compatibility update and push it to GitHub `main`.
- **Spec coverage:** SC-006, SC-007, NFR-006
- **Estimated effort:** 15-30 minutes
- **Plan:** `.planning/work/03-publish-main/plan.md`
- **Tasks:** 1 stub

---

## Progress Summary

| Metric | Value |
|--------|-------|
| Total phases | 3 |
| Complete | 2 |
| In progress | 1 |
| Pending | 0 |
| Cancelled | 0 |
| Last updated | 2026-07-08 |

---

## Decisions Made

| Decision | Rationale | Phase | Date |
|----------|-----------|-------|------|
| Publish to GitHub `main` after tests pass | User approved direct main publication | 03-publish-main | 2026-07-08 |
| Bump plugin version to next patch version | User requested version bump after update | 01-compatibility-fix | 2026-07-08 |
| Exclude `_hermes-mneme-native` from this update | Native hooks depend on an unmerged Hermes PR | 01-compatibility-fix | 2026-07-08 |

---

## Risks & Open Questions

| Risk / Question | Impact | Resolution | Status |
|-----------------|--------|------------|--------|
| Temporary Hermes smoke may need local model/network access | Could block final live-like validation | Run all local tests first; request escalation only for smoke/push if needed | open |
| Current local branch may not track `origin/main` | Could make push command less obvious | Use explicit `git push origin main` after final status check | open |

---

## Spec Change Log

| Date | Spec Change | Phases Affected | Action Taken |
|------|-------------|-----------------|--------------|
| 2026-07-08 | Spec approved; publish target `main`; version bump required | 01, 03 | Roadmap includes version and publication tasks |
