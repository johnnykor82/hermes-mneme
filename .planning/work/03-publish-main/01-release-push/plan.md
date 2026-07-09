# Plan: Release Push

## Level
task

## Parent
`.planning/work/03-publish-main/plan.md`

## Status
in_progress

## Goal

Publish the tested compatibility commit to GitHub `main`.

---

## Spec Coverage

| Req ID | Requirement Summary |
|--------|---------------------|
| SC-006 | Git hygiene checks pass |
| SC-007 | Tested commit is pushed |
| NFR-006 | Runtime artifacts excluded |

---

## Active Task / Step

Active Task: none (leaf)

---

## Steps

- [ ] **Step 1:** Review final status and diff.
  - *Verification:* `git status --short --branch` and `git diff --stat` show only intentional changes.
- [ ] **Step 2:** Commit compatibility update.
  - *Verification:* `git show --stat --oneline HEAD` shows the expected commit.
- [ ] **Step 3:** Push to GitHub `main`.
  - *Verification:* `git push origin main` succeeds.
- [ ] **Step 4:** Confirm remote `main` points at the tested commit.
  - *Verification:* `git ls-remote origin refs/heads/main` matches local `HEAD`.

---

## Spec Compliance

| Req ID | Status | Verification Evidence |
|--------|--------|-----------------------|
| SC-006 | pending | |
| SC-007 | pending | |
| NFR-006 | pending | |

**Compliance Status: PENDING**

---

## Decisions Made

| Decision | Rationale |
|----------|-----------|
| Do not commit runtime artifacts | Required by spec boundaries |

---

## Errors Encountered

| Error | Attempt | Resolution |
|-------|---------|------------|
| GitHub Actions run `28973003465` failed during pytest collection: `ModuleNotFoundError: No module named 'agent'` in `tests/unit/test_current_hermes_compat.py`. | 1 | Added standalone Hermes test stubs in `tests/conftest.py` when `agent.context_engine` is unavailable; local workflow command now passes. |

---

## Notes

- If push is rejected, stop and report before rebasing or force-pushing.
