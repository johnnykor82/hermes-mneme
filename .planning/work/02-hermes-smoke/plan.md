# Plan: Hermes Smoke

## Level
phase

## Parent
`.planning/roadmap.md`

## Status
complete

## Goal

Run a live-like Hermes validation in a temporary home without mutating the user's live Hermes state.

---

## Spec Coverage

| Req ID | Requirement Summary |
|--------|---------------------|
| SC-002 | Hermes no longer falls back from `hermes-mneme` |
| NFR-002 | Avoid destructive live-state changes |
| NFR-005 | Validation uses existing Hermes install |

---

## Active Task / Step

Active Task: none

---

## Tasks

### 01-temp-home-smoke
- **Status:** complete
- **Goal:** Check Hermes startup/turn logs for `hermes-mneme` selection without fallback.
- **Spec coverage:** SC-002, NFR-002, NFR-005
- **Plan:** `./01-temp-home-smoke/plan.md`
- **Acceptance criteria:**
  - [x] Hermes smoke runs in a temporary home.
  - [x] Logs do not contain context-engine copy fallback for `hermes-mneme`.

---

## Spec Compliance

| Req ID | Status | Verification Evidence |
|--------|--------|-----------------------|
| SC-002 | ✓ met | Temporary-home Hermes plugin selection deep-copied and updated `hermes-mneme`; no fallback warning. |
| NFR-002 | ✓ met | Smoke used `/private/tmp/hermes-mneme-compat-smoke-20260708`. |
| NFR-005 | ✓ met | Smoke used installed Hermes CLI and Hermes venv. |

**Compliance Status: VERIFIED - 2026-07-08**

---

## Decisions Made

| Decision | Rationale |
|----------|-----------|
| Use temporary `HERMES_HOME` | Avoid mutating live logs, DB, and config |

---

## Errors Encountered

| Error | Attempt | Resolution |
|-------|---------|------------|

---

## Notes

- This phase may require approval for network/model access depending on Hermes configuration.
