# Plan: Temporary Home Smoke

## Level
task

## Parent
`.planning/work/02-hermes-smoke/plan.md`

## Status
complete

## Goal

Verify current Hermes can initialize and use the updated plugin without the deepcopy fallback warning.

---

## Spec Coverage

| Req ID | Requirement Summary |
|--------|---------------------|
| SC-002 | Hermes no longer falls back from `hermes-mneme` |
| NFR-002 | Avoid destructive live-state changes |
| NFR-005 | Existing Hermes install validates behavior |

---

## Active Task / Step

Active Task: none (leaf)

---

## Steps

- [x] **Step 1:** Prepare a temporary Hermes home and plugin copy.
  - *Verification:* `/private/tmp/hermes-mneme-compat-smoke-20260708` contains minimal config and copied plugin without original `db/trace/.git`.
- [x] **Step 2:** Run one minimal Hermes command with `HERMES_HOME` pointed at the temporary home.
  - *Verification:* `hermes -z` reached the expected external-service boundary: local API call failed after plugin registration.
- [x] **Step 3:** Inspect temporary logs and plugin trace for selection behavior.
  - *Verification:* Temp logs show plugin registration; programmatic Hermes selection from `/private/tmp` deep-copied and updated `hermes-mneme` without fallback.

---

## Spec Compliance

| Req ID | Status | Verification Evidence |
|--------|--------|-----------------------|
| SC-002 | ✓ met | `get_plugin_context_engine()` returned `hermes-mneme`; `copy.deepcopy()` and `update_model()` succeeded; no fallback warning in temp logs. |
| NFR-002 | ✓ met | Smoke used `/private/tmp/hermes-mneme-compat-smoke-20260708`, not live `~/.hermes`. |
| NFR-005 | ✓ met | Smoke used `/Users/openclaw/.local/bin/hermes` and Hermes venv Python. |

**Compliance Status: VERIFIED - 2026-07-08**

---

## Decisions Made

| Decision | Rationale |
|----------|-----------|
| Smoke after local verification | Avoid noisy live-like runs until unit diagnostics prove the fix |

---

## Errors Encountered

| Error | Attempt | Resolution |
|-------|---------|------------|
| Programmatic smoke run from plugin cwd shadowed Hermes `tools.registry` with local `tools.py` | 1 | Re-ran from `/private/tmp`; smoke passed. |

---

## Notes

- If model access blocks the smoke, record the exact boundary and rely on initialization/log evidence where possible.
