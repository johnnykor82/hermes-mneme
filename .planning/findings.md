# Findings: hermes-mneme Current Hermes Compatibility

## 2026-07-08

- Live Hermes config still points `context.engine` at `hermes-mneme`, but current Hermes falls back before using it.
- Current Hermes deep-copies plugin context engines during agent initialization before `update_model`.
- Published `hermes-mneme@1e306f6` fails direct `copy.deepcopy(CustomRouterContextEngine(...))` with `RecursionError`.
- The first observed recursion source is `PluginConfig.__getattr__`, but the engine also owns runtime resources such as DB wrappers, tools, and thread executors, so an explicit engine-level `__deepcopy__` is safer than recursively copying the whole object graph.
- Current Hermes calls `compress(..., focus_topic=..., force=...)`; published Mneme accepts `focus_topic` but not `force`.
- Hermes catches the `force` `TypeError` and retries without it, so missing `force` is secondary, not the current fallback cause.
- Baseline test command passed before implementation: `31 passed`.
- Diagnostic module `tests.diagnostic.reproduce_lineage_loss` passed all invariants before implementation.
- Diagnostic module `tests.diagnostic.reproduce_state_bugs` passed all invariants before implementation.
- Diagnostic embedding endpoint warnings are expected in sandboxed runs when `127.0.0.1:8000` is blocked; stored-event invariants still pass.
- `_hermes-mneme-native` intentionally remains unpublished and is not sufficient for this fix because it also lacks current-Hermes deepcopy compatibility.
- Version metadata in `pyproject.toml` and `plugin.yaml` is `0.2.0`, while `CHANGELOG.md` already contains a `0.3.0` section. Per the approved plan, this compatibility fix bumps metadata to `0.2.1` and does not broaden scope into changelog/version-history cleanup.
- Current Hermes copies the plugin singleton before `update_model()` and does not require the plugin copy to carry `agent` state. Fresh clone semantics are correct for this compatibility fix.
- Programmatic smoke must not run from the plugin directory because local `tools.py` shadows Hermes' `tools` package. Running the same check from `/private/tmp` succeeds.
- Temporary-home smoke result: plugin discovery registered `hermes-mneme`; `copy.deepcopy()` succeeded; `update_model()` produced context length `128000` and budget `89600`; no fallback warning was logged.
- GitHub Actions failure for run `28973003465` was a pytest collection error, not a runtime regression: `tests/unit/test_current_hermes_compat.py` imported `hermes_mneme.engine`, which imports `agent.context_engine`. CI installs `hermes-mneme` standalone and does not have the Hermes Agent `agent` package.
- Existing diagnostic scripts already had the correct pattern: install minimal Hermes-side stubs before importing the plugin engine. The shared pytest bootstrap now uses the same pattern only when real Hermes modules are absent.
