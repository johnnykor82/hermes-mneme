# Spec: hermes-mneme Compatibility With Current Hermes Context Engine

Version: 0.1.1
Status: APPROVED
Date: 2026-07-08
Repository Baseline: `hermes-mneme@1e306f6f49c1657a2bef09c947f3d973dc095e45`

## Assumptions
1. The immediate target is the published legacy `hermes-mneme` plugin, not `_hermes-mneme-native`.
2. The native-hooks copy remains unpublished until the Hermes PR with `prepare_request_messages` and `on_turn_complete` is merged upstream.
3. The compatibility fix must work with current live Hermes, where plugin context engines are deep-copied before model/session use.
4. Runtime data such as `db/`, `trace.jsonl`, caches, and live Hermes config/state must not be committed.
5. GitHub publication will push the tested update to `main`.

## Objective
Restore stable operation of `hermes-mneme` as Hermes' configured `context.engine` under the current Hermes release line.

The user should be able to run `hermes` with `context.engine: hermes-mneme` and see Hermes select Mneme instead of falling back to the built-in compressor. Success means the plugin can be safely copied during Hermes agent initialization, accepts the current compression call shape, keeps existing Mneme behavior intact, and passes focused regression and diagnostic tests.

## Tech Stack
- Python >= 3.10
- Hermes plugin API via `register(ctx)` and `CustomRouterContextEngine`
- SQLite-backed Mneme store with optional `sqlite-vec`
- Dependencies from `pyproject.toml`: `sqlite-vec`, `tiktoken`, `numpy`, `requests`, `pyyaml`
- Test runner: `pytest`
- Live validation interpreter: `/Users/openclaw/.hermes/hermes-agent/venv/bin/python3`

## Commands
Baseline and regression tests:

```bash
/Users/openclaw/.hermes/hermes-agent/venv/bin/python3 -m pytest tests/unit tests/integration/test_store.py
```

Diagnostic invariants:

```bash
/Users/openclaw/.hermes/hermes-agent/venv/bin/python3 -m tests.diagnostic.reproduce_lineage_loss
/Users/openclaw/.hermes/hermes-agent/venv/bin/python3 -m tests.diagnostic.reproduce_state_bugs
```

Syntax and packaging checks:

```bash
/Users/openclaw/.hermes/hermes-agent/venv/bin/python3 -m py_compile __init__.py engine.py config.py tools.py prompt_builder.py
git diff --check
git status --short --branch
```

Current-Hermes compatibility smoke, using a temporary home rather than live state:

```bash
HERMES_HOME=/private/tmp/hermes-mneme-compat-smoke /Users/openclaw/.local/bin/hermes -z "mneme compatibility smoke"
```

## Project Structure
- `__init__.py` - Hermes plugin registration entry point.
- `engine.py` - `CustomRouterContextEngine`; main compatibility work belongs here.
- `config.py` - plugin configuration wrapper; may need copy-safety support.
- `tools.py` - Mneme tool integration exposed to Hermes.
- `store.py`, `index.py`, `router.py`, `prompt_builder.py` - retrieval and context assembly internals; should remain behaviorally unchanged unless a regression test proves otherwise.
- `tests/unit/` - focused unit regressions for deepcopy and compression signature compatibility.
- `tests/integration/` - existing store/integration coverage.
- `tests/diagnostic/` - scenario reproducers that must remain green.
- `.planning/` - specification, roadmap, task plans, findings, and progress for this compatibility effort.

## Code Style
Follow the current flat-module style, small helper methods, and explicit runtime-state handling. Prefer narrow compatibility methods over broad refactors.

Example target style:

```python
def __deepcopy__(self, memo):
    clone = type(self)(db_path=self.store.db_path, config=self.config.as_dict)
    memo[id(self)] = clone
    clone.context_length = self.context_length
    clone.threshold_tokens = self.threshold_tokens
    return clone

def compress(self, messages, current_tokens=None, focus_topic=None, force=False, **kwargs):
    ...
```

The exact implementation may differ after inspection, but it must stay explicit about what is copied, reset, and intentionally not copied.

## Functional Requirements
- FR-001: `copy.deepcopy(CustomRouterContextEngine(...))` must complete without `RecursionError` or copying unsafe runtime resources directly.
- FR-002: The copied engine must be a distinct, usable engine instance with fresh runtime resources such as executors, tool bindings, and database/store wrappers.
- FR-003: The copied engine must preserve model/budget state needed after Hermes calls `update_model`, including context length, threshold tokens, budget tokens, and token-accounting state where relevant.
- FR-004: Per-session mutable state must not leak incorrectly from the plugin singleton into copied agent instances.
- FR-005: `compress()` must accept Hermes' current call shape, including `focus_topic` and `force`, without relying on Hermes' TypeError fallback.
- FR-006: Existing legacy behavior must remain intact: `should_compress()` keeps driving Mneme through the legacy compression hook until native Hermes hooks are available.
- FR-007: No native-hooks implementation from `_hermes-mneme-native` is merged into this published compatibility fix.

## Non-Functional Requirements
- NFR-001: Keep the change narrowly scoped to compatibility and regression coverage.
- NFR-002: Avoid destructive changes to live Hermes, live Mneme DBs, or user configuration.
- NFR-003: Do not add new third-party dependencies.
- NFR-004: Preserve current public plugin metadata and tool names.
- NFR-005: Tests must be runnable locally with the existing Hermes virtualenv.
- NFR-006: Git history must identify the compatibility baseline and avoid committing runtime artifacts.

## Testing Strategy
Add focused regression tests before or alongside implementation:

- Unit test that reproduces the current deepcopy failure and verifies the fixed copy behavior.
- Unit test that verifies copied engine state isolation after `update_model`.
- Unit test or direct behavioral test that `compress(..., force=True)` is accepted.
- Existing unit and integration tests must keep passing.
- Existing diagnostic modules must report all invariants passed.
- A final temporary-home Hermes smoke should verify that Hermes no longer logs fallback from `hermes-mneme` to the built-in compressor.

Embedding endpoint warnings from diagnostics are acceptable when sandboxed network access blocks `127.0.0.1:8000`, as long as the diagnostic invariants pass.

## Boundaries
- Always: work from `1e306f6`, keep edits minimal, add regression tests for compatibility behavior, bump the plugin version to the next patch version, run listed tests before pushing to `main`.
- Ask first: modifying live Hermes source/config, changing database schema, adding dependencies, merging native-hooks code, or changing the approved publication target away from `main`.
- Never: commit `db/`, `trace.jsonl`, `.pytest_cache/`, `__pycache__/`, secrets, live Hermes config, or unrelated planning artifacts from the earlier native-hooks effort.

## Success Criteria
- SC-001: A direct deepcopy reproduction that currently fails with recursion succeeds after the change.
- SC-002: Hermes context-engine selection no longer falls back because `hermes-mneme` could not be safely copied.
- SC-003: `compress()` accepts the current Hermes `force` keyword.
- SC-004: The baseline test command passes.
- SC-005: Both diagnostic modules finish with all invariants passed.
- SC-006: `git diff --check` passes and `git status --short --branch` shows only intentional source, test, and `.planning` changes.
- SC-007: The final pushed GitHub state corresponds to the tested commit.

## Open Questions
None.

## Change Log
| Version | Date | Change |
|---------|------|--------|
| 0.1.0 | 2026-07-08 | Initial draft spec for current Hermes compatibility fix. |
| 0.1.1 | 2026-07-08 | Approved spec; publication target set to GitHub `main`; version bump required. |
