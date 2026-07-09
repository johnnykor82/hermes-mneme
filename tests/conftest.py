"""Pytest bootstrap.

Plugin directory has a hyphen (`hermes-mneme`) which isn't a valid Python
identifier, but internal modules use relative imports (`from . import store`).
We register the plugin directory under the alias `hermes_mneme` in sys.modules
so tests can do `from hermes_mneme import classifier`. Hermes itself loads
the plugin through its own discovery — this shim is test-only.
"""

from __future__ import annotations

import importlib.util
import os
import pathlib
import sys
import types
from typing import Any, Dict, Optional

_PLUGIN_DIR = pathlib.Path(__file__).resolve().parent.parent
_PKG_NAME = "hermes_mneme"


def _ensure_hermes_test_stubs() -> None:
    """Let standalone package tests import modules that normally run in Hermes."""
    try:
        import agent.context_engine  # noqa: F401
    except ModuleNotFoundError:
        agent_pkg = sys.modules.get("agent") or types.ModuleType("agent")
        ctx_mod = types.ModuleType("agent.context_engine")

        class ContextEngine:
            last_prompt_tokens: int = 0
            last_completion_tokens: int = 0
            last_total_tokens: int = 0
            threshold_tokens: int = 0
            context_length: int = 0
            compression_count: int = 0
            threshold_percent: float = 0.75
            protect_first_n: int = 3
            protect_last_n: int = 6

            def __init__(self) -> None:
                self.agent = None

            def on_session_reset(self) -> None:
                pass

            def get_status(self) -> Dict[str, Any]:
                return {}

        ctx_mod.ContextEngine = ContextEngine
        agent_pkg.context_engine = ctx_mod
        sys.modules.setdefault("agent", agent_pkg)
        sys.modules["agent.context_engine"] = ctx_mod

    try:
        import hermes_logging  # noqa: F401
    except ModuleNotFoundError:
        hl_mod = types.ModuleType("hermes_logging")

        class _SessionContext:
            session_id: Optional[str] = None

        hl_mod._session_context = _SessionContext()
        sys.modules["hermes_logging"] = hl_mod


_ensure_hermes_test_stubs()

# Also keep raw plugin dir on sys.path for any legacy `import classifier`
# style imports, but the canonical way is `from hermes_mneme import X`.
plugin_path = str(_PLUGIN_DIR)
if plugin_path not in sys.path:
    sys.path.insert(0, plugin_path)

if _PKG_NAME not in sys.modules:
    spec = importlib.util.spec_from_file_location(
        _PKG_NAME,
        _PLUGIN_DIR / "__init__.py",
        submodule_search_locations=[str(_PLUGIN_DIR)],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[_PKG_NAME] = module
    spec.loader.exec_module(module)
