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

_PLUGIN_DIR = pathlib.Path(__file__).resolve().parent.parent
_PKG_NAME = "hermes_mneme"

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
