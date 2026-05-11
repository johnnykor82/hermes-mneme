#!/usr/bin/env bash
# Hermes-Mneme installer.
# Detects the Hermes venv and installs plugin dependencies into it.

set -euo pipefail

PLUGIN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HERMES_VENV="${HERMES_VENV:-$HOME/.hermes/hermes-agent/venv}"

echo "Hermes-Mneme installer"
echo "  Plugin dir: $PLUGIN_DIR"
echo "  Hermes venv: $HERMES_VENV"
echo

if [[ ! -x "$HERMES_VENV/bin/pip" ]]; then
  echo "ERROR: Hermes venv not found at $HERMES_VENV"
  echo "Set HERMES_VENV=/path/to/venv and re-run, or install Hermes first."
  exit 1
fi

echo "Installing plugin dependencies into $HERMES_VENV ..."
"$HERMES_VENV/bin/pip" install -q -r "$PLUGIN_DIR/requirements.txt"

echo "Verifying installation ..."
"$HERMES_VENV/bin/python" - <<'PY'
import importlib.util, sys
mods = ["tiktoken", "yaml", "requests", "numpy", "sqlite_vec"]
missing = [m for m in mods if not importlib.util.find_spec(m)]
if missing:
    print("MISSING:", missing); sys.exit(1)
print("OK — all deps present.")
PY

echo
echo "Hermes-Mneme installed at: $PLUGIN_DIR"
echo "Restart Hermes to activate:"
echo "  hermes gateway restart"
echo
echo "Verify:"
echo "  tail -f ~/.hermes/logs/agent.log | grep -i mneme"
