#!/usr/bin/env bash
# Build the reference environments the contract tests compare cvsim with.
#
#   packages/cvsim/tools/setup_reference_envs.sh            # from the repository root
#
# The Walrus is a normal optional dependency (extra `compare`, installed by `uv sync`). MrMustard
# 0.7.3 pins NumPy < 2, TensorFlow and The Walrus 0.19, so it lives in its own virtual environment,
# .reference-envs/mrmustard (git-ignored); the tests (marker `mrmustard`) talk to it through
# tests/reference/mrmustard_bridge.py. Override the location with CVSIM_MRMUSTARD_PYTHON.
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel)"
ENV="$ROOT/.reference-envs/mrmustard"
PY="${PYTHON311:-python3.11}"   # MrMustard 0.7.3 supports Python 3.9-3.11

if [ -x "$ENV/bin/python" ] && "$ENV/bin/python" -c "import mrmustard; assert mrmustard.__version__ == '0.7.3'" 2>/dev/null; then
    echo "MrMustard reference environment already present: $ENV"
    exit 0
fi
mkdir -p "$ROOT/.reference-envs"
"$PY" -m venv "$ENV"
"$ENV/bin/python" -m pip install --quiet --upgrade pip
"$ENV/bin/python" -m pip install --quiet "mrmustard==0.7.3"
"$ENV/bin/python" -c "import mrmustard, thewalrus, numpy; print('MrMustard', mrmustard.__version__, '| The Walrus', thewalrus.__version__, '| NumPy', numpy.__version__)"
