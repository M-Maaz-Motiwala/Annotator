#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

export SKIP_MODEL_LOAD=1

python -m pip install -r requirements-test.txt
python -m pytest "$@"
