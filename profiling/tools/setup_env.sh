#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV_DIR="${ROOT_DIR}/.venv-profile"

python3 -m venv "${VENV_DIR}"
# shellcheck disable=SC1090
source "${VENV_DIR}/bin/activate"
python -m pip install --upgrade pip
python -m pip install -r "${ROOT_DIR}/profiling/tools/requirements.txt"

cat <<EOF
Profile analysis environment created at:
  ${VENV_DIR}
Activate it with:
  source ${VENV_DIR}/bin/activate
EOF
