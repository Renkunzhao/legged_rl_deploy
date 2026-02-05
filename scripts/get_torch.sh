#!/usr/bin/env bash

PY_VERSION="${1:-}"
WHL_LINK="${2:-}"

PROJECT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"/../
echo "Project directory: $PROJECT_DIR"

sudo apt-get install -y libgfortran5 libopenblas0

cd $PROJECT_DIR
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
uv venv --python "$PY_VERSION" --clear
uv pip install numpy 

if [[ -n "$WHL_LINK" ]]; then
  echo "Installing torch from input: $WHL_LINK"
  uv pip install "$WHL_LINK"
else
  echo "No input provided, installing default torch from PyPI..."
  uv pip install torch
fi
