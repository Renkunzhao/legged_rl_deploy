#!/usr/bin/env bash

sudo apt-get install -y libgfortran5 libopenblas0

PY_VERSION="${1:-}"
WHL_LINK="${2:-}"

PROJECT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"/../
echo "Project directory: $PROJECT_DIR"

cd $PROJECT_DIR

if [[ "$(uname -m)" == "x86_64" ]]; then

  mkdir -p thirdparty
  cd thirdparty
  wget -nc https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-2.10.0%2Bcpu.zip
  unzip -n libtorch-shared-with-deps-2.10.0+cpu.zip

else

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

fi
