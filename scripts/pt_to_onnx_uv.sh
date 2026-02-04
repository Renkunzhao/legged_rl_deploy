#!/usr/bin/env bash

PROJECT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"/../
echo "Project directory: $PROJECT_DIR"

export PATH="$HOME/.local/bin:$PATH"

cd $PROJECT_DIR
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv --clear
uv pip install pyyaml numpy onnxscript torch --index https://download.pytorch.org/whl/cpu
