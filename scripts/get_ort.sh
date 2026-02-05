#!/usr/bin/env bash

PROJECT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"/../
echo "Project directory: $PROJECT_DIR"

cd $PROJECT_DIR
mkdir -p thirdparty/onnxruntime
cd thirdparty

ARCH="$(uname -m | sed 's/x86_64/x64/')"
TGZ="onnxruntime-linux-${ARCH}-1.23.2.tgz"
URL="https://github.com/microsoft/onnxruntime/releases/download/v1.23.2/${TGZ}"

wget -nc "$URL"
tar -xzf "$TGZ" --strip-components=1 -C onnxruntime

cd $PROJECT_DIR/scripts
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
uv venv --clear
uv pip install pyyaml numpy onnxscript torch --index https://download.pytorch.org/whl/cpu
