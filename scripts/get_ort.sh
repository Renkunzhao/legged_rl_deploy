#!/usr/bin/env bash

PROJECT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"/../
echo "Project directory: $PROJECT_DIR"

mkdir -p thirdparty/onnxruntime
cd thirdparty
wget -nc https://github.com/microsoft/onnxruntime/releases/download/v1.23.2/onnxruntime-linux-$(uname -m | sed 's/x86_64/x64/')-1.23.2.tgz
tar -xzf onnxruntime-linux-$(uname -m | sed 's/x86_64/x64/')-1.23.2.tgz --strip-components=1 -C onnxruntime