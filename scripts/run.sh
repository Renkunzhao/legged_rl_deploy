#!/usr/bin/env bash

PROJECT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )/.."
echo "Project directory: $PROJECT_DIR"

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <network_interface> <command...>" >&2
  exit 2
fi

NET_IF="$1"
ROS_DISTRO="$2"
shift 2

source "$PROJECT_DIR/../unitree_lowlevel/scripts/setup.sh" "$NET_IF" "$ROS_DISTRO"

cd "$PROJECT_DIR"
if [[ "$(uname -m)" == "x86_64" ]]; then

  export CMAKE_PREFIX_PATH=thirdparty/libtorch:$CMAKE_PREFIX_PATH
  export LD_LIBRARY_PATH=thirdparty/libtorch/lib:$LD_LIBRARY_PATH

else

  export CMAKE_PREFIX_PATH="$(uv run python3 -c 'import torch; print(torch.utils.cmake_prefix_path)'):${CMAKE_PREFIX_PATH}"
  export LD_LIBRARY_PATH="$(uv run python3 -c 'import torch, pathlib; p=pathlib.Path(torch.__file__).resolve().parent; print(p / "lib")'):${LD_LIBRARY_PATH}"

fi

echo "CMAKE_PREFIX_PATH: $CMAKE_PREFIX_PATH"
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"

cd $PROJECT_DIR/../../
"$@"
