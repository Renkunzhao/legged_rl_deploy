# LibTorch

LibTorch enables inference using C++. Installation methods vary by platform.

## PC (x86_64)

Download from: https://pytorch.org/get-started/locally/

The CPU version is sufficient for most laptop deployments.

```bash
cd /opt

# CPU
LIBTORCH_URL="https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-2.10.0%2Bcpu.zip"
LIBTORCH_ZIP="libtorch-shared-with-deps-2.10.0+cpu.zip"

# GPU example (pick the CUDA version that matches your system)
# LIBTORCH_URL="https://download.pytorch.org/libtorch/cu126/libtorch-shared-with-deps-2.10.0%2Bcu126.zip"
# LIBTORCH_ZIP="libtorch-shared-with-deps-2.10.0+cu126.zip"

sudo wget -nc -O "${LIBTORCH_ZIP}" "${LIBTORCH_URL}"
sudo unzip -n "${LIBTORCH_ZIP}"

export CMAKE_PREFIX_PATH="/opt/libtorch:${CMAKE_PREFIX_PATH}"
export LD_LIBRARY_PATH="/opt/libtorch/lib:${LD_LIBRARY_PATH}"
```

## Jetson (aarch64)

LibTorch is typically not provided for Jetson. Install PyTorch first, then point CMake at the Torch CMake package.

### Install PyTorch

#### CPU
For CPU-only, a simple install may work:

```bash
uv pip install torch numpy
```

**Note:** A CPU-only PyTorch build on Jetson is often insufficient for real-time deployments.

#### GPU (on Jetson host)

Reference: https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform/index.html#install-multiple-versions-pytorch

Check your JetPack version (archive: https://developer.nvidia.com/embedded/jetpack-archive):

```bash
cat /etc/nv_tegra_release
```

Example output:
    
    # R35 (release), REVISION: 3.1, GCID: 32827747, BOARD: t186ref, EABI: aarch64, DATE: Sun Mar 19 15:19:21 UTC 2023
This indicates L4T 35.3.1 (JetPack 5.1.1). Then pick the corresponding wheel from:

https://developer.download.nvidia.com/compute/redist/jp/

```bash
uv pip install <whl_url>
```

#### GPU (inside a Jetson container)

Use an NVIDIA L4T base image that matches the host JetPack/L4T version.

### Cmake

After PyTorch is installed, set CMake-related environment variables:

```bash
export CMAKE_PREFIX_PATH="$(uv run python3 -c 'import torch; print(torch.utils.cmake_prefix_path)'):${CMAKE_PREFIX_PATH}"
export LD_LIBRARY_PATH="$(uv run python3 -c 'import torch, pathlib; p=pathlib.Path(torch.__file__).resolve().parent; print(p / "lib")'):${LD_LIBRARY_PATH}"
```