# TorchScript

TorchScript enables inference using C++. Installation methods vary by platform.

**x86 - [LibTorch](https://pytorch.org/get-started/locally/)**

The CPU version is sufficient for most laptop deployments. For faster inference, install the GPU version (ensure CUDA compatibility).

```bash
cd /opt
wget -nc https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-2.10.0%2Bcpu.zip
# wget -nc https://download.pytorch.org/libtorch/cu126/libtorch-shared-with-deps-2.10.0%2Bcu126.zip
unzip -n libtorch-shared-with-deps-2.10.0+cpu.zip
echo 'export CMAKE_PREFIX_PATH=/opt/libtorch:$CMAKE_PREFIX_PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/opt/libtorch/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
```

**Jetson - [PyTorch](https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform/index.html#install-multiple-versions-pytorch)**

LibTorch is not available for Jetson platforms. Install PyTorch first, then set `Torch_dir` to the PyTorch installation directory.

Check your JetPack version:
```bash
cat /etc/nv_tegra_release
```

Find wheels at https://developer.download.nvidia.com/compute/redist/jp/

```bash
export TORCH_INSTALL=https://developer.download.nvidia.com/compute/redist/jp/v$JP_VERSION/pytorch/$PYT_VERSION
uv pip install $TORCH_INSTALL
```

Set CMake environment variables:
```bash
export CMAKE_PREFIX_PATH="$(uv run python3 -c 'import torch; print(torch.utils.cmake_prefix_path)'):${CMAKE_PREFIX_PATH}"
export LD_LIBRARY_PATH="$(uv run python3 -c 'import torch, pathlib; p=pathlib.Path(torch.__file__).resolve().parent; print(p / "lib")'):${LD_LIBRARY_PATH}"
```