#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

TORCH_WHL="https://github.com/ultralytics/assets/releases/download/v0.0.0/torch-2.5.0a0+872d972e41.nv24.08-cp310-cp310-linux_aarch64.whl"
TORCHVISION_WHL="https://github.com/ultralytics/assets/releases/download/v0.0.0/torchvision-0.20.0a0+afc54f7-cp310-cp310-linux_aarch64.whl"
ONNXRUNTIME_WHL="https://github.com/ultralytics/assets/releases/download/v0.0.0/onnxruntime_gpu-1.20.0-cp310-cp310-linux_aarch64.whl"
CUDA_KEYRING_DEB="cuda-keyring_1.1-1_all.deb"
CUDA_KEYRING_URL="https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/arm64/${CUDA_KEYRING_DEB}"

echo "Installing Jetson Python stack for SUSPEKT Demonstrator"
echo "Project: $ROOT_DIR"
echo

sudo apt update
sudo apt install -y python3-pip wget

python3 -m pip install --upgrade --user pip setuptools wheel

# Keep Ultralytics on the version currently used in this project setup.
python3 -m pip install --force-reinstall --user "ultralytics==8.3.148"

# Replace generic torch/torchvision with Jetson-compatible NVIDIA wheels.
python3 -m pip uninstall -y torch torchvision
python3 -m pip install --user "$TORCH_WHL"
python3 -m pip install --user "$TORCHVISION_WHL"

wget -O "/tmp/${CUDA_KEYRING_DEB}" "$CUDA_KEYRING_URL"
sudo dpkg -i "/tmp/${CUDA_KEYRING_DEB}"
sudo apt-get update
sudo apt-get install -y libcusparselt0 libcusparselt-dev

python3 -m pip install --user "$ONNXRUNTIME_WHL"

# Reinstall the project so pinned app dependencies (for example numpy/pillow) win.
python3 -m pip install -e "$ROOT_DIR" --user

echo
echo "Jetson dependency installation complete."
