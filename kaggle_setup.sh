#!/usr/bin/env bash
set -euo pipefail

# CMake >= 3.24 (Kaggle ships 3.22)
wget -qO - https://apt.kitware.com/keys/kitware-archive-latest.asc \
    | gpg --dearmor -o /usr/share/keyrings/kitware-archive-keyring.gpg
echo 'deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] https://apt.kitware.com/ubuntu/ jammy main' \
    > /etc/apt/sources.list.d/kitware.list

# clangd-18 (Kaggle ships 14)
wget -qO - https://apt.llvm.org/llvm-snapshot.gpg.key | apt-key add -
echo "deb http://apt.llvm.org/jammy/ llvm-toolchain-jammy-18 main" \
    > /etc/apt/sources.list.d/llvm.list

apt-get update -qq
apt-get install -y cmake clangd-18 clang-format libopencv-dev ninja-build
ln -sf /usr/bin/clangd-18 /usr/local/bin/clangd

# uv + python deps
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv sync

echo "Done. Re-export PATH or open a new shell if uv is not found."
