#!/usr/bin/env bash
# setup_starpu_starfwi.sh — Build and install StarPU and StarFWI on a cluster node.
#
# Run this AFTER the base AMI setup (amazon_linux_setup_cpu_only.sh).
# Both StarPU and StarFWI are cloned automatically from their upstream repos.
#
# Usage:
#   bash starpu_and_starfwi_setup_cpu_only.sh
#
# Overrides:
#   INSTALL_PREFIX    Installation prefix (default: /usr/local)
#   STARPU_REPO       StarPU git URL
#   STARPU_BRANCH     StarPU branch (default: spot-cloud-support)
#   STARFWI_REPO      StarFWI git URL
#   STARFWI_BRANCH    StarFWI branch (default: main)

set -euo pipefail

STARPU_REPO="${STARPU_REPO:-https://gitlab.inria.fr/starpu/starpu.git}"
STARPU_BRANCH="${STARPU_BRANCH:-spot-cloud-support}"
STARPU_SRC_DIR="/tmp/starpu"
STARFWI_REPO="${STARFWI_REPO:-https://github.com/vanderlei-filho/starfwi.git}"
STARFWI_BRANCH="${STARFWI_BRANCH:-main}"
STARFWI_SRC_DIR="/tmp/starfwi"
INSTALL_PREFIX="${INSTALL_PREFIX:-/usr/local}"
SHARED_DIR="${SHARED_DIR:-/shared}"

# ── 1. StarPU ─────────────────────────────────────────────────────────────────
echo ""
echo "==> Cloning StarPU (branch '${STARPU_BRANCH}')..."

rm -rf "$STARPU_SRC_DIR"
git clone --branch "$STARPU_BRANCH" --depth 1 "$STARPU_REPO" "$STARPU_SRC_DIR"

echo "==> Building StarPU..."
cd "$STARPU_SRC_DIR"
./autogen.sh
./configure \
    --prefix="$INSTALL_PREFIX" \
    --disable-opencl \
    --disable-build-examples \
    --disable-build-doc \
    --enable-mpi \
    --enable-mpi-ft \
    --with-fxt="$INSTALL_PREFIX" \
    --with-mpicc="$INSTALL_PREFIX/bin/mpicc" \
    --with-hwloc="$INSTALL_PREFIX"
make -j$(nproc)
sudo make install
sudo ldconfig

echo "==> StarPU installed successfully."
cd / && rm -rf "$STARPU_SRC_DIR"

# ── 2. StarFWI ────────────────────────────────────────────────────────────────
echo ""
echo "==> Cloning StarFWI (branch '${STARFWI_BRANCH}')..."

rm -rf "$STARFWI_SRC_DIR"
git clone --branch "$STARFWI_BRANCH" --depth 1 "$STARFWI_REPO" "$STARFWI_SRC_DIR"

echo "==> Building StarFWI..."

# GCC 14 is used for StarFWI (matches Dockerfile)
export CC=/usr/bin/gcc14-gcc
export CXX=/usr/bin/gcc14-g++
export PKG_CONFIG_PATH="${INSTALL_PREFIX}/lib/pkgconfig:${INSTALL_PREFIX}/lib64/pkgconfig${PKG_CONFIG_PATH:+:${PKG_CONFIG_PATH}}"

cd "$STARFWI_SRC_DIR"
rm -rf build
cmake -B build -S . \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
cmake --build build -- -j$(nproc)

sudo cp build/starfwi-modeling "$INSTALL_PREFIX/bin/starfwi-modeling"
sudo cp build/starfwi-fwi      "$INSTALL_PREFIX/bin/starfwi-fwi"
sudo chmod +x \
    "$INSTALL_PREFIX/bin/starfwi-modeling" \
    "$INSTALL_PREFIX/bin/starfwi-fwi"

echo "==> StarFWI installed successfully."
