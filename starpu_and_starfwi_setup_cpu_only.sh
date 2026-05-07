#!/usr/bin/env bash
# setup_starpu_starfwi.sh — Build and install StarPU and StarFWI on a cluster node.
#
# Run this AFTER the base AMI setup (amazon_linux_setup_cpu_only.sh).
# StarFWI source must be copied to the instance before running this script.
# StarPU is cloned directly from the upstream repository.
#
# Expected layout on the instance before running:
#   $STARFWI_SRC_DIR  — StarFWI source tree (default: /tmp/starfwi)
#
# Usage:
#   bash setup_starpu_starfwi.sh
#
# Overrides:
#   STARFWI_SRC_DIR   Path to StarFWI source
#   INSTALL_PREFIX    Installation prefix (default: /usr/local)

set -euo pipefail

STARPU_REPO="${STARPU_REPO:-https://gitlab.inria.fr/starpu/starpu.git}"
STARPU_BRANCH="${STARPU_BRANCH:-spot-cloud-support}"
STARPU_SRC_DIR="/tmp/starpu"
STARFWI_SRC_DIR="${STARFWI_SRC_DIR:-/tmp/starfwi}"
INSTALL_PREFIX="${INSTALL_PREFIX:-/usr/local}"

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
echo "==> Building StarFWI from '${STARFWI_SRC_DIR}'..."

[[ -d "$STARFWI_SRC_DIR" ]] || { echo "ERROR: StarFWI source not found at '${STARFWI_SRC_DIR}'"; exit 1; }

# GCC 14 is used for StarFWI (matches Dockerfile)
export CC=/usr/bin/gcc14-gcc
export CXX=/usr/bin/gcc14-g++

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

echo ""
echo "Verify installation:"
echo "  starpu_machine_display   # should list CPU workers and MPI"
echo "  starfwi-fwi --help       # should print usage"
