set -euo pipefail

# CUDA env — auto-detect install, fallback to /usr/local/cuda symlink
if [ -z "${CUDA_HOME:-}" ]; then
    if command -v nvcc &>/dev/null; then
        CUDA_HOME=$(dirname "$(dirname "$(realpath "$(command -v nvcc)")")")
    elif [ -d /usr/local/cuda ]; then
        CUDA_HOME=$(realpath /usr/local/cuda)
    else
        echo "ERROR: CUDA not found. Install cuda-toolkit before running this script." >&2
        exit 1
    fi
fi
export CUDA_HOME
export PATH="$CUDA_HOME/bin:${PATH:-}"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

STARPU_REPO="${STARPU_REPO:-https://gitlab.inria.fr/starpu/starpu.git}"
STARPU_BRANCH="${STARPU_BRANCH:-spot-cloud-support}"
STARPU_SRC_DIR="/tmp/starpu"
STARFWI_REPO="${STARFWI_REPO:-https://github.com/vanderlei-filho/starfwi.git}"
STARFWI_BRANCH="${STARFWI_BRANCH:-main}"
STARFWI_SRC_DIR="/tmp/starfwi"
INSTALL_PREFIX="${INSTALL_PREFIX:-/usr/local}"
SHARED_DIR="${SHARED_DIR:-/shared}"

# ── 1. StarPU ─────────────────────────────────────────────────────────────────
rm -rf "$STARPU_SRC_DIR"
git clone --branch "$STARPU_BRANCH" --depth 1 "$STARPU_REPO" "$STARPU_SRC_DIR"

cd "$STARPU_SRC_DIR"
./autogen.sh
./configure \
    --prefix="$INSTALL_PREFIX" \
    --disable-opencl \
    --disable-build-examples \
    --disable-build-doc \
    --enable-mpi \
    --enable-mpi-ft \
    --enable-cuda \
    --with-cuda="$CUDA_HOME" \
    --with-fxt="$INSTALL_PREFIX" \
    --with-mpicc="$INSTALL_PREFIX/bin/mpicc" \
    --with-hwloc="$INSTALL_PREFIX"
make -j$(nproc)
sudo make install
sudo ldconfig
cd / && rm -rf "$STARPU_SRC_DIR"

# ── 2. StarFWI ────────────────────────────────────────────────────────────────
rm -rf "$STARFWI_SRC_DIR"
git clone --branch "$STARFWI_BRANCH" --depth 1 "$STARFWI_REPO" "$STARFWI_SRC_DIR"
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
