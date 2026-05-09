#!/usr/bin/env bash
# amazon_linux_setup_cpu_only.sh — Base AMI setup for StarFWI HPC cluster nodes (CPU only)
# Run once on a fresh Amazon Linux 2023 instance, then create an AMI from it.
set -euo pipefail

# ── 1. Environment variables ──────────────────────────────────────────────────
# /etc/profile.d: sourced for interactive/login shells (SSH, MPI)
sudo tee /etc/profile.d/hpc-env.sh << 'EOF'
export LANG=C.UTF-8
export LC_ALL=C.UTF-8
export PKG_CONFIG_PATH="/usr/local/lib/pkgconfig:/usr/local/lib64/pkgconfig:${PKG_CONFIG_PATH:-}"
export LD_LIBRARY_PATH="/usr/local/lib:/usr/local/lib64:${LD_LIBRARY_PATH:-}"
EOF
source /etc/profile.d/hpc-env.sh

# /etc/environment: sourced by PAM for all sessions including AWS SSM Run Command
# (does not support variable expansion — paths must be explicit)
sudo tee /etc/environment << 'EOF'
LANG=C.UTF-8
LC_ALL=C.UTF-8
PKG_CONFIG_PATH=/usr/local/lib/pkgconfig:/usr/local/lib64/pkgconfig
LD_LIBRARY_PATH=/usr/local/lib:/usr/local/lib64
EOF

# ── 2. Base packages ──────────────────────────────────────────────────────────
sudo dnf -y update && sudo dnf -y upgrade
sudo dnf -y group install "Development Tools"
sudo dnf -y install \
    wget \
    gcc gcc-c++ \
    gcc14 gcc14-c++ \
    clang clang-tools-extra \
    cmake \
    ninja-build \
    pkg-config automake autoconf libtool \
    hwloc hwloc-devel numactl-devel \
    kernel6.18-devel \
    kernel6.18-headers \
    dnf-plugins-core \
    openssh-server openssh-clients \
    nfs-utils

# ── 3. ldconfig path (persistent across reboots and new instances) ────────────
# Ensures /usr/local/lib and /usr/local/lib64 are always in the dynamic linker
# cache — required for MPI, StarPU, FxT, and HWLOC libraries built from source.
echo -e "/usr/local/lib\n/usr/local/lib64" | sudo tee /etc/ld.so.conf.d/local-libs.conf
sudo ldconfig

# ── 4. HWLOC 2.12.0 from source ──────────────────────────────────────────────
sudo dnf -y remove hwloc hwloc-devel
cd /tmp
wget https://download.open-mpi.org/release/hwloc/v2.12/hwloc-2.12.0.tar.bz2
tar -xf hwloc-2.12.0.tar.bz2
cd hwloc-2.12.0
./configure --prefix=/usr/local
make -j$(nproc)
sudo make install
sudo ldconfig
cd / && rm -rf /tmp/hwloc-*

# ── 5. OpenMPI 5.0.7 ─────────────────────────────────────────────────────────
cd /tmp
wget https://download.open-mpi.org/release/open-mpi/v5.0/openmpi-5.0.7.tar.bz2
tar xf openmpi-5.0.7.tar.bz2
cd openmpi-5.0.7
./configure \
    --prefix=/usr/local \
    --disable-mpi-fortran \
    --enable-mca-no-build=btl-uct \
    --enable-mpi-thread-multiple \
    --enable-shared \
    --enable-static
make -j$(nproc)
sudo make install
sudo ldconfig
cd / && rm -rf /tmp/openmpi-*

# ── 6. FxT ───────────────────────────────────────────────────────────────────
cd /tmp
wget https://salsa.debian.org/debian/fxt/-/archive/master/fxt-master.tar.gz
tar -xzf fxt-master.tar.gz
cd fxt-master
./configure --prefix=/usr/local
make -j$(nproc)
sudo make install
sudo ldconfig
cd / && rm -rf /tmp/fxt-*

# ── 7. SSH client config (no StrictHostKeyChecking for MPI) ──────────────────
sudo tee /etc/ssh/ssh_config.d/99-mpi.conf << 'EOF'
Host *
  StrictHostKeyChecking no
  UserKnownHostsFile=/dev/null
EOF

# ── 8. mpiuser ────────────────────────────────────────────────────────────────
sudo useradd -m -s /bin/bash mpiuser
echo "mpiuser ALL=(ALL) NOPASSWD:ALL" | sudo tee -a /etc/sudoers

# ── 9. SSH keys for mpiuser ───────────────────────────────────────────────────
sudo ssh-keygen -A
sudo mkdir -p /home/mpiuser/.ssh
sudo ssh-keygen -t rsa -N "" -f /home/mpiuser/.ssh/id_rsa
sudo cp /home/mpiuser/.ssh/id_rsa.pub /home/mpiuser/.ssh/authorized_keys
sudo chmod 700 /home/mpiuser/.ssh
sudo chmod 600 /home/mpiuser/.ssh/authorized_keys /home/mpiuser/.ssh/id_rsa
sudo chown -R mpiuser:mpiuser /home/mpiuser/.ssh

# ── 10. Shared directory ──────────────────────────────────────────────────────
sudo mkdir -p /shared
sudo chown -R mpiuser:mpiuser /shared

echo ""
echo "Setup complete. Verify before creating the AMI:"
echo "  mpicc --version              # should show OpenMPI 5.0.7"
echo "  hwloc-info                   # should show HWLOC 2.12.0"
echo "  ls /usr/local/lib/libfxt*    # should exist"
echo "  id mpiuser                   # should exist"
echo "  ls -la /shared               # should be mpiuser-owned"
echo "  cat /etc/ld.so.conf.d/local-libs.conf  # should list /usr/local/lib paths"
echo "  cat /etc/environment         # should have PKG_CONFIG_PATH and LD_LIBRARY_PATH"
echo ""
read -rp "Verification done? Proceed with AMI cleanup and shutdown? [y/N] " confirm
if [[ "${confirm,,}" != "y" ]]; then
    echo "Cleanup skipped. Re-run this script or run the cleanup block manually when ready."
    exit 0
fi

# ── 11. AMI pre-capture cleanup ───────────────────────────────────────────────
# Remove SSH host keys — cloud-init regenerates unique ones per instance on boot.
# (mpiuser keys in /home/mpiuser/.ssh/ are kept intentionally: all nodes from
# this AMI share the same key pair to enable passwordless MPI SSH between them.)
sudo rm -f /etc/ssh/ssh_host_*

# Reset cloud-init so initialization runs on new instances launched from this AMI
sudo cloud-init clean --logs

# Clear shell history to avoid baking credentials into the image
history -c
sudo rm -f /root/.bash_history ~/.bash_history

# Remove leftover temp files
sudo rm -rf /tmp/* /var/tmp/*

echo "Cleanup done. Shutting down — create the AMI once the instance shows as Stopped."
sudo shutdown -h now
