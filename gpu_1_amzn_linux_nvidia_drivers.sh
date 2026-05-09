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
echo -e "/usr/local/lib\n/usr/local/lib64" | sudo tee /etc/ld.so.conf.d/local-libs.conf
sudo ldconfig

# ── 4. NVIDIA driver ─────────────────────────────────────────────────────────
# g7e GPU: NVIDIA RTX PRO 6000 Blackwell (requires open kernel modules).
# The base Amazon Linux 2023 AMI does not include the NVIDIA driver.
# open-dkms installs the driver via DKMS, which requires a reboot to load
# the kernel module. Run gpu_2_amzn_linux_setup.sh AFTER the reboot.
cd /tmp
wget https://us.download.nvidia.com/tesla/595.71.05/nvidia-driver-local-repo-amzn2023-595.71.05-1.0-1.x86_64.rpm -O nvidia-driver.rpm
sudo rpm --install nvidia-driver.rpm
sudo dnf module install -y nvidia-driver:open-dkms

# ── 5. Reboot ─────────────────────────────────────────────────────────
sudo reboot
