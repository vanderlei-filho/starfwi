# ── 1. HWLOC 2.12.0 from source ──────────────────────────────────────────────
sudo dnf -y remove hwloc hwloc-devel
cd /tmp
wget https://download.open-mpi.org/release/hwloc/v2.12/hwloc-2.12.0.tar.bz2
tar -xf hwloc-2.12.0.tar.bz2
cd hwloc-2.12.0
./configure --prefix=/usr/local --with-cuda=/usr/local/cuda-12.8
make -j$(nproc)
sudo make install
sudo ldconfig
cd / && rm -rf /tmp/hwloc-*

sudo tee /etc/profile.d/pkgconfig.sh << 'EOF'
#!/bin/bash
export PKG_CONFIG_PATH="/usr/local/lib/pkgconfig:$PKG_CONFIG_PATH"
EOF

sudo chmod +x /etc/profile.d/pkgconfig.sh
sudo reboot

# ── 2. OpenMPI 5.0.7 ─────────────────────────────────────────────────────────
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

# ── 3. FxT ───────────────────────────────────────────────────────────────────
cd /tmp
wget https://salsa.debian.org/debian/fxt/-/archive/master/fxt-master.tar.gz
tar -xzf fxt-master.tar.gz
cd fxt-master
./configure --prefix=/usr/local
make -j$(nproc)
sudo make install
sudo ldconfig
cd / && rm -rf /tmp/fxt-*

# ── 5. SSH client config (no StrictHostKeyChecking for MPI) ─────────────────
sudo tee /etc/ssh/ssh_config.d/99-mpi.conf << 'EOF'
Host *
  StrictHostKeyChecking no
  UserKnownHostsFile=/dev/null
EOF

# ── 6. mpiuser ───────────────────────────────────────────────────────────────
sudo useradd -m -s /bin/bash mpiuser
echo "mpiuser ALL=(ALL) NOPASSWD:ALL" | sudo tee -a /etc/sudoers

# ── 7. SSH keys for mpiuser ──────────────────────────────────────────────────
sudo ssh-keygen -A
sudo mkdir -p /home/mpiuser/.ssh
sudo ssh-keygen -t rsa -N "" -f /home/mpiuser/.ssh/id_rsa
sudo cp /home/mpiuser/.ssh/id_rsa.pub /home/mpiuser/.ssh/authorized_keys
sudo chmod 700 /home/mpiuser/.ssh
sudo chmod 600 /home/mpiuser/.ssh/authorized_keys /home/mpiuser/.ssh/id_rsa
sudo chown -R mpiuser:mpiuser /home/mpiuser/.ssh

# ── 8. Shared directory ──────────────────────────────────────────────────────
sudo mkdir -p /shared
sudo chown -R mpiuser:mpiuser /shared
sudo chmod 777 /shared
# Allow ec2-user (used by hpcac-toolkit for uploads) to write to /shared
sudo usermod -aG mpiuser ec2-user

