# ── 4. CUDA toolkit 12.8 ─────────────────────────────────────────────────────
sudo dnf config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/amzn2023/x86_64/cuda-amzn2023.repo
sudo dnf clean all
sudo dnf -y install cuda-toolkit-12-8

sudo tee /etc/profile.d/cuda.sh << 'EOF'
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=$CUDA_HOME/bin:${PATH:-}
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}
EOF
source /etc/profile.d/cuda.sh

# ── 5. Reboot ─────────────────────────────────────────────────────────
sudo reboot
