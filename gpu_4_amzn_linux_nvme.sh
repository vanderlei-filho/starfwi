# ── 9. NVMe instance store ────────────────────────────────────────────────────
# g7e instances have an NVMe instance store SSD at /dev/nvme1n1.
# It is ephemeral: data is lost on stop/termination. We use it for fast
# checkpoint flush benchmarks only.
#
# A systemd service handles format+mount on every boot (since the device is
# fresh/unformatted each time a new instance is launched from this AMI).
sudo mkdir -p /mnt/nvme

sudo tee /etc/systemd/system/nvme-mount.service << 'EOF'
[Unit]
Description=Format and mount NVMe instance store at /mnt/nvme
DefaultDependencies=no
Before=local-fs.target
ConditionPathExists=/dev/nvme1n1

[Service]
Type=oneshot
RemainAfterExit=yes
ExecStart=/bin/bash -c 'mkfs.ext4 -F /dev/nvme1n1'
ExecStart=/bin/bash -c 'mount /dev/nvme1n1 /mnt/nvme'
ExecStart=/bin/bash -c 'chown -R mpiuser:mpiuser /mnt/nvme'

[Install]
WantedBy=local-fs.target
EOF
sudo systemctl enable nvme-mount.service

# Format and mount for this instance right now (service will re-run on next boot)
sudo mkfs.ext4 -F /dev/nvme1n1
sudo mount /dev/nvme1n1 /mnt/nvme

# mpiuser owns the NVMe mount too
sudo chown -R mpiuser:mpiuser /mnt/nvme
