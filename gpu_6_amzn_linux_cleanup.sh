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
