FROM amazonlinux:2023.9.20251027.0

ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

ENV PKG_CONFIG_PATH="/usr/local/lib/pkgconfig:/usr/local/lib64/pkgconfig:${PKG_CONFIG_PATH}"
ENV LD_LIBRARY_PATH="/usr/local/lib:/usr/local/lib64:${LD_LIBRARY_PATH}"

# GPU_SUPPORT=true: installs CUDA toolkit and enables CUDA in StarPU
# IMPORTANT: CUDA toolkit is installed AFTER OpenMPI so that OpenMPI's configure
# never sees nvcc/CUDA headers, keeping libmpi.so free of libnvidia-ml dependencies.
# If OpenMPI is built with CUDA present, libmpi.so gains a hard dependency on
# libnvidia-ml.so.1 (a driver-only library not available at build time), which then
# gets loaded by hwloc-calc during StarPU's configure and corrupts its stdout output.
ARG GPU_SUPPORT=false

# Install compilers and base dependencies
RUN dnf -y update \
    && dnf -y upgrade \
    && dnf -y group install "Development Tools" \
    && dnf -y install \
        wget \
        gcc gcc-c++ \
        gcc14 gcc14-c++ \
        clang clang-tools-extra \
        cmake \
        ninja-build \
        pkg-config automake autoconf libtool \
        hwloc hwloc-devel numactl-devel \
        kernel6.12-devel \
        kernel6.12-headers \
        dnf-plugins-core

# Re-install HWLOC — needed for StarPU CPU topology detection.
# Note: HWLOC is intentionally built WITHOUT --enable-cuda even in GPU mode.
# Enabling it causes hwloc-calc to link against libnvidia-ml (a driver-only lib
# not present at build time), which breaks StarPU's configure script via corrupted
# stdout output. StarPU manages CUDA devices directly and does not need HWLOC
# to be CUDA-aware.
RUN cd /tmp \
    && dnf -y remove hwloc hwloc-devel \
    && wget https://download.open-mpi.org/release/hwloc/v2.12/hwloc-2.12.0.tar.bz2 \
    && tar -xf hwloc-2.12.0.tar.bz2 \
    && cd hwloc-2.12.0 \
    && ./configure \
        --prefix=/usr/local \
    && make -j$(nproc) \
    && make install \
    && ldconfig \
    && cd / \
    && rm -rf /tmp/hwloc-*

# Install OpenMPI
RUN cd /tmp \
    && wget https://download.open-mpi.org/release/open-mpi/v5.0/openmpi-5.0.7.tar.bz2 \
    && tar xf openmpi-5.0.7.tar.bz2 \
    && cd openmpi-5.0.7 \
    && ./configure \
        --prefix=/usr/local \
        --disable-mpi-fortran \
        --enable-mca-no-build=btl-uct \
        --enable-mpi-thread-multiple \
        --enable-shared \
        --enable-static \
    && make -j$(nproc) \
    && make install \
    && ldconfig \
    && cd / \
    && rm -rf /tmp/openmpi-*

# Install FxT for StarPU performance analysis
RUN cd /tmp \
    && wget https://salsa.debian.org/debian/fxt/-/archive/master/fxt-master.tar.gz \
    && tar -xzf fxt-master.tar.gz \
    && cd fxt-master \
    && ./configure \
        --prefix=/usr/local \
    && make -j$(nproc) \
    && make install \
    && ldconfig \
    && cd / \
    && rm -rf /tmp/fxt-*

# Install CUDA toolkit 12.6 (nvcc + headers + libs; runtime provided by host driver via CDI).
# Placed AFTER OpenMPI and FxT so their configure scripts never see CUDA headers/nvcc,
# keeping libmpi.so free of libnvidia-ml dependencies.
RUN if [ "${GPU_SUPPORT}" = "true" ]; then \
        dnf config-manager --add-repo \
            https://developer.download.nvidia.com/compute/cuda/repos/amzn2023/x86_64/cuda-amzn2023.repo \
        && dnf -y install cuda-toolkit-12-6 \
        && ldconfig; \
    fi

# Make CUDA driver stubs available for StarPU's build-time linking.
# libnvidia-ml.so.1 and libcuda.so.1 are driver libs supplied at runtime via CDI;
# the stubs here satisfy the dynamic linker during the container build only.
# Placed after HWLOC and OpenMPI so neither links against libnvidia-ml.
RUN if [ "${GPU_SUPPORT}" = "true" ]; then \
        ln -sf /usr/local/cuda/lib64/stubs/libnvidia-ml.so \
               /usr/local/cuda/lib64/stubs/libnvidia-ml.so.1 \
        && ln -sf /usr/local/cuda/lib64/stubs/libcuda.so \
                  /usr/local/cuda/lib64/stubs/libcuda.so.1 \
        && echo "/usr/local/cuda/lib64/stubs" > /etc/ld.so.conf.d/cuda-stubs.conf \
        && ldconfig; \
    fi

# Install StarPU from local fork (injected via --build-context starpu=~/Code/starpu)
COPY --from=starpu . /tmp/starpu/
RUN cd /tmp/starpu \
    && ./autogen.sh \
    && CUDA_FLAGS="" \
    && if [ "${GPU_SUPPORT}" = "true" ]; then CUDA_FLAGS="--enable-cuda --with-cuda=/usr/local/cuda"; fi \
    && ./configure \
        --prefix=/usr/local \
        --disable-opencl \
        --disable-build-examples \
        --disable-build-doc \
        --enable-mpi \
        --enable-mpi-ft \
        ${CUDA_FLAGS} \
        --with-fxt=/usr/local \
        --with-mpicc=/usr/local/bin/mpicc \
        --with-hwloc=/usr/local \
    && make -j$(nproc) \
    && make install \
    && ldconfig \
    && rm -rf /tmp/starpu

# Remove CUDA stubs from the runtime ld cache.
# The stubs were needed only so the dynamic linker could satisfy libnvidia-ml/libcuda
# references while building StarPU. At runtime the real driver libraries are injected
# into the container by CDI; if the stubs conf remains, the stub takes precedence and
# StarPU's CUDA init fails to talk to the real driver (0 CUDA workers detected).
RUN if [ "${GPU_SUPPORT}" = "true" ]; then \
        rm -f /etc/ld.so.conf.d/cuda-stubs.conf \
        && ldconfig; \
    fi

# Setup passwordless SSH for MPI communications
RUN cat > /etc/ssh/ssh_config <<EOF
Host *
  StrictHostKeyChecking no
  UserKnownHostsFile=/dev/null
EOF

# Setup non-root user for MPI and workdir at /shared
RUN useradd -m -s /bin/bash mpiuser \
    && echo "mpiuser ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

RUN dnf -y install openssh-server openssh-clients

# Generate SSH keys for mpiuser and configure SSH server
RUN ssh-keygen -A \
    && mkdir -p /home/mpiuser/.ssh \
    && ssh-keygen -t rsa -N "" -f /home/mpiuser/.ssh/id_rsa \
    && cat /home/mpiuser/.ssh/id_rsa.pub > /home/mpiuser/.ssh/authorized_keys \
    && chmod 700 /home/mpiuser/.ssh \
    && chmod 600 /home/mpiuser/.ssh/authorized_keys \
    && chmod 600 /home/mpiuser/.ssh/id_rsa \
    && chown -R mpiuser:mpiuser /home/mpiuser/.ssh

# Configure shared directory and permissions
RUN mkdir -p /shared \
    && chown -R mpiuser:mpiuser /shared
WORKDIR /shared

# Copy StarFWI source code and build configuration
COPY CMakeLists.txt ./
COPY src/ ./src/
COPY include/ ./include/

# Set GCC 14 as default compiler for StarFWI build
ENV CC=/usr/bin/gcc14-gcc
ENV CXX=/usr/bin/gcc14-g++

# Add CUDA to PATH for nvcc (no-op if CUDA not installed)
ENV PATH="/usr/local/cuda/bin:${PATH}"

# Configure and build StarFWI
# Clean any existing build directory to avoid cache issues
RUN rm -rf build \
    && cmake -B build -S . \
        -DCMAKE_BUILD_TYPE=Debug \
        -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
    && cmake --build build \
    && cp build/starfwi-modeling /usr/local/bin/starfwi-modeling \
    && cp build/starfwi-fwi /usr/local/bin/starfwi-fwi \
    && chmod +x /usr/local/bin/starfwi-modeling /usr/local/bin/starfwi-fwi

# Note: Container runs as root to allow sshd to start
# MPI jobs will run as mpiuser via 'podman exec -u mpiuser'
