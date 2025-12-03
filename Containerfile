FROM amazonlinux:2023.9.20251027.0

ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

ENV PKG_CONFIG_PATH="/usr/local/lib/pkgconfig:/usr/local/lib64/pkgconfig:${PKG_CONFIG_PATH}"
ENV LD_LIBRARY_PATH="/usr/local/lib:/usr/local/lib64:${LD_LIBRARY_PATH}"

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
        kernel6.12-headers

# Re-install HWLOC
# Why? This is needed for correct CUDA support in StarPU
RUN cd /tmp \
    && dnf -y remove hwloc hwloc-devel \
    && wget https://download.open-mpi.org/release/hwloc/v2.12/hwloc-2.12.0.tar.bz2 \
    && tar -xf hwloc-2.12.0.tar.bz2 \
    && cd hwloc-2.12.0 \
    && ./configure \
        --prefix=/usr/local \
        #--enable-cuda \
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

# Install StarPU with fully distributed MPI support
RUN cd /tmp \
    && wget https://files.inria.fr/starpu/starpu-1.4.7/starpu-1.4.7.tar.gz \
    && tar xzf starpu-1.4.7.tar.gz \
    && cd starpu-1.4.7 \
    && ./configure \
        --prefix=/usr/local \
        --disable-opencl \
        --disable-build-examples \
        --disable-build-doc \
        --enable-mpi \
        --with-fxt=/usr/local \
        --with-mpicc=/usr/local/bin/mpicc \
        --with-hwloc=/usr/local \
    && make -j$(nproc) \
    && make install \
    && ldconfig \
    && cd / \
    && rm -rf /tmp/starpu-*

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

# Configure and build StarFWI
# Clean any existing build directory to avoid cache issues
RUN rm -rf build \
    && cmake -B build -S . \
        -DCMAKE_BUILD_TYPE=Debug \
        -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
    && cmake --build build \
    && cp build/starfwi /usr/local/bin/starfwi \
    && chmod +x /usr/local/bin/starfwi

# Note: Container runs as root to allow sshd to start
# MPI jobs will run as mpiuser via 'podman exec -u mpiuser'
