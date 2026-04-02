// forward_propagation.cu — CUDA implementation of the forward propagation codelet.
//
// Design notes:
//   - VARIABLE buffers (shot, task_config) stay in host memory: the codelet struct
//     sets specific_nodes=1 with STARPU_SPECIFIC_NODE_CPU for those slots, so
//     StarPU never transfers them to GPU. STARPU_VARIABLE_GET_PTR returns a valid
//     host pointer in both CPU and CUDA codelet functions.
//   - VECTOR buffers (velocity, recv_x/y/z) ARE auto-transferred to the GPU;
//     STARPU_VECTOR_GET_PTR returns a device pointer in CUDA context.
//   - The three wavefield arrays (p_cur, p_old, p_new) and vel2 are allocated
//     inside this function and freed before return — they are not StarPU handles.
//   - STARPU_CUDA_ASYNC is set in the codelet struct, so StarPU records a CUDA
//     event on the stream after this function returns to detect task completion.
//     All async work (kernels, memcpies) must be submitted to that stream before
//     returning. Per-step cudaStreamSynchronize is used only when host-side I/O
//     (disk writes) requires it.
//   - No C++23 headers: use printf/fprintf instead of std::println.

#include "codelets/forward_propagation.hpp"
#include <cuda_runtime.h>
#include <starpu.h>
#include <starpu_cuda.h>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sys/stat.h>
#include <sys/sysinfo.h>

namespace starfwi {

// ─────────────────────────────────────────────────────────────────────────────
// Device kernels
// ─────────────────────────────────────────────────────────────────────────────

// Square the velocity model: vel2[i] = vel[i]²
__global__ void vel_square_kernel(const float * __restrict__ vel,
                                   float       * __restrict__ vel2,
                                   int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) vel2[i] = vel[i] * vel[i];
}

// Add source amplitude to p_cur at the source grid point.
// Single-thread kernel launched with <<<1,1>>> to match the CPU "hard source"
// injection that modifies p_cur before the Laplacian is computed.
__global__ void apply_source_kernel(float * __restrict__ p_cur,
                                     int src_ix, int src_iz,
                                     int nx, float amp) {
    p_cur[src_ix + nx * src_iz] += amp;
}

// 2D finite-difference stencil for the acoustic wave equation (ny == 1 path).
//
// Grid index: idx = ix + nx * iz  (row-major, X fast)
//
// Update rule (second-order in space and time):
//   p_new = 2*p_cur - p_old + vel2 * dt² * (∂²p/∂x² + ∂²p/∂z²)
//
// Boundary threads (ix==0, ix==nx-1, iz==0, iz==nz-1) apply Dirichlet
// (zero-pressure) conditions by writing 0 and returning early — identical
// to apply_boundary_conditions() in the CPU solver.
__global__ void fd2d_step_kernel(const float * __restrict__ p_cur,
                                  const float * __restrict__ p_old,
                                  float       * __restrict__ p_new,
                                  const float * __restrict__ vel2,
                                  int nx, int nz,
                                  float dx2_inv, float dz2_inv, float dt2) {
    int ix = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int iz = (int)(blockIdx.y * blockDim.y + threadIdx.y);

    if (ix >= nx || iz >= nz) return;

    int idx = ix + nx * iz;

    if (ix == 0 || ix == nx - 1 || iz == 0 || iz == nz - 1) {
        p_new[idx] = 0.0f;
        return;
    }

    float d2p_dx2 = (p_cur[idx - 1] - 2.0f * p_cur[idx] + p_cur[idx + 1]) * dx2_inv;
    float d2p_dz2 = (p_cur[idx - nx] - 2.0f * p_cur[idx] + p_cur[idx + nx]) * dz2_inv;

    p_new[idx] = 2.0f * p_cur[idx] - p_old[idx]
                 + vel2[idx] * dt2 * (d2p_dx2 + d2p_dz2);
}

// Record receiver values using 2D bilinear interpolation.
// Matches ReceiverRecorder::interpolate_at_point for the ny==1 case.
// Seismogram layout: seismogram[receiver * nt + t]
__global__ void record_receivers_kernel(const float * __restrict__ p,
                                         float       * __restrict__ seismogram,
                                         const float * __restrict__ recv_x,
                                         const float * __restrict__ recv_z,
                                         int n_receivers, int nt, int t,
                                         int nx, int nz, float dx, float dz) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= n_receivers) return;

    float fx = recv_x[r] / dx;
    float fz = recv_z[r] / dz;

    int ix0 = (int)floorf(fx);
    int iz0 = (int)floorf(fz);

    float wx = fx - (float)ix0;
    float wz = fz - (float)iz0;

    ix0 = max(0, min(ix0, nx - 1));
    iz0 = max(0, min(iz0, nz - 1));
    int ix1 = min(ix0 + 1, nx - 1);
    int iz1 = min(iz0 + 1, nz - 1);

    float v00 = p[ix0 + nx * iz0];
    float v10 = p[ix1 + nx * iz0];
    float v01 = p[ix0 + nx * iz1];
    float v11 = p[ix1 + nx * iz1];

    float v0 = v00 * (1.0f - wx) + v10 * wx;
    float v1 = v01 * (1.0f - wx) + v11 * wx;

    seismogram[r * nt + t] = v0 * (1.0f - wz) + v1 * wz;
}

// ─────────────────────────────────────────────────────────────────────────────
// CUDA codelet
// ─────────────────────────────────────────────────────────────────────────────

void forward_propagation_cuda(void *buffers[], void *cl_arg) {
    // ── StarPU buffer extraction ──────────────────────────────────────────────
    // VECTOR buffers: auto-transferred to GPU → device pointers
    const float *d_velocity = (const float *)STARPU_VECTOR_GET_PTR(buffers[0]);
    // VARIABLE buffers: never transferred → host pointers
    ShotData   *shot        = (ShotData   *)STARPU_VARIABLE_GET_PTR(buffers[1]);
    TaskConfig *task_config = (TaskConfig *)STARPU_VARIABLE_GET_PTR(buffers[2]);
    // Receiver coordinates: device pointers (VECTOR, auto-transferred)
    const float *d_recv_x = (const float *)STARPU_VECTOR_GET_PTR(buffers[3]);
    // buffers[4] is recv_y — not used for 2D recording
    const float *d_recv_z = (const float *)STARPU_VECTOR_GET_PTR(buffers[5]);

    CodeletArg *arg      = (CodeletArg *)cl_arg;
    const char *hostname = arg ? arg->hostname : "unknown";
    bool        verbose  = arg ? arg->verbose  : false;

    cudaStream_t stream = starpu_cuda_get_local_stream();

    // ── Grid parameters ───────────────────────────────────────────────────────
    const int   nx        = (int)task_config->nx;
    const int   nz        = (int)task_config->nz;
    const int   nt        = (int)task_config->nt;
    const float dx        = task_config->dx;
    const float dz        = task_config->dz;
    const float dt        = task_config->dt;
    const int   n_recv    = (int)task_config->n_receivers;
    const int   grid_size = nx * nz;  // ny == 1 for 2D

    const float dx2_inv = 1.0f / (dx * dx);
    const float dz2_inv = 1.0f / (dz * dz);
    const float dt2     = dt * dt;

    if (verbose) {
        printf("[starfwi][%s][forward_propagation_cuda] Shot %zu: "
               "nx=%d nz=%d nt=%d n_recv=%d\n",
               hostname, shot->shot_id, nx, nz, nt, n_recv);
    }

    // ── Source grid index ─────────────────────────────────────────────────────
    int src_ix = (int)(shot->source_x / dx);
    int src_iz = (int)(shot->source_z / dz);
    src_ix = max(0, min(src_ix, nx - 1));
    src_iz = max(0, min(src_iz, nz - 1));

    // ── Kernel launch config ──────────────────────────────────────────────────
    dim3 block2d(32, 8);
    dim3 grid2d((nx + 31) / 32, (nz + 7) / 8);
    int  vel_blocks  = (grid_size + 255) / 256;
    dim3 recv_block(128);
    dim3 recv_grid((n_recv + 127) / 128);

    // ── Allocate device wavefields ────────────────────────────────────────────
    float *d_p_cur, *d_p_old, *d_p_new, *d_vel2;
    cudaMalloc(&d_p_cur, (size_t)grid_size * sizeof(float));
    cudaMalloc(&d_p_old, (size_t)grid_size * sizeof(float));
    cudaMalloc(&d_p_new, (size_t)grid_size * sizeof(float));
    cudaMalloc(&d_vel2,  (size_t)grid_size * sizeof(float));
    cudaMemsetAsync(d_p_cur, 0, (size_t)grid_size * sizeof(float), stream);
    cudaMemsetAsync(d_p_old, 0, (size_t)grid_size * sizeof(float), stream);
    cudaMemsetAsync(d_p_new, 0, (size_t)grid_size * sizeof(float), stream);

    // Pre-compute squared velocities
    vel_square_kernel<<<vel_blocks, 256, 0, stream>>>(d_velocity, d_vel2, grid_size);

    // ── Seismogram buffer ─────────────────────────────────────────────────────
    float *d_seismogram = nullptr;
    if (n_recv > 0) {
        cudaMalloc(&d_seismogram, (size_t)n_recv * nt * sizeof(float));
        cudaMemsetAsync(d_seismogram, 0, (size_t)n_recv * nt * sizeof(float), stream);
        shot->synthetic_data.resize((size_t)n_recv * nt);
    }

    // ── Snapshot storage — three-tier auto-detection ──────────────────────────
    // NONE (wavefield_storage==2): no snapshots needed (modeling path).
    // Otherwise, select the best available strategy at runtime:
    //   Tier 1 — GPU VRAM:  d_all_snapshots, device-to-device per step,
    //             single bulk D2H after the loop. Fastest option.
    //   Tier 2 — Host RAM:  shot->pressure_snapshots pre-allocated, per-step
    //             D2H with stream sync. Avoids disk I/O.
    //   Tier 3 — Disk:      pinned staging buffer (1 snapshot), per-step D2H
    //             + file write. Used only when both GPU and host RAM fall short.
    float       *d_all_snapshots = nullptr;  // Tier 1: full GPU snapshot buffer
    float       *h_staging       = nullptr;  // Tier 3: per-step pinned staging
    std::ofstream wf_out;
    int actual_storage = -1; // -1 = NONE

    if (task_config->wavefield_storage != 2) {
        const size_t snap_bytes = (size_t)nt * grid_size * sizeof(float);

        // Tier 1: GPU VRAM
        size_t free_gpu, total_gpu;
        cudaMemGetInfo(&free_gpu, &total_gpu);
        if (snap_bytes < free_gpu * 8 / 10) {
            cudaError_t err = cudaMalloc(&d_all_snapshots, snap_bytes);
            if (err != cudaSuccess) {
                cudaGetLastError();
                d_all_snapshots = nullptr;
            }
        }
        if (d_all_snapshots) {
            actual_storage = 0; // MEMORY
            if (verbose)
                printf("[starfwi][%s][forward_propagation_cuda] Shot %zu: "
                       "snapshots in GPU VRAM (%zu MB)\n",
                       hostname, shot->shot_id, snap_bytes >> 20);
        }

        // Tier 2: Host RAM
        if (!d_all_snapshots) {
            struct sysinfo si{};
            sysinfo(&si);
            size_t free_ram = (size_t)si.freeram * si.mem_unit;
            if (snap_bytes < free_ram * 7 / 10) {
                shot->pressure_snapshots.resize((size_t)nt * grid_size);
                actual_storage = 0; // MEMORY
                if (verbose)
                    printf("[starfwi][%s][forward_propagation_cuda] Shot %zu: "
                           "GPU VRAM insufficient, snapshots in host RAM "
                           "(%zu MB needed, %zu MB free)\n",
                           hostname, shot->shot_id,
                           snap_bytes >> 20, (free_ram * 7 / 10) >> 20);
            }
        }

        // Tier 3: Disk
        if (!d_all_snapshots && actual_storage == -1) {
            actual_storage = 1; // DISK
            cudaMallocHost(&h_staging, (size_t)grid_size * sizeof(float));
            mkdir(task_config->wavefield_dir, 0755);
            char wf_path[512];
            snprintf(wf_path, sizeof(wf_path), "%s/fwd_shot_%zu.bin",
                     task_config->wavefield_dir, shot->shot_id);
            wf_out.open(wf_path, std::ios::binary | std::ios::trunc);
            fprintf(stderr,
                    "[starfwi][%s][forward_propagation_cuda] Shot %zu: "
                    "GPU VRAM and host RAM insufficient — writing snapshots "
                    "to disk\n", hostname, shot->shot_id);
            if (!wf_out)
                fprintf(stderr,
                        "[starfwi][%s][forward_propagation_cuda] WARNING: "
                        "cannot open wavefield file '%s'\n",
                        hostname, wf_path);
        }
    }
    shot->wavefield_storage_actual = actual_storage;

    const bool use_memory = (actual_storage == 0);
    const bool use_disk   = (actual_storage == 1);

    // ── Time loop ─────────────────────────────────────────────────────────────
    float *d_cur = d_p_cur, *d_old = d_p_old, *d_new = d_p_new;

    for (int t = 0; t < nt; ++t) {
        // 1. Inject source into d_cur (hard source, matches CPU path)
        float amp = (t < (int)shot->source_wavelet.size())
                    ? shot->source_wavelet[t] : 0.0f;
        apply_source_kernel<<<1, 1, 0, stream>>>(d_cur, src_ix, src_iz, nx, amp);

        // 2. FD stencil: reads d_cur (with source), d_old; writes d_new
        fd2d_step_kernel<<<grid2d, block2d, 0, stream>>>(
            d_cur, d_old, d_new, d_vel2, nx, nz, dx2_inv, dz2_inv, dt2);

        // 3. Record receivers from d_new (field after this step — matches CPU
        //    path where recording happens after swap_time_levels)
        if (n_recv > 0 && d_seismogram) {
            record_receivers_kernel<<<recv_grid, recv_block, 0, stream>>>(
                d_new, d_seismogram, d_recv_x, d_recv_z,
                n_recv, nt, t, nx, nz, dx, dz);
        }

        // 4. Save snapshot for adjoint propagation
        if (d_all_snapshots) {
            // Tier 1: device-to-device, fast, no sync needed
            cudaMemcpyAsync(d_all_snapshots + (size_t)t * grid_size,
                            d_new, (size_t)grid_size * sizeof(float),
                            cudaMemcpyDeviceToDevice, stream);
        } else if (use_memory) {
            // Tier 2: per-step D2H into pre-allocated host buffer; must sync
            // before the CPU writes to the next slot (unpinned memory means
            // cudaMemcpyAsync is effectively synchronous, but we sync explicitly
            // to be safe).
            cudaMemcpyAsync(
                shot->pressure_snapshots.data() + (size_t)t * grid_size,
                d_new, (size_t)grid_size * sizeof(float),
                cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream);
        } else if (use_disk && h_staging) {
            // Tier 3: per-step D2H into pinned staging buffer, then file write
            cudaMemcpyAsync(h_staging, d_new,
                            (size_t)grid_size * sizeof(float),
                            cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream);
            if (wf_out)
                wf_out.write(reinterpret_cast<const char *>(h_staging),
                             (size_t)grid_size * sizeof(float));
        }

        // 5. Rotate time levels (pointer swap — no data copy)
        float *tmp = d_old;
        d_old = d_cur;
        d_cur = d_new;
        d_new = tmp;
    }

    // ── Post-loop: bulk D2H transfers ─────────────────────────────────────────

    // Seismogram: one bulk copy after the time loop
    if (n_recv > 0 && d_seismogram) {
        cudaMemcpyAsync(shot->synthetic_data.data(), d_seismogram,
                        (size_t)n_recv * nt * sizeof(float),
                        cudaMemcpyDeviceToHost, stream);
    }

    // Tier 1 only: bulk D2H after the loop (snapshots were kept on GPU)
    if (d_all_snapshots) {
        shot->pressure_snapshots.resize((size_t)nt * grid_size);
        cudaMemcpyAsync(shot->pressure_snapshots.data(), d_all_snapshots,
                        (size_t)nt * grid_size * sizeof(float),
                        cudaMemcpyDeviceToHost, stream);
    }
    // Tier 2: already copied per-step into shot->pressure_snapshots — nothing to do.

    // Note: no cudaStreamSynchronize here. STARPU_CUDA_ASYNC causes StarPU to
    // insert a CUDA event on the stream after this function returns, which fires
    // once all submitted work (including the D2H copies above) completes.
    // Downstream tasks (backward propagation) that depend on the shot handle
    // will wait for that event before starting.

    // ── Cleanup ───────────────────────────────────────────────────────────────
    cudaFree(d_p_cur);
    cudaFree(d_p_old);
    cudaFree(d_p_new);
    cudaFree(d_vel2);
    if (d_seismogram)    cudaFree(d_seismogram);
    if (d_all_snapshots) cudaFree(d_all_snapshots);
    if (h_staging)       cudaFreeHost(h_staging);
    if (wf_out.is_open()) wf_out.close();

    if (verbose) {
        printf("[starfwi][%s][forward_propagation_cuda] Shot %zu complete\n",
               hostname, shot->shot_id);
    }
}

} // namespace starfwi
