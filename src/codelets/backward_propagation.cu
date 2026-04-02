// backward_propagation.cu — CUDA implementation of the adjoint (backward)
// propagation codelet.
//
// Design notes (same conventions as forward_propagation.cu):
//   - VARIABLE buffers (shot, task_config) stay in host memory: the codelet struct
//     sets specific_nodes=1 with STARPU_SPECIFIC_NODE_CPU for those slots.
//   - VECTOR buffers (velocity, recv_x/y/z) are auto-transferred to GPU.
//   - STARPU_CUDA_ASYNC: all async work must be submitted before returning.
//     cudaStreamSynchronize is used only when host-side I/O requires it.
//   - Forward snapshots are in host RAM (MEMORY) or on disk (DISK).
//     A 3-slot rolling device buffer (d_p_hi/mid/lo) avoids uploading all
//     nt snapshots to GPU at once.
//   - No C++23 headers: use printf/fprintf instead of std::println.

#include "codelets/backward_propagation.hpp"
#include "codelets/forward_propagation.hpp" // ShotData, TaskConfig, CodeletArg
#include <cuda_runtime.h>
#include <starpu.h>
#include <starpu_cuda.h>
#include <cstdio>
#include <cstring>
#include <cstdlib>  // std::remove
#include <fstream>

namespace starfwi {

// ─────────────────────────────────────────────────────────────────────────────
// Device kernels
// ─────────────────────────────────────────────────────────────────────────────

// Compute vel²[i] and the per-point gradient factor −2/(dt·c³[i]) in one pass.
__global__ void bwd_prep_kernel(
    const float * __restrict__ vel,
    float       * __restrict__ vel2,
    float       * __restrict__ grad_factor,
    int n, float dt)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float c  = vel[i];
    float c2 = c * c;
    vel2[i]        = c2;
    float c3       = c2 * c;
    grad_factor[i] = (c3 > 0.0f) ? (-2.0f / (dt * c3)) : 0.0f;
}

// 2D finite-difference stencil for the adjoint wavefield (identical physics
// to fd2d_step_kernel in forward_propagation.cu).
//   p_new = 2·p_cur − p_old + vel²·dt²·(∂²p/∂x² + ∂²p/∂z²)
// Dirichlet (zero-pressure) boundary conditions on all edges.
__global__ void bwd_fd2d_step_kernel(
    const float * __restrict__ p_cur,
    const float * __restrict__ p_old,
    float       * __restrict__ p_new,
    const float * __restrict__ vel2,
    int nx, int nz,
    float dx2_inv, float dz2_inv, float dt2)
{
    int ix = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int iz = (int)(blockIdx.y * blockDim.y + threadIdx.y);
    if (ix >= nx || iz >= nz) return;

    int idx = ix + nx * iz;
    if (ix == 0 || ix == nx - 1 || iz == 0 || iz == nz - 1) {
        p_new[idx] = 0.0f;
        return;
    }

    float d2p_dx2 = (p_cur[idx-1] - 2.0f*p_cur[idx] + p_cur[idx+1]) * dx2_inv;
    float d2p_dz2 = (p_cur[idx-nx] - 2.0f*p_cur[idx] + p_cur[idx+nx]) * dz2_inv;
    p_new[idx] = 2.0f*p_cur[idx] - p_old[idx]
                 + vel2[idx] * dt2 * (d2p_dx2 + d2p_dz2);
}

// Inject time-reversed residuals as adjoint point sources.
// residuals layout: shot->residuals[r * nt + s]  (s = adjoint time index)
// Uses nearest-grid-point (matches CPU to_grid helper).
__global__ void inject_adjoint_sources_kernel(
    float       * __restrict__ p_adj,
    const int   * __restrict__ rx_ix,
    const int   * __restrict__ rx_iz,
    const float * __restrict__ residuals,
    int n_receivers, int nt, int s, int nx)
{
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= n_receivers) return;
    float adj_src = residuals[r * nt + s];
    if (adj_src == 0.0f) return;
    atomicAdd(&p_adj[rx_ix[r] + nx * rx_iz[r]], adj_src);
}

// Cross-correlation imaging condition:
//   gradient[i] += grad_factor[i] · p̈[i] · q[i]
// where  p̈ = (p_hi − 2·p_mid + p_lo) · dt2_inv
// and    grad_factor[i] = −2 / (dt · c³[i])
// This is equivalent to the CPU formula: grad += (−2·dt/c³) · p̈ · q.
__global__ void accumulate_gradient_kernel(
    const float * __restrict__ p_hi,
    const float * __restrict__ p_mid,
    const float * __restrict__ p_lo,
    const float * __restrict__ q,
    const float * __restrict__ grad_factor,
    float       * __restrict__ gradient,
    int n, float dt2_inv)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float p_ddot = (p_hi[i] - 2.0f*p_mid[i] + p_lo[i]) * dt2_inv;
    gradient[i] += grad_factor[i] * p_ddot * q[i];
}

// ─────────────────────────────────────────────────────────────────────────────
// Device allocation helper: allocate and zero-init a float array
// ─────────────────────────────────────────────────────────────────────────────
static float *cuda_alloc_zero(size_t n_floats, cudaStream_t stream) {
    float *p = nullptr;
    cudaMalloc(&p, n_floats * sizeof(float));
    if (p) cudaMemsetAsync(p, 0, n_floats * sizeof(float), stream);
    return p;
}

// ─────────────────────────────────────────────────────────────────────────────
// CUDA codelet entry point
// ─────────────────────────────────────────────────────────────────────────────

void backward_propagation_cuda(void *buffers[], void *cl_arg) {
    // ── StarPU buffer extraction ──────────────────────────────────────────────
    // VECTOR → device pointers (auto-transferred by StarPU)
    const float      *d_velocity = (const float *)STARPU_VECTOR_GET_PTR(buffers[0]);
    // VARIABLE → host pointers (never transferred)
    ShotData         *shot       = (ShotData *)STARPU_VARIABLE_GET_PTR(buffers[1]);
    const TaskConfig *tc         = (const TaskConfig *)STARPU_VARIABLE_GET_PTR(buffers[2]);
    const float      *d_recv_x   = (const float *)STARPU_VECTOR_GET_PTR(buffers[3]);
    // buffers[4] recv_y: unused in 2D path
    const float      *d_recv_z   = (const float *)STARPU_VECTOR_GET_PTR(buffers[5]);

    CodeletArg *arg      = (CodeletArg *)cl_arg;
    const char *hostname = arg ? arg->hostname : "unknown";
    bool        verbose  = arg ? arg->verbose  : false;

    if (shot->residuals.empty()) {
        if (verbose)
            printf("[starfwi][%s][backward_propagation_cuda] Shot %zu: "
                   "no residuals, skipping\n", hostname, shot->shot_id);
        return;
    }

    cudaStream_t stream = starpu_cuda_get_local_stream();

    const int   nx        = (int)tc->nx;
    const int   nz        = (int)tc->nz;
    const int   nt        = (int)tc->nt;
    const float dx        = tc->dx;
    const float dz        = tc->dz;
    const float dt        = tc->dt;
    const int   n_recv    = (int)tc->n_receivers;
    const int   grid_size = nx * nz;

    const float dx2_inv = 1.0f / (dx * dx);
    const float dz2_inv = 1.0f / (dz * dz);
    const float dt2     = dt * dt;
    const float dt2_inv = 1.0f / dt2;

    const bool use_memory = (shot->wavefield_storage_actual == 0);

    if (verbose)
        printf("[starfwi][%s][backward_propagation_cuda] Shot %zu: "
               "adjoint propagation (nt=%d, grid=%d, storage=%s)\n",
               hostname, shot->shot_id, nt, grid_size,
               use_memory ? "MEMORY" : "DISK");

    // ── Open disk wavefield file early (before any allocations) ──────────────
    std::ifstream wf_in;
    char          wf_path[512] = {};
    float        *h_snap_staging = nullptr; // pinned, 1-snapshot staging buffer

    if (!use_memory) {
        snprintf(wf_path, sizeof(wf_path), "%s/fwd_shot_%zu.bin",
                 tc->wavefield_dir, shot->shot_id);
        wf_in.open(wf_path, std::ios::binary);
        if (!wf_in) {
            fprintf(stderr,
                    "[starfwi][%s][backward_propagation_cuda] ERROR: Shot %zu "
                    "cannot open wavefield file '%s'\n",
                    hostname, shot->shot_id, wf_path);
            return;
        }
        cudaMallocHost(&h_snap_staging, (size_t)grid_size * sizeof(float));
    }

    // ── Kernel launch config ──────────────────────────────────────────────────
    dim3 block2d(32, 8);
    dim3 grid2d((nx + 31) / 32, (nz + 7) / 8);
    int  scalar_blocks = (grid_size + 255) / 256;
    dim3 recv_block(128);
    dim3 recv_grid((n_recv + 127) / 128);

    // ── Device allocations ────────────────────────────────────────────────────
    float *d_vel2        = cuda_alloc_zero((size_t)grid_size, stream);
    float *d_grad_factor = cuda_alloc_zero((size_t)grid_size, stream);
    float *d_gradient    = cuda_alloc_zero((size_t)grid_size, stream);
    bwd_prep_kernel<<<scalar_blocks, 256, 0, stream>>>(
        d_velocity, d_vel2, d_grad_factor, grid_size, dt);

    // Adjoint wavefields: 3 levels.
    // d_adj_alloc holds original addresses for cleanup; working pointers are swapped.
    float *d_adj_alloc[3] = {
        cuda_alloc_zero((size_t)grid_size, stream),
        cuda_alloc_zero((size_t)grid_size, stream),
        cuda_alloc_zero((size_t)grid_size, stream),
    };
    float *d_adj     = d_adj_alloc[0]; // current field (source injected here)
    float *d_adj_old = d_adj_alloc[1]; // previous field
    float *d_adj_new = d_adj_alloc[2]; // written by FD kernel, becomes next current

    // Forward snapshot rolling buffer: 3 slots (p_hi, p_mid, p_lo).
    // d_snap_alloc holds original addresses for cleanup.
    float *d_snap_alloc[3] = {
        cuda_alloc_zero((size_t)grid_size, stream), // p[t_fwd+1] — zero init = p[nt] ≈ 0
        cuda_alloc_zero((size_t)grid_size, stream), // p[t_fwd]
        cuda_alloc_zero((size_t)grid_size, stream), // p[t_fwd-1]
    };
    float *d_p_hi  = d_snap_alloc[0];
    float *d_p_mid = d_snap_alloc[1];
    float *d_p_lo  = d_snap_alloc[2];

    // Receiver grid indices and residuals on device
    int   *d_rx_ix    = nullptr;
    int   *d_rx_iz    = nullptr;
    float *d_residuals = nullptr;
    cudaMalloc(&d_rx_ix,     (size_t)n_recv * sizeof(int));
    cudaMalloc(&d_rx_iz,     (size_t)n_recv * sizeof(int));
    cudaMalloc(&d_residuals, (size_t)n_recv * nt * sizeof(float));

    // ── Receiver grid indices: D2H → compute → H2D ───────────────────────────
    // d_recv_x / d_recv_z are device pointers; bring them to host to compute
    // integer grid indices (nearest-grid-point, matches CPU to_grid helper).
    {
        float *h_recv_x = new float[n_recv];
        float *h_recv_z = new float[n_recv];
        cudaMemcpyAsync(h_recv_x, d_recv_x, (size_t)n_recv * sizeof(float),
                        cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(h_recv_z, d_recv_z, (size_t)n_recv * sizeof(float),
                        cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);

        int *h_rx_ix = new int[n_recv];
        int *h_rx_iz = new int[n_recv];
        for (int r = 0; r < n_recv; ++r) {
            int ix = (int)(h_recv_x[r] / dx);
            int iz = (int)(h_recv_z[r] / dz);
            h_rx_ix[r] = max(0, min(ix, nx - 1));
            h_rx_iz[r] = max(0, min(iz, nz - 1));
        }
        cudaMemcpyAsync(d_rx_ix, h_rx_ix, (size_t)n_recv * sizeof(int),
                        cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(d_rx_iz, h_rx_iz, (size_t)n_recv * sizeof(int),
                        cudaMemcpyHostToDevice, stream);
        delete[] h_recv_x; delete[] h_recv_z;
        delete[] h_rx_ix;  delete[] h_rx_iz;
    }

    // Upload residuals to device
    cudaMemcpyAsync(d_residuals, shot->residuals.data(),
                    (size_t)n_recv * nt * sizeof(float),
                    cudaMemcpyHostToDevice, stream);

    // ── Snapshot load helper ──────────────────────────────────────────────────
    // Loads forward snapshot at t_snap into d_dst.
    // MEMORY: async H2D on stream — sequenced with subsequent kernels on same
    //   stream, so d_dst is ready before the next gradient kernel runs.
    // DISK: read into pinned staging then async H2D, then sync so the staging
    //   buffer is safe to reuse on the next call.
    // t_snap outside [0, nt) → zeroes d_dst (p ≈ 0 outside the time window).
    auto load_snap = [&](int t_snap, float *d_dst) {
        if (t_snap < 0 || t_snap >= nt) {
            cudaMemsetAsync(d_dst, 0, (size_t)grid_size * sizeof(float), stream);
            return;
        }
        if (use_memory) {
            cudaMemcpyAsync(d_dst,
                            shot->pressure_snapshots.data() + (size_t)t_snap * grid_size,
                            (size_t)grid_size * sizeof(float),
                            cudaMemcpyHostToDevice, stream);
        } else {
            wf_in.seekg((std::streamoff)((size_t)t_snap * grid_size * sizeof(float)));
            wf_in.read(reinterpret_cast<char *>(h_snap_staging),
                       (size_t)grid_size * sizeof(float));
            cudaMemcpyAsync(d_dst, h_snap_staging,
                            (size_t)grid_size * sizeof(float),
                            cudaMemcpyHostToDevice, stream);
            // Sync so staging buffer is free for the next call.
            cudaStreamSynchronize(stream);
        }
    };

    // ── Prime the rolling buffer ──────────────────────────────────────────────
    // At s=0, t_fwd = nt-1:
    //   d_p_hi  = p[nt]   ≈ 0  (already zero from init above)
    //   d_p_mid = p[nt-1]
    //   d_p_lo  = p[nt-2]
    load_snap(nt - 1, d_p_mid);
    load_snap(nt - 2, d_p_lo);
    cudaStreamSynchronize(stream); // ensure priming is complete before the loop

    // ── Adjoint backward time loop ────────────────────────────────────────────
    for (int s = 0; s < nt; ++s) {
        const int t_fwd = nt - 1 - s;

        // 1. Inject time-reversed residuals as adjoint sources into d_adj.
        //    Matches CPU: apply_source_at modifies pressure_ before step().
        if (n_recv > 0) {
            inject_adjoint_sources_kernel<<<recv_grid, recv_block, 0, stream>>>(
                d_adj, d_rx_ix, d_rx_iz, d_residuals, n_recv, nt, s, nx);
        }

        // 2. Advance adjoint wavefield: reads d_adj (with sources) and d_adj_old,
        //    writes d_adj_new (Dirichlet boundaries applied inside the kernel).
        bwd_fd2d_step_kernel<<<grid2d, block2d, 0, stream>>>(
            d_adj, d_adj_old, d_adj_new, d_vel2, nx, nz, dx2_inv, dz2_inv, dt2);

        // 3. Accumulate gradient (skip t_fwd == 0 and t_fwd == nt-1 where
        //    the p̈ finite-difference stencil is not well-defined).
        if (t_fwd >= 1 && t_fwd < nt - 1) {
            accumulate_gradient_kernel<<<scalar_blocks, 256, 0, stream>>>(
                d_p_hi, d_p_mid, d_p_lo,
                d_adj_new,      // q = adjoint field after this step
                d_grad_factor, d_gradient,
                grid_size, dt2_inv);
        }

        // 4. Rotate adjoint time levels (pointer swap, no data copy).
        {
            float *tmp = d_adj_old;
            d_adj_old  = d_adj;
            d_adj      = d_adj_new;
            d_adj_new  = tmp;
        }

        // 5. Advance forward snapshot rolling buffer.
        //    At the next iteration t_fwd' = t_fwd - 1:
        //      d_p_hi  must be p[t_fwd]     → current d_p_mid
        //      d_p_mid must be p[t_fwd - 1] → current d_p_lo
        //      d_p_lo  must be p[t_fwd - 2] → load now
        if (t_fwd >= 1) {
            float *tmp = d_p_hi;
            d_p_hi     = d_p_mid;
            d_p_mid    = d_p_lo;
            d_p_lo     = tmp;           // reuse old d_p_hi slot
            load_snap(t_fwd - 2, d_p_lo);
        }
    }

    // ── Copy gradient to host ─────────────────────────────────────────────────
    // STARPU_CUDA_ASYNC: StarPU records a CUDA event after we return; downstream
    // tasks that depend on the shot handle will not start until that event fires,
    // guaranteeing shot->gradient is fully populated before any consumer sees it.
    shot->gradient.resize((size_t)grid_size);
    cudaMemcpyAsync(shot->gradient.data(), d_gradient,
                    (size_t)grid_size * sizeof(float),
                    cudaMemcpyDeviceToHost, stream);

    // ── Release forward snapshot resources ────────────────────────────────────
    // For DISK: close and delete the temp wavefield file (matches CPU path).
    // For MEMORY: pressure_snapshots will be freed when ShotData is destroyed.
    //   Clearing eagerly would require a completion callback since the async
    //   gradient D2H above is still in flight.
    if (!use_memory) {
        wf_in.close();
        std::remove(wf_path);
    }

    // ── Cleanup ───────────────────────────────────────────────────────────────
    cudaFree(d_vel2);
    cudaFree(d_grad_factor);
    cudaFree(d_gradient);
    for (int k = 0; k < 3; ++k) cudaFree(d_adj_alloc[k]);
    for (int k = 0; k < 3; ++k) cudaFree(d_snap_alloc[k]);
    cudaFree(d_rx_ix);
    cudaFree(d_rx_iz);
    cudaFree(d_residuals);
    if (h_snap_staging) cudaFreeHost(h_snap_staging);

    if (verbose)
        printf("[starfwi][%s][backward_propagation_cuda] Shot %zu: "
               "adjoint propagation complete\n", hostname, shot->shot_id);
}

} // namespace starfwi
