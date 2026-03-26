# StarFWI

A distributed Full Waveform Inversion (FWI) benchmark built on [StarPU](https://starpu.gitlabpages.inria.fr/) and StarPU-MPI. It is designed as a realistic HPC workload for studying fault-tolerant, elastic execution on heterogeneous multi-node systems.

---

## Getting Started

### Prerequisites

| Requirement | Notes |
|---|---|
| Podman or Docker | Container runtime |
| podman-compose or docker-compose | For multi-node test setup |
| A sibling `../starpu` directory | Local StarPU fork built into the image |

The build injects the local StarPU fork using `--build-context`, so both repositories must live in the same parent directory:

```
Code/
├── starpu/       ← local StarPU fork (must be here)
└── starfwi/      ← this repository
```

### Makefile targets

```
make build    Build the container image (compiles StarPU + StarFWI inside the container)
make start    Start the two MPI test nodes (node1, node2)
make test     Run the full two-phase test pipeline
make trace    Convert FxT traces and render StarVZ plots (requires starvz-shell)
make kill     Stop and remove all running containers
make clean    Stop containers and delete the image
```

### Running the test

```bash
make test
```

This command builds the image if needed, starts two containers (`node1`, `node2`), and runs the full pipeline automatically.

What happens internally:

**Phase 1 — Modeling** (`starfwi-modeling`)
Forward-propagates the true Marmousi P-wave velocity model and writes one synthetic seismogram per shot to `/shared/observed/`.

```
mpirun --app appfile-modeling
  node1: 1 MPI rank, 2 StarPU CPU workers
  node2: 1 MPI rank, 3 StarPU CPU workers
  model: MODEL_P-WAVE_VELOCITY_1.25m.segy
  shots: 8, timesteps: 100
```

**Phase 2 — Inversion** (`starfwi-fwi`)
Loads the observed seismograms produced in Phase 1. Applies a 5% velocity reduction to the true model to create a non-trivial starting point, then runs the inversion pipeline.

```
mpirun --app appfile-fwi
  node1: 1 MPI rank, 2 StarPU CPU workers
  node2: 1 MPI rank, 3 StarPU CPU workers
  model: MODEL_P-WAVE_VELOCITY_1.25m.segy  (scaled by 0.95)
  shots: 8, timesteps: 100
  observed-dir: /shared/observed
```

The shared volume `.local_shared_volume/` is mounted at `/shared` inside both containers, giving all MPI ranks access to the same observed data, performance model cache, FxT trace files, and wavefield snapshots.

### IDE support (clangd)

Copy `.clangd.example` to `.clangd` and build the project once inside the container to generate `build/compile_commands.json`. clangd will pick that up for accurate code navigation and diagnostics.

---

## How It Works

### Full Waveform Inversion

FWI is an iterative method that recovers subsurface physical properties (here: acoustic wave velocity) by minimising the difference between observed seismic recordings and synthetic ones produced by a numerical wave simulation. The objective function is the L2 misfit:

```
J(m) = ½ Σ_shots || d_syn(m) - d_obs ||²
```

Each iteration requires three passes per shot:

1. **Forward propagation** — simulate wave propagation through the current velocity model and record synthetic seismograms at receiver positions.
2. **Misfit computation** — compute residuals `r = d_syn − d_obs` and the scalar misfit value J.
3. **Adjoint (backward) propagation** — propagate the adjoint wavefield driven by the time-reversed residuals, and cross-correlate it with the forward wavefield to obtain the gradient ∂J/∂m.

The gradient is then used by an optimisation step (e.g. gradient descent) to update the velocity model, and the loop repeats.

### Two-binary design

StarFWI uses two separate executables, following the approach used by the [mamute](https://github.com/HPCSys-Lab/simwave) family of FWI codes:

```
starfwi-modeling   Forward model the true velocity → write observed seismograms to disk
starfwi-fwi        Load observed data → iterate forward / misfit / adjoint passes
```

This separation keeps the two concerns independent: modeling only needs `forward_propagation` + `save_observed`; inversion adds `compute_misfit` + `backward_propagation`. It also lets the two phases run with different velocity models (true vs. perturbed) without any special-casing inside the code.

### Task scheduling with StarPU-MPI

StarPU is a task-based runtime for heterogeneous architectures. Each unit of computation is a **codelet** — a function descriptor that can have CPU and/or CUDA implementations. StarPU schedules codelets onto available workers and manages data transfers between memory spaces transparently.

StarPU-MPI extends this to multi-node systems. Each data handle is assigned an MPI tag and an owner rank. When a task running on rank A needs data owned by rank B, StarPU-MPI issues the necessary `MPI_Send`/`MPI_Recv` automatically.

**Collective task submission requirement**: all MPI ranks must submit all tasks with identical conditions, even if a given task will only execute on one rank. This is why every rank loads every shot's observed data from disk during startup — it ensures the `!observed_data.empty()` condition that guards task submission is uniformly true across all ranks.

#### Modeling task DAG (per shot)

```
forward_propagation  →  save_observed
```

`forward_propagation` simulates the wave and fills `ShotData::synthetic_data`.
`save_observed` writes those synthetic seismograms to `observed/observed_shot_NNN.bin`.

#### Inversion task DAG (per shot)

```
forward_propagation  →  compute_misfit  →  backward_propagation
```

`forward_propagation` fills `ShotData::synthetic_data`.
`compute_misfit` computes `residuals = synthetic − observed` and the scalar `misfit`.
`backward_propagation` re-runs the forward simulation to rebuild pressure snapshots, then propagates the adjoint wavefield driven by the time-reversed residuals, accumulating the gradient in `ShotData::gradient` via the cross-correlation imaging condition:

```
grad[x] += (−2 / c³[x]) × p̈[x,t] × q[x,T−t] × dt
```

where `p̈` is the second time-derivative of the forward pressure field and `q` is the adjoint field.

#### Shot distribution across MPI ranks

Shots are distributed in round-robin order: shot `i` is owned by rank `i % size`. With 8 shots and 2 ranks, each rank owns 4 shots. StarPU-MPI schedules each shot's task chain on its owner rank and automatically transfers the shared velocity model and receiver geometry to wherever they are needed.

### Data structures

| Struct | Purpose |
|---|---|
| `SimulationConfig` | Grid dimensions, spacing, time step, velocity model data loaded from SEG-Y |
| `TaskConfig` | POD version of simulation parameters — safe to copy via MPI as a StarPU variable handle |
| `ShotData` | All per-shot state: source position, wavelet, synthetic/observed seismograms, residuals, misfit, gradient |
| `ReceiverGeometry` | Global fixed receiver array shared across all shots |

### Fault tolerance

`starfwi-fwi` integrates with StarPU's MPI fault-tolerance layer (`starpu_mpi_ft.h`). When `--checkpoint-dir` and `--checkpoint-interval` are provided, the application flushes a checkpoint to shared storage every N shots. On restart, `starpu_mpi_init_from_checkpoint()` detects the saved checkpoint and restores shot data handles, allowing the inversion to resume from the last saved state even if the number of MPI ranks changes between runs (elastic restart).

### Source layout

```
starfwi/
├── src/
│   ├── main_modeling.cpp          Entry point for starfwi-modeling
│   ├── main_fwi.cpp               Entry point for starfwi-fwi
│   ├── acoustics/
│   │   ├── finite_difference_solver.cpp   2nd-order FD acoustic wave solver
│   │   ├── receiver_recorder.cpp          Interpolates pressure to receiver positions
│   │   └── misfit.cpp                     L2 misfit and residual computation
│   ├── codelets/
│   │   ├── forward_propagation.cpp        StarPU codelet: forward wave simulation
│   │   ├── save_observed.cpp              StarPU codelet: write seismograms to disk
│   │   ├── compute_misfit.cpp             StarPU codelet: misfit and residuals
│   │   └── backward_propagation.cpp       StarPU codelet: adjoint state gradient
│   └── utils/
│       ├── segy_loader.cpp                SEG-Y velocity model reader
│       ├── seismogram_io.cpp              Binary seismogram read/write
│       ├── wavelet.cpp                    Ricker wavelet generation
│       ├── snapshot_writer.cpp            Wavefield snapshot writer
│       └── cli_parser.cpp                 Command-line argument parser
├── include/                       Corresponding headers
├── CMakeLists.txt                 Builds starfwi-modeling and starfwi-fwi
├── Containerfile                  Container image definition
├── compose.yml                    Two-node test cluster definition
└── Makefile                       Developer workflow (build / test / trace)
```

---

## Comparison with Mamute

[Mamute](https://github.com/HPCSys-Lab/simwave) is a production-quality acoustic FWI code that also uses the two-binary approach and inspired StarFWI's workflow design. The table below summarises the key differences.

### Parallelism

| | StarFWI | Mamute |
|---|---|---|
| Intra-node | StarPU task runtime (auto-schedules to CPU/GPU workers) | OpenMP (`collapse(3)` over grid loops) |
| Inter-node | StarPU-MPI (data handles with MPI tags, collective task submission) | Plain MPI + MPI Windows for work stealing |
| Shot distribution | Static round-robin at registration time | Static block **or** dynamic work-stealing (MPI RMA) |
| Load balancing | None — StarPU is the subject of study, not load balancing | Optional work-stealing: idle ranks steal shots from busy ones |
| GPU support | Built in — codelets declare CPU and CUDA implementations, StarPU picks at runtime | Not present |

### Forward wavefield storage during adjoint

The adjoint state method needs the forward pressure field `p[x,t]` while running the adjoint backward in time. How that field is stored is one of the most consequential design choices in any FWI code.

**Mamute** offers three compile-time strategies:

| Strategy | How it works | Memory |
|---|---|---|
| `MEMORY` (default) | Allocate `p[nt][grid]`, store all timesteps | O(nt × grid) |
| `DISK` | Write each snapshot to `wf_shot_#.bin`, read back during adjoint | O(grid) — but I/O bound |
| `CHECKPOINTING` | Revolve algorithm — store only strategically chosen checkpoints, recompute between them | O(√nt × grid) |

The Revolve algorithm (Griewank & Walther, 2000) is the theoretically optimal approach: it minimises total recomputation for a given memory budget. Mamute queries available system RAM at runtime and sets the number of checkpoint buffers accordingly.

**StarFWI** uses a single fixed strategy: **time-reversal backward reconstruction**. The acoustic wave equation is time-symmetric (no dissipation), so the same FD stencil runs both forward and backward:

```
p[t-1] = 2·p[t] - p[t+1] + (c·dt)²·∇²p[t]
```

Only 3 rolling states are needed, giving O(grid) memory without any disk I/O. The trade-off is that absorbing boundary condition errors accumulate slightly as reconstruction proceeds further back in time. Mamute's `CHECKPOINTING` mode avoids this by recomputing the forward field exactly from a stored checkpoint.

### FD stencil and physics

| | StarFWI | Mamute |
|---|---|---|
| Spatial order | **2nd order** (3-point stencil) | **8th order** (5-point stencil per direction) |
| Time order | 2nd order | 2nd order |
| Boundary conditions | Dirichlet (zero) at domain edges | Sponge layer **or** CPML (8th-order), configurable at compile time |
| Free surface | Not implemented | Optional mirror boundary at z=0 |
| Density | Constant | Optional spatially-varying density |

Mamute's 8th-order stencil suppresses numerical dispersion, which matters for real geophysical use. For a benchmark whose subject is StarPU scheduling, 2nd-order is sufficient.

### Optimiser

| | StarFWI | Mamute |
|---|---|---|
| Method | Not implemented — gradient is computed but no velocity update loop | L-BFGS-B (limited-memory quasi-Newton with bound constraints) |
| Line search | — | Wolfe conditions |
| Gradient preconditioning | — | Optional Bessel/Laplace smoothing; near-surface zeroing |

StarFWI computes one complete forward+adjoint pass per shot (the full StarPU task graph), but does not iterate toward convergence. The inversion loop and model update are future work.

### Fault tolerance

| | StarFWI | Mamute |
|---|---|---|
| Mechanism | StarPU-MPI FT API (`starpu_mpi_ft.h`) — checkpoint/restart at the task-graph level | Optional DeLIA library — heartbeat monitoring, per-rank local checkpoint, shot-list recovery |
| Elastic restart | **Yes** — `starpu_mpi_checkpoint_restore_handle` restores data handles even when `n_ranks` changes between failure and restart | **No** — recovery restores the same number of ranks as before the failure |

This is the central research contribution of StarFWI. StarPU's fault-tolerance layer supports **elastic restart**: the job can resume on whatever nodes are available after a failure, not just the original set. DeLIA does not provide this capability.
