# Towards Compiler-Generated Distributed Tree Traversal

## Motivation

Acceleration structures such as BVHs are central to ray tracing, collision detection, and nearest-neighbor queries. As scene scale increases (millions to billions of primitives), acceleration structure memory can exceed the capacity of a single device. At that point, systems either fail, degrade via paging/out-of-core behavior, or require handcrafted distributed implementations with custom partitioning, replication, and ray routing logic when a machine with larger capacity is unavailable. 

Recent compiler systems have substantially improved **single-device** tree traversal:

- **Bonsai** compiles high-level declarative queries into work-efficient pruned traversals via symbolic interval/predicate reasoning.
- **Scion** separates traversal semantics from physical layout, enabling systematic data-layout exploration without changing traversal code.

Together they form a clean compilation stack:

**query/specification -> traversal IR -> physical layout -> executable**

However, this stack currently assumes a unified memory space on one device. There is no first-class support for distribution, sharding, replication, or movement across multiple memory domains (e.g., L1/L2 clusters, sockets, GPUs, nodes).

## Project Hypothesis

A runtime-layer extension can make Bonsai/Scion-generated code usable in distributed settings **without modifying compiler internals initially**.

Core idea:
1. Keep existing Bonsai/Scion code generation unchanged.
2. Add a runtime layer that controls partitioning, scheduling, and ray/data movement.
3. Determine, empirically, **which workload + memory-hierarchy regimes benefit from sharding/distribution**.
4. Use those findings to motivate a general abstraction that could eventually be compiler-generated.

This project is not, at first, about designing new language syntax and semantics. It is about establishing **when distribution is worth it**, and what control knobs are needed. 

---

## Related Work

- **Bonsai (Root et al., PLDI 2026)**  
  Generates pruned tree traversals from high-level queries; lowers TTIR to efficient CUDA/C++ for single-device execution.

- **Scion (Gyurgyik et al., PLDI 2026)**  
  Decouples logical tree structure from physical layout; enables layout exploration in a single address space.

- **R2E2 (Fouladi et al., SIGGRAPH 2022)**  
  Distributed BVH ray tracing with manual treelet partitioning and replication; strong results but scene-specific and hand-tuned.

- **Ray Queue Cycling (Wald et al., HPG 2023)**  
  Multi-GPU data-parallel path tracing with ring-based ray queue cycling; demonstrates practical distribution on single-node multi-GPU systems, implemented manually in OptiX/CUDA.

- **Sequoia (Fatahalian et al., SC 2006)**  
  Hierarchical machine models for memory-aware scheduling; highly relevant conceptually, but focused on streaming kernels rather than irregular tree traversal.

- **Legion (Bauer et al., SC 2012)**  
  Distributed region-based runtime and mapping model; useful reference for logical partitioning and placement policies.

---

## Theoretical framing for this project

Distribution/sharding is useful only when its gains exceed its overhead.  
For this project, we model that tradeoff as:

**Net Benefit ≈ (Locality/aggregate-bandwidth gain from distribution/sharding) - (ray/data transfer + queue/sync overhead)**

A practical criterion:

Distribution is promising when:
- baseline is memory-system constrained at a target hierarchy level (e.g., L2<->L1 or socket<->socket),
- partitioning reduces pressure at that level,
- communication/synchronization overhead is amortized by reduced stall time.

Distribution is not promising when:
- workload is compute-bound,
- per-ray hot set already fits local cache well,
- communication dominates.

This criterion motivates the checkpoint experiments below.

---

## Project Tasks

### Task 1: Establish single-device baseline
Reproduce Bonsai RTIOW closest-hit performance on:
- Sherlock CPU backend
- Apple M4 Pro CPU backend

### Task 2: Build evaluation harness to identify beneficial regimes
Design and run a harness varying:
- scene size (`RTIOW_GRID_HALF`),
- thread concurrency (`t1..tN` variants),
- ray traffic (`RTIOW_SAMPLES`).

Collect:
- wall-clock metrics,
- structural metrics (`n_nodes`, `bvh_bytes`, etc.),
- hardware counters (where available: L1D/L2 miss behavior, stall cycles).

Goal: identify which regimes are bandwidth/latency constrained and therefore plausible targets for sharding/distribution.

### Task 3: Handwritten distributed RTIOW prototype + transfer-cost analysis
Implement a distributed/sharded RTIOW path in handwritten runtime code (no compiler changes yet), and measure:
- speedup/slowdown vs baseline,
- ray transfer volume and penalty,
- queue/sync overhead.

Goal: quantify the communication-vs-locality tradeoff directly.

### Task 4: Two backend demonstration + abstraction proposal
Demonstrate Task 3 on two machine backends with different memory hierarchies and derive a common abstraction suitable for future Bonsai/Scion integration.

---

## Checkpoint 1 Status

### Backend status
We target two backends with fundamentally different memory topologies. The M4 Pro has a flat unified memory with cache-level partitioning; Sherlock has physically separate DRAM pools connected by an explicit interconnect. Together they let us study the distribution question at two different scales of the memory hierarchy.
 
**Backend 1: Apple M4 Pro**
 
```
┌─── Performance Cluster 0 ─────────────────┐  ┌─── Performance Cluster 1 ─────────────────┐
│ Core 0   Core 1   Core 2   Core 3  Core 4 │  │ Core 5   Core 6   Core 7   Core 8  Core 9 │
│ L1d:128K L1d:128K L1d:128K L1d:128K  128K │  │ L1d:128K L1d:128K L1d:128K L1d:128K  128K │
│ (private, ~1ns)                           │  │ (private, ~1ns)                           │
│               Shared L2: 16 MB            │  │               Shared L2: 16 MB            │
│               (~4ns, 5 cores share)       │  │               (~4ns, 5 cores share)       │
└───────────────────┬───────────────────────┘  └──────────────────┬────────────────────────┘
                    │                                             │
                    └─────────────┬───────────────────────────────┘
                                  │
                    ┌─────────────┴──────────────┐
                    │   Unified DRAM (LPDDR5X)   │
                    │   273 GB/s peak, ~80-100ns │
                    │   24 or 48 GB              │
                    └───────────────────────────-┘
 
              ┌───────────────────────────────────────┐
              │       Efficiency Cluster              │
              │ Core 10  Core 11  Core 12  Core 13    │
              │ L1d:64K  L1d:64K  L1d:64K  L1d:64K    │
              │           Shared L2: 4 MB             │
              └───────────────────────────────────────┘
```
 
Each core has a private L1 data cache (128 KB on perf cores). Five performance cores share a 16 MB L2. All cores share a single unified DRAM pool. Cache line size is 128 bytes (not 64 as on x86). 

**Backend 2: Sherlock `sh02-01n58` (dual-socket NUMA)**
 
```
┌──────────── Socket 0 (NUMA Node 0) ───────────┐     ┌──────────── Socket 1 (NUMA Node 1) ───────────┐
│                                               │     │                                               │
│  Core 0 ── L1d ──┐                            │     │  Core 48 ── L1d ──┐                           │
│  Core 1 ── L1d ──┤                            │     │  Core 49 ── L1d ──┤                           │
│  ...              ├── L2 (per-core, 1 MB)     │     │  ...               ├── L2 (per-core, 1 MB)    │
│  Core 47 ── L1d ──┘                           │     │  Core 95 ── L1d ──┘                           │
│                                               │     │                                               │
│           Shared LLC: 30-60 MB                │     │           Shared LLC: 30-60 MB                │
│           (~15-20ns)                          │     │           (~15-20ns)                          │
│                  │                            │     │                  │                            │
│        Memory Controller (IMC 0)              │     │        Memory Controller (IMC 1)              │
│                  │                            │     │                  │                            │
│           DRAM 0: 96 GB                       │     │           DRAM 1: 96 GB                       │
│           ~100 GB/s, ~80ns                    │     │           ~100 GB/s, ~80ns                    |
│                                               │     │                                               │
└──────────────────┬────────────────────────────┘     └───────────────────┬───────────────────────────┘
                   │                                                      │
                   │                UPI Interconnect                      │
                   │              ~50 GB/s per direction                  │
                   │              ~140-180ns round trip                   │
                   └──────────────────────────────────────────────────────┘
```
 
Each socket has its own physical DRAM pool attached via its own integrated memory controller. The two sockets are connected by UPI (Ultra Path Interconnect). A thread on socket 0 can access DRAM 1, but the cache line must traverse UPI, paying ~140-180 ns instead of ~80 ns for local DRAM, and consuming limited UPI bandwidth (~50 GB/s) instead of local DRAM bandwidth (~100 GB/s). The OS exposes both pools as a single virtual address space; `numactl` and `mbind` control physical page placement. 

### Workload status
- Current workload: RTIOW spheres.
- Observation: RTIOW sphere scenes can be have small per-ray hot set, coherent traversal to strongly expose distribution benefits.
- Next workload class: memory-heavier/divergence-heavy scenes (e.g., hair/curve-rich scenes) to stress cache/bandwidth behavior more realistically.

### Harness status
Implemented under for L2<->L1 for the case of M4 Pro ([README](https://github.com/sgauthamr2001/bonsai/tree/main/apps/rtiow/cpu/bench)):

`code/bonsai/apps/rtiow/cpu/bench/`

Includes:
- `run_grid_sweep.sh`
- `run_thread_sweep.sh`
- `run_samples_sweep.sh`
- CSV parsing/aggregation scripts
- plotting scripts for grid/thread/samples curves
- optional counter capture wrapper (`run_xctrace.sh`)

### Current interpretation
- To get the harness working it was implemented for L2<->L1 level, initial samples scaling appears close to linear on current RTIOW settings, suggesting no strong evidence yet that L2<->L1 bandwidth contention dominates.
- Therefore, checkpoint result is **not** “distribution helps now,” but they are rather placeholder plots indicating how:
  - the evaluation infrastructure is run,
  - baseline behavior is characterized,
  - criteria for deciding when distribution/sharding can be justified. 

---

## Planned next steps (post-checkpoint)

1. Run counter-backed analysis on selected anchors (small/mid/large grid, low/high thread count).
2. Add memory-heavier benchmark scenes (hair/curve-like or similarly irregular geometry/material access).
3. Extend Bonsai’s tree representation to use 32-bit node indices so stress tests can exceed today’s scene-size limits without failure. Bigger scenes lengthen each run, so the evaluation needs to be carefully designed to avoid large design and testing times. For that reason, we build the harness at L2 <-> L1D level first. The CPU harness was exercised on Apple Silicon; threading via GCD may require checks or small portability fixes on Intel (Sherlock). (Similar fixes were required to run Bonsai on Sherlock).
4. Implement handwritten sharded/distributed prototype and measure ray movement overhead explicitly.
5. Compare two backends and come up with a portable abstraction.

---

## Repository Structure

Bonsai is included as a Git submodule at `code/bonsai` ([sgauthamr2001/bonsai](https://github.com/sgauthamr2001/bonsai))

After cloning this project, initialize submodules:

```bash
git submodule update --init --recursive
```

Or clone with submodules in one step:

```bash
git clone --recurse-submodules <url-of-this-repo>
```
```
cs348k-project/
├── code/bonsai/             # Submodule: Bonsai compiler + RTIOW benchmark harness
├── checkpoint_2             # Files and document for checkpoint2
├── README.md                # Project README (this file)
```
---

## Checkpoint-2: 
Described here: ([Checkpoint-2](https://github.com/sgauthamr2001/cs348k_project/checkpoint_2))

```