# Resume and Interview Notes

These notes summarize the implemented M0–M5 work. They use measured claims from this repository and avoid presenting environment-sensitive resident-mode results as universal speedups.

## 30-Second Project Explanation

I built an exact FP32 vector-similarity search engine in C++17 and CUDA, with a deterministic CPU implementation as the correctness reference. I started with a one-thread-per-vector-pair CUDA kernel, then used stage timing and Nsight Compute to show that its row-major loads used only 4 of 32 bytes in each global-load sector. I redesigned the mapping first to one block and then one warp per pair, achieving fully useful 32/32-byte sectors and about a 9.3x kernel speedup at dimension 768. After the faster kernel exposed database transfer as a system bottleneck, I added an explicit GPU-resident database lifecycle for repeated query batches.

## Resume Bullet Candidates

- Built an exact vector-search engine in C++17/CUDA with deterministic CPU validation, staged `cuda-naive`, block-per-pair, and warp-per-pair backends, and reproducible CMake, CTest, benchmark, and Nsight workflows.
- Used Nsight Compute to trace 4/32-byte global-load-sector utilization to a strided row-major access pattern; redesigned the kernel for coalesced warp-level reduction, reaching 32/32 useful bytes and reducing representative `D=768` kernel latency from 64.447 ms to 6.918 ms (**9.3x**), then added a GPU-resident database lifecycle for repeated queries.

## Key Interview Questions

### 1. Why was `cuda-naive` slow?

One thread handled an entire database-vector/query pair. For a fixed dimension in row-major storage, neighboring threads in a warp read different database rows. At `D=768`, those addresses were about 768 × 4 = 3,072 bytes apart. Nsight Compute measured only 4 useful bytes per 32-byte global-load sector, 97.99% no-eligible-warp cycles in the M2.5 profile, and strong LG-queue and L1TEX-scoreboard stalls. The GPU had high occupancy, but most resident warps were waiting on inefficient memory access.

### 2. Why did block-per-vector help at D=768?

A whole block cooperated on one vector pair, so consecutive threads processed consecutive dimensions. That converted the database access to 32/32 useful bytes per sector. The 256 threads also exposed enough parallel work across a long 768-element dot product to amortize the shared-memory reduction. In the primary run, kernel latency fell from 64.447 ms to 7.879 ms.

### 3. Why did block-per-vector hurt at D=128?

The design still launched 256 threads and performed a full-block shared-memory reduction for a 128-element dot product. Some threads had little or no accumulation work, while synchronization and reduction instructions remained. In the dimension sweep, block measured 5.712 ms versus 3.388 ms for naive; Nsight Compute also identified the CTA barrier as the leading sampled D=128 block stall.

### 4. Why did warp-per-vector improve small-D performance?

A 32-lane warp was a better-sized cooperative unit for shorter vectors. It preserved adjacent, coalesced dimension loads but replaced shared-memory and block-wide barriers with register accumulation and `__shfl_down_sync`. At `D=128`, the warp design executed about 80.4 million dynamic instructions versus 444.0 million for the block design and measured 1.123 ms in the same sweep.

### 5. Why did kernel speedup not translate directly into equal end-to-end speedup?

Kernel time is only one part of a stateless search. Allocation, database and query H2D copies, score D2H, CPU Top-K, cleanup, and host orchestration remain. In the M4 primary run, the warp kernel was about 9.3x faster than naive, but end-to-end latency was 50.381 ms versus 108.782 ms because those non-kernel costs did not receive the same speedup.

### 6. What changed after the database became GPU resident?

Database allocation and H2D upload moved from every search into an explicit preparation phase. Warm searches upload the queries, execute the same M4 warp kernel, copy scores back, and run CPU Top-K. One-shot cold work can be slower because preparation still has to occur, but repeated searches can amortize it. The interface also defines reload and clear behavior so ownership and lifetime are explicit.

### 7. Why was GPU Top-K not prioritized earlier?

The M2 stage data did not support it as the first target. For `N=100,000`, `D=768`, CPU Top-K measured 2.398 ms, while the naive kernel measured 64.144 ms and database H2D measured 31.888 ms. Optimizing Top-K first would have attacked a much smaller component while leaving the demonstrated bottlenecks intact. Keeping Top-K on the CPU also preserved identical ranking semantics across backend experiments.

### 8. What would you optimize next if given more time?

I would re-profile the completed M5 system and choose the next target from a new end-to-end baseline. Candidates include warps-per-block tuning, GPU Top-K to reduce score transfer and host ranking, pinned memory or asynchronous batching, and lower-precision kernels where accuracy requirements permit. Each would be a separate measured milestone with CPU-reference validation; none is implemented in M5.5.

## Claims to Use Carefully

- Prefer the primary **9.3x kernel speedup** because it compares compatible kernel-only measurements from one run.
- State that GPU residency removes repeated database transfer; do not claim that it improves one-shot latency.
- If discussing the measured approximately 29.76x warm-query ratio, identify it as an environment-sensitive M5 repeated-query result whose stateless H2D time was unusually high.
- Describe this as an educational exact-search and CUDA performance-engineering project, not an ANN library, FAISS replacement, vector database, or production serving system.
