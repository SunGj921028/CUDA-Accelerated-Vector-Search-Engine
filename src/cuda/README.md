# CUDA backends

Milestone M1 provides naive_search.cu, selected through the backend name
cuda-naive. It deliberately uses one CUDA thread per complete
query/database-vector dot product and leaves Top-K selection on the CPU.

Milestone M3 provides block_search.cu, selected through the backend name
cuda-block. It maps one 256-thread CUDA block to one query/database-vector
pair, distributes dimensions across threads, and explicitly reduces partial
dot products through shared memory. It preserves the same host pipeline and
CPU Top-K as cuda-naive.

Milestone M4 provides warp_search.cu, selected through the backend name
cuda-warp. It keeps a 256-thread block but maps each of its eight warps to
one query/database-vector pair. Each warp distributes adjacent dimensions
across its 32 lanes and reduces the partial dot product with synchronized
shuffle operations, without explicit shared-memory reduction state or
__syncthreads(). It preserves the same host pipeline and CPU Top-K.

Later CUDA implementations must remain separate backends so the M1 kernel can
continue to serve as a correctness and performance baseline.


Milestone M5 adds resident_search.cu, selected as cuda-warp-resident. It
implements an explicit ResidentSearchBackend lifecycle:

~~~text
prepare_database -> search many query batches -> reload_database/clear_database
~~~

The database device allocation and one synchronous H2D upload persist across
search calls. Query and score buffers, score D2H, CPU Top-K, and the exact M4
warp-per-vector kernel remain per-search behavior. cuda-warp is not changed;
it remains the stateless comparison backend.

The resident implementation reports database preparation separately from warm
query stages and keeps CUDA resources under RAII with checked CUDA operations.
