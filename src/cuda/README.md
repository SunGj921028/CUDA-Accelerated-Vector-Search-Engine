# CUDA backends

Milestone M1 provides `naive_search.cu`, selected through the backend name
`cuda-naive`. It deliberately uses one CUDA thread per complete
query/database-vector dot product and leaves Top-K selection on the CPU.

Later CUDA implementations must remain separate backends so the M1 kernel can
continue to serve as a correctness and performance baseline.
