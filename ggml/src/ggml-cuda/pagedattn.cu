#include "pagedattn.cuh"

#include "cpy-utils.cuh"
#include "dequantize.cuh"

__device__ __forceinline__ float block_reduce_sum_full(float val, float * __restrict__ smem, int tid, int head_dim) {
    const int lane    = tid & 31;
    const int warp_id = tid >> 5;
    const int n_warps = (head_dim + 31) >> 5;

    // warp-level reduce
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffffu, val, offset);
    }
    // Each warp's lane 0 depositits partial sum into shared memory
    if (lane == 0) {
        smem[warp_id] = val;
    }
    __syncthreads();

    // First warp reduces the per-warp partial
    float warp_val = (tid < n_warps) ? smem[tid] : 0.0f;
    if (warp_id == 0) {
        for (int offset = 16; offset > 0; offset >>= 1) {
            warp_val += __shfl_down_sync(0xffffffffu, warp_val, offset);
        }
        if (lane == 0) {
            smem[0] = warp_val;
        }
    }
    __syncthreads();
    return smem[0];  // this will be identical in every thread
}

__global__ void paged_attention_write_kernel(const float * __restrict__ k_new,  // [batch_size, n_heads_kv, head_dim]
                                             const float * __restrict__ v_new,  // [batch_size, n_heads_kv, head_dim]
                                             half * __restrict__ kv_cache,      // The paged cache
                                             const int * __restrict__ write_slots,  // Global slot index for each token
                                             const int * __restrict__ batch_offsets,
                                             const int * __restrict__ batch_lens,
                                             const size_t stride_token,  // Elements between tokens in a block (nb1)
                                             const size_t stride_head,   // Elements between heads (nb2)
                                             const size_t stride_block,  // Elements between physical blocks (nb3)
                                             const int    n_heads_kv,
                                             const int    block_size) {
    const int head_idx = blockIdx.x;   // 0 to n_heads_kv - 1
    const int seq_idx  = blockIdx.y;
    const int tid      = threadIdx.x;  // 0 to head_dim - 1
    const int head_dim = blockDim.x;

    const int seq_start  = batch_offsets[seq_idx];
    const int num_tokens = batch_lens[seq_idx];

    for (int i = 0; i < num_tokens; ++i) {
        const int token_batch_idx = seq_start + i;
        const int target_slot     = write_slots[token_batch_idx];

        // Map slot to block and internal offset
        const int block_id       = target_slot / block_size;
        const int token_in_block = target_slot % block_size;

        // K is at head_idx, V is at n_heads_kv + head_idx
        const size_t k_cache_idx = (size_t) block_id * stride_block + (size_t) head_idx * stride_head +
                                   (size_t) token_in_block * stride_token + tid;
        const size_t v_cache_idx = (size_t) block_id * stride_block + (size_t) (n_heads_kv + head_idx) * stride_head +
                                   (size_t) token_in_block * stride_token + tid;

        // Input offset: [token][head][dim]
        const size_t input_off = (size_t) token_batch_idx * n_heads_kv * head_dim + (size_t) head_idx * head_dim + tid;

        kv_cache[k_cache_idx] = __float2half(k_new[input_off]);
        kv_cache[v_cache_idx] = __float2half(v_new[input_off]);
    }
}

__global__ void paged_attention_decode_kernel(const float * __restrict__ q,
                                              const half * __restrict__ kv_cache,
                                              const int * __restrict__ block_table,
                                              const int * __restrict__ context_lens,
                                              const int * __restrict__ batch_offsets,
                                              const int * __restrict__ batch_lens,
                                              const size_t stride_token,
                                              const size_t stride_head,
                                              const size_t stride_block,
                                              const int    n_heads_kv,
                                              const int    block_size,
                                              const int    max_blocks,
                                              const float  scale,
                                              float * __restrict__ out) {
    extern __shared__ float smem[];

    const int head_idx = blockIdx.x;
    const int seq_idx  = blockIdx.y;
    const int tid      = threadIdx.x;

    const int n_heads  = gridDim.x;
    const int head_dim = blockDim.x;

    const int kv_head_idx = head_idx / (n_heads / n_heads_kv);

    const int seq_start      = batch_offsets[seq_idx];
    const int num_new_tokens = batch_lens[seq_idx];

    for (int i = 0; i < num_new_tokens; i++) {
        const int token_batch_idx = seq_start + i;

        float q_val = q[(size_t) token_batch_idx * n_heads * head_dim + (size_t) head_idx * head_dim + tid] * scale;

        float qk_max  = -FLT_MAX;
        float exp_sum = 0.0f;
        float acc     = 0.0f;

        const int ctx_len    = context_lens[seq_idx];
        const int q_pos      = (ctx_len - num_new_tokens) + i;
        const int num_blocks = (q_pos / block_size) + 1;

        for (int bid = 0; bid < num_blocks; bid++) {
            const int physical_block = block_table[seq_idx * max_blocks + bid];
            const int start_token    = bid * block_size;
            const int end_token      = min(start_token + block_size, q_pos + 1);

            for (int token = start_token; token < end_token; ++token) {
                const int token_in_block = token % block_size;

                const size_t k_idx =
                    tid + token_in_block * stride_token + kv_head_idx * stride_head + physical_block * stride_block;

                const size_t v_idx = tid + token_in_block * stride_token + (n_heads_kv + kv_head_idx) * stride_head +
                                     physical_block * stride_block;

                float k_val = __half2float(kv_cache[k_idx]);
                float v_val = __half2float(kv_cache[v_idx]);

                // Calculate full dot product and return the same scalar in every thread
                const float qk = block_reduce_sum_full(q_val * k_val, smem, tid, head_dim);

                // Online softmax update
                const float qk_max_new = fmaxf(qk_max, qk);
                const float exp_old    = __expf(qk_max - qk_max_new);
                const float exp_new    = __expf(qk - qk_max_new);

                exp_sum = exp_sum * exp_old + exp_new;
                acc     = acc * exp_old + exp_new * v_val;
                qk_max  = qk_max_new;
            }
        }

        const int out_idx = (size_t) token_batch_idx * n_heads * head_dim + (size_t) head_idx * head_dim + tid;

        out[out_idx] = acc / (exp_sum + 1e-6f);
    }
}

static __device__ __forceinline__ void paged_quantize_block(const float * x, block_q4_0 * dst) {
    quantize_f32_q4_0_block(x, dst);
}

static __device__ __forceinline__ void paged_quantize_block(const float * x, block_q8_0 * dst) {
    quantize_f32_q8_0_block(x, dst);
}

static __device__ __forceinline__ float paged_dequantize_value(const block_q4_0 * row, int index) {
    const int ib  = index / QK4_0;
    const int iqs = index % QK4_0;
    float2 values;
    dequantize_q4_0(row, ib, iqs % (QK4_0 / 2), values);
    return iqs < QK4_0 / 2 ? values.x : values.y;
}

static __device__ __forceinline__ float paged_dequantize_value(const block_q8_0 * row, int index) {
    const int ib  = index / QK8_0;
    const int iqs = index % QK8_0;
    float2 values;
    dequantize_q8_0(row, ib, iqs & ~1, values);
    return (iqs & 1) ? values.y : values.x;
}

template<typename block_t, int QK>
static __device__ __forceinline__ void paged_quantize_row(const float * values, block_t * row, int head_dim) {
    for (int ib = 0; ib < head_dim / QK; ++ib) {
        paged_quantize_block(values + ib * QK, row + ib);
    }
}

template<typename block_t, int QK>
static __global__ void paged_attention_quant_write_kernel(
        const float * __restrict__ k_new,
        const float * __restrict__ v_new,
        void * __restrict__ kv_cache,
        const int * __restrict__ write_slots,
        const int * __restrict__ batch_offsets,
        const int * __restrict__ batch_lens,
        const size_t stride_token,
        const size_t stride_head,
        const size_t stride_block,
        const int n_heads_kv,
        const int block_size) {
    extern __shared__ float values[];

    const int head_idx = blockIdx.x;
    const int seq_idx  = blockIdx.y;
    const int tid      = threadIdx.x;
    const int head_dim = blockDim.x;

    const int seq_start  = batch_offsets[seq_idx];
    const int num_tokens = batch_lens[seq_idx];
    char * cache         = static_cast<char *>(kv_cache);

    for (int i = 0; i < num_tokens; ++i) {
        const int token_batch_idx = seq_start + i;
        const int target_slot     = write_slots[token_batch_idx];
        const int block_id        = target_slot / block_size;
        const int token_in_block  = target_slot % block_size;
        const size_t input_off    = (size_t) token_batch_idx * n_heads_kv * head_dim +
                                    (size_t) head_idx * head_dim;

        values[tid] = k_new[input_off + tid];
        __syncthreads();
        if (tid == 0) {
            // ponytail: one thread quantizes each row; parallelize block quantization if write throughput matters.
            const size_t row_offset = (size_t) block_id * stride_block + (size_t) head_idx * stride_head +
                                      (size_t) token_in_block * stride_token;
            paged_quantize_row<block_t, QK>(values, reinterpret_cast<block_t *>(cache + row_offset), head_dim);
        }
        __syncthreads();

        values[tid] = v_new[input_off + tid];
        __syncthreads();
        if (tid == 0) {
            const size_t row_offset = (size_t) block_id * stride_block +
                                      (size_t) (n_heads_kv + head_idx) * stride_head +
                                      (size_t) token_in_block * stride_token;
            paged_quantize_row<block_t, QK>(values, reinterpret_cast<block_t *>(cache + row_offset), head_dim);
        }
        __syncthreads();
    }
}

template<typename block_t, int QK>
static __global__ void paged_attention_quant_decode_kernel(
        const float * __restrict__ q,
        const void * __restrict__ kv_cache,
        const int * __restrict__ block_table,
        const int * __restrict__ context_lens,
        const int * __restrict__ batch_offsets,
        const int * __restrict__ batch_lens,
        const size_t stride_token,
        const size_t stride_head,
        const size_t stride_block,
        const int n_heads_kv,
        const int block_size,
        const int max_blocks,
        const float scale,
        float * __restrict__ out) {
    extern __shared__ float smem[];

    const int head_idx = blockIdx.x;
    const int seq_idx  = blockIdx.y;
    const int tid      = threadIdx.x;
    const int n_heads  = gridDim.x;
    const int head_dim = blockDim.x;
    const int kv_head_idx = head_idx / (n_heads / n_heads_kv);
    const char * cache = static_cast<const char *>(kv_cache);

    const int seq_start      = batch_offsets[seq_idx];
    const int num_new_tokens = batch_lens[seq_idx];

    for (int i = 0; i < num_new_tokens; ++i) {
        const int token_batch_idx = seq_start + i;
        const float q_val = q[(size_t) token_batch_idx * n_heads * head_dim +
                              (size_t) head_idx * head_dim + tid] * scale;

        float qk_max  = -FLT_MAX;
        float exp_sum = 0.0f;
        float acc     = 0.0f;

        const int ctx_len    = context_lens[seq_idx];
        const int q_pos      = (ctx_len - num_new_tokens) + i;
        const int num_blocks = (q_pos / block_size) + 1;

        for (int bid = 0; bid < num_blocks; ++bid) {
            const int physical_block = block_table[seq_idx * max_blocks + bid];
            const int start_token    = bid * block_size;
            const int end_token      = min(start_token + block_size, q_pos + 1);

            for (int token = start_token; token < end_token; ++token) {
                const int token_in_block = token % block_size;
                const size_t k_offset = (size_t) physical_block * stride_block +
                                        (size_t) kv_head_idx * stride_head +
                                        (size_t) token_in_block * stride_token;
                const size_t v_offset = (size_t) physical_block * stride_block +
                                        (size_t) (n_heads_kv + kv_head_idx) * stride_head +
                                        (size_t) token_in_block * stride_token;
                const block_t * k_row = reinterpret_cast<const block_t *>(cache + k_offset);
                const block_t * v_row = reinterpret_cast<const block_t *>(cache + v_offset);

                const float k_val = paged_dequantize_value(k_row, tid);
                const float v_val = paged_dequantize_value(v_row, tid);
                const float qk = block_reduce_sum_full(q_val * k_val, smem, tid, head_dim);

                const float qk_max_new = fmaxf(qk_max, qk);
                const float exp_old    = __expf(qk_max - qk_max_new);
                const float exp_new    = __expf(qk - qk_max_new);

                exp_sum = exp_sum * exp_old + exp_new;
                acc     = acc * exp_old + exp_new * v_val;
                qk_max  = qk_max_new;
            }
        }

        const size_t out_idx = (size_t) token_batch_idx * n_heads * head_dim +
                               (size_t) head_idx * head_dim + tid;
        out[out_idx] = acc / (exp_sum + 1e-6f);
    }
}

template<typename block_t, int QK>
static void ggml_cuda_op_paged_attn_quantized(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * q             = dst->src[0];
    const ggml_tensor * k_new         = dst->src[1];
    const ggml_tensor * v_new         = dst->src[2];
    const ggml_tensor * kv_cache      = dst->src[3];
    const ggml_tensor * block_table   = dst->src[5];
    const ggml_tensor * write_slots   = dst->src[6];
    const ggml_tensor * context_lens  = dst->src[7];
    const ggml_tensor * batch_offsets = dst->src[8];
    const ggml_tensor * batch_lens    = dst->src[9];

    const float * op_params_f = (const float *) (dst->op_params);
    const float scale         = op_params_f[0];
    const int block_size      = ((const int32_t *) (op_params_f + 1))[0];
    const int max_blocks      = ((const int32_t *) (op_params_f + 2))[0];
    const int head_dim        = q->ne[0];
    const int n_heads         = q->ne[1];
    const int n_seq           = batch_lens->ne[0];
    const int n_heads_kv      = k_new->ne[1];

    GGML_ASSERT(head_dim % QK == 0 && "paged quantized KV head_dim must align to its quantization block size");
    GGML_ASSERT(n_heads != 0 && n_heads_kv != 0 && n_heads % n_heads_kv == 0);

    const size_t stride_token = kv_cache->nb[1];
    const size_t stride_head  = kv_cache->nb[2];
    const size_t stride_block = kv_cache->nb[3];
    const size_t write_smem   = (size_t) head_dim * sizeof(float);

    paged_attention_quant_write_kernel<block_t, QK><<<dim3(n_heads_kv, n_seq), dim3(head_dim), write_smem, ctx.stream()>>>(
        (const float *) k_new->data, (const float *) v_new->data, kv_cache->data,
        (const int *) write_slots->data, (const int *) batch_offsets->data, (const int *) batch_lens->data,
        stride_token, stride_head, stride_block, n_heads_kv, block_size);

    const size_t n_warps    = ((size_t) head_dim + 31) / 32;
    const size_t smem_bytes = n_warps * sizeof(float);
    paged_attention_quant_decode_kernel<block_t, QK><<<dim3(n_heads, n_seq), dim3(head_dim), smem_bytes, ctx.stream()>>>(
        (const float *) q->data, kv_cache->data, (const int *) block_table->data,
        (const int *) context_lens->data, (const int *) batch_offsets->data, (const int *) batch_lens->data,
        stride_token, stride_head, stride_block, n_heads_kv, block_size, max_blocks, scale, (float *) dst->data);
}

void ggml_cuda_op_paged_attn(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * q             = dst->src[0];
    const ggml_tensor * k_new         = dst->src[1];
    const ggml_tensor * v_new         = dst->src[2];
    const ggml_tensor * kv_cache      = dst->src[3];  // KV interleaved layout
    const ggml_tensor * block_table   = dst->src[5];
    const ggml_tensor * write_slots   = dst->src[6];
    const ggml_tensor * context_lens  = dst->src[7];
    const ggml_tensor * batch_offsets = dst->src[8];
    const ggml_tensor * batch_lens    = dst->src[9];

    const float * op_params_f = (const float *) (dst->op_params);
    const float   scale       = op_params_f[0];
    const int     block_size  = ((const int32_t *) (op_params_f + 1))[0];
    const int     max_blocks  = ((const int32_t *) (op_params_f + 2))[0];

    const int head_dim   = q->ne[0];
    const int n_heads    = q->ne[1];
    const int n_seq      = batch_lens->ne[0];
    const int n_heads_kv = k_new->ne[1];

    GGML_ASSERT(n_heads != 0 && "n_head cannot be 0.");
    GGML_ASSERT(n_heads_kv != 0 && "n_heads_kv cannot be 0.");
    GGML_ASSERT(head_dim <= 1024 && "head_dim exceeds maximum supported (1024)");
    GGML_ASSERT(n_heads % n_heads_kv == 0 && "n_heads must be divisible by n_heads_kv");

    if (kv_cache->type == GGML_TYPE_Q4_0) {
        ggml_cuda_op_paged_attn_quantized<block_q4_0, QK4_0>(ctx, dst);
        return;
    }
    if (kv_cache->type == GGML_TYPE_Q8_0) {
        ggml_cuda_op_paged_attn_quantized<block_q8_0, QK8_0>(ctx, dst);
        return;
    }
    GGML_ASSERT(kv_cache->type == GGML_TYPE_F16 && "unsupported paged KV type for CUDA");

    // Extracting strides
    const size_t stride_token = kv_cache->nb[1] / sizeof(half);
    const size_t stride_head  = kv_cache->nb[2] / sizeof(half);
    const size_t stride_block = kv_cache->nb[3] / sizeof(half);

    dim3 block_dims(head_dim);       // one thread per dimension of head
    dim3 grid_dims(n_heads, n_seq);  // one block per head per sequence

    // Write kernel - Grid (n_heads_kv, n_seq), Block (head_dim)
    paged_attention_write_kernel<<<dim3(n_heads_kv, n_seq), dim3(head_dim), 0, ctx.stream()>>>(
        (const float *) k_new->data, (const float *) v_new->data, (half *) kv_cache->data,
        (const int *) write_slots->data, (const int *) batch_offsets->data, (const int *) batch_lens->data,
        stride_token, stride_head, stride_block, n_heads_kv, block_size);

    // Shared memory
    const size_t n_warps    = ((size_t) head_dim + 31) / 32;
    const size_t smem_bytes = n_warps * sizeof(float);

    // Manually request extended shared memory if needed (>48 KB)
    // https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html
    if (smem_bytes > 48 * 1024) {
        GGML_ASSERT(smem_bytes <= 96 * 1024 && "smem exceeds 96KB limit");
        CUDA_CHECK(cudaFuncSetAttribute(paged_attention_decode_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        (int) smem_bytes));
    }

    // Read kernel - Grid (n_heads_kv, n_seq), Block (head_dim)
    paged_attention_decode_kernel<<<dim3(n_heads, n_seq), dim3(head_dim), smem_bytes, ctx.stream()>>>(
        (const float *) q->data, (const half *) kv_cache->data, (const int *) block_table->data,
        (const int *) context_lens->data, (const int *) batch_offsets->data, (const int *) batch_lens->data,
        stride_token, stride_head, stride_block, n_heads_kv, block_size, max_blocks, scale, (float *) dst->data);
}
