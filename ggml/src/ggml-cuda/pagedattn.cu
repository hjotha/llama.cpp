#include "pagedattn.cuh"

#include "cpy-utils.cuh"
#include "dequantize.cuh"

static constexpr int PAGED_ATTN_CONTEXT_CTAS = 64;
static constexpr int PAGED_ATTN_PARALLEL_MAX_TOKENS = 4;

__device__ __forceinline__ float paged_warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
#if defined(GGML_USE_HIP)
        val += __shfl_down_sync(0xffffffffffffffffULL, val, offset, 32);
#else
        val += __shfl_down_sync(0xffffffffu, val, offset, 32);
#endif
    }
    return val;
}

__device__ __forceinline__ float paged_warp_reduce_max(float val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
#if defined(GGML_USE_HIP)
        val = fmaxf(val, __shfl_down_sync(0xffffffffffffffffULL, val, offset, 32));
#else
        val = fmaxf(val, __shfl_down_sync(0xffffffffu, val, offset, 32));
#endif
    }
    return val;
}

__device__ __forceinline__ float paged_block_reduce_sum(float val, float * smem, int tid, int n_threads) {
    const int lane    = tid & 31;
    const int warp_id = tid >> 5;
    const int n_warps = (n_threads + 31) >> 5;
    val = paged_warp_reduce_sum(val);
    if (lane == 0) {
        smem[warp_id] = val;
    }
    __syncthreads();
    float reduced = tid < n_warps ? smem[tid] : 0.0f;
    if (warp_id == 0) {
        reduced = paged_warp_reduce_sum(reduced);
        if (lane == 0) {
            smem[0] = reduced;
        }
    }
    __syncthreads();
    return smem[0];
}

__device__ __forceinline__ float paged_block_reduce_max(float val, float * smem, int tid, int n_threads) {
    const int lane    = tid & 31;
    const int warp_id = tid >> 5;
    const int n_warps = (n_threads + 31) >> 5;
    val = paged_warp_reduce_max(val);
    if (lane == 0) {
        smem[warp_id] = val;
    }
    __syncthreads();
    float reduced = tid < n_warps ? smem[tid] : -FLT_MAX;
    if (warp_id == 0) {
        reduced = paged_warp_reduce_max(reduced);
        if (lane == 0) {
            smem[0] = reduced;
        }
    }
    __syncthreads();
    return smem[0];
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
    const int token_idx  = blockIdx.z;
    if (token_idx >= num_tokens) {
        return;
    }

    const int token_batch_idx = seq_start + token_idx;
    const int target_slot     = write_slots[token_batch_idx];
    const int block_id        = target_slot / block_size;
    const int token_in_block  = target_slot % block_size;
    const size_t k_cache_idx  = (size_t) block_id * stride_block + (size_t) head_idx * stride_head +
                                (size_t) token_in_block * stride_token + tid;
    const size_t v_cache_idx  = (size_t) block_id * stride_block + (size_t) (n_heads_kv + head_idx) * stride_head +
                                (size_t) token_in_block * stride_token + tid;
    const size_t input_off    = (size_t) token_batch_idx * n_heads_kv * head_dim + (size_t) head_idx * head_dim + tid;

    kv_cache[k_cache_idx] = __float2half(k_new[input_off]);
    kv_cache[v_cache_idx] = __float2half(v_new[input_off]);
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
    const int token_idx      = blockIdx.z;
    if (token_idx >= num_new_tokens) {
        return;
    }
    const int token_batch_idx = seq_start + token_idx;
    const int ctx_len         = context_lens[seq_idx];
    const int q_pos           = (ctx_len - num_new_tokens) + token_idx;
    const size_t q_offset     = (size_t) token_batch_idx * n_heads * head_dim + (size_t) head_idx * head_dim;
    const float q_val        = q[q_offset + tid] * scale;
    float qk_max  = -FLT_MAX;
    float exp_sum = 0.0f;
    float acc     = 0.0f;

    for (int token = 0; token <= q_pos; ++token) {
        const int bid            = token / block_size;
        const int token_in_block = token % block_size;
        const int physical_block = block_table[seq_idx * max_blocks + bid];
        const size_t k_idx       = (size_t) tid + (size_t) token_in_block * stride_token +
                                   (size_t) kv_head_idx * stride_head + (size_t) physical_block * stride_block;
        const size_t v_idx       = (size_t) tid + (size_t) token_in_block * stride_token +
                                   (size_t) (n_heads_kv + kv_head_idx) * stride_head +
                                   (size_t) physical_block * stride_block;
        const float qk           = paged_block_reduce_sum(q_val * __half2float(kv_cache[k_idx]), smem, tid, head_dim);
        const float qk_max_new   = fmaxf(qk_max, qk);
        const float exp_old      = __expf(qk_max - qk_max_new);
        const float exp_new      = __expf(qk - qk_max_new);
        exp_sum = exp_sum * exp_old + exp_new;
        acc     = acc * exp_old + exp_new * __half2float(kv_cache[v_idx]);
        qk_max  = qk_max_new;
    }

    const int out_idx = (size_t) token_batch_idx * n_heads * head_dim + (size_t) head_idx * head_dim + tid;
    out[out_idx] = acc / (exp_sum + 1e-6f);
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

template<typename block_t, bool quantized>
static __device__ __forceinline__ float paged_cache_value(
        const void * cache, size_t row_offset, int index) {
    if constexpr (quantized) {
        return paged_dequantize_value(reinterpret_cast<const block_t *>(
            static_cast<const char *>(cache) + row_offset), index);
    } else {
        return __half2float(static_cast<const half *>(cache)[row_offset + index]);
    }
}

template<typename block_t, bool quantized>
static __global__ void paged_attention_score_kernel(
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
        const size_t max_context,
        float * __restrict__ scores) {
    extern __shared__ float q_shared[];

    const int head_idx   = blockIdx.x;
    const int seq_idx    = blockIdx.y;
    const int context_cta = blockIdx.z % PAGED_ATTN_CONTEXT_CTAS;
    const int token_idx   = blockIdx.z / PAGED_ATTN_CONTEXT_CTAS;
    const int tid         = threadIdx.x;
    const int lane        = tid & 31;
    const int warp_id     = tid >> 5;
    const int head_dim    = blockDim.x;
    const int n_warps     = head_dim >> 5;
    const int n_heads     = gridDim.x;
    const int kv_head_idx = head_idx / (n_heads / n_heads_kv);
    const int seq_start   = batch_offsets[seq_idx];
    const int num_tokens  = batch_lens[seq_idx];
    if (token_idx >= num_tokens) {
        return;
    }

    const int token_batch_idx = seq_start + token_idx;
    const int q_pos = context_lens[seq_idx] - num_tokens + token_idx;
    const size_t q_offset = (size_t) token_batch_idx * n_heads * head_dim +
                            (size_t) head_idx * head_dim;
    q_shared[tid] = q[q_offset + tid] * scale;
    __syncthreads();

    float * score_row = scores + ((size_t) token_batch_idx * n_heads + head_idx) * max_context;
    for (int token = context_cta * n_warps + warp_id;
         token <= q_pos;
         token += PAGED_ATTN_CONTEXT_CTAS * n_warps) {
        const int bid            = token / block_size;
        const int token_in_block = token % block_size;
        const int physical_block = block_table[seq_idx * max_blocks + bid];
        const size_t k_offset = (size_t) physical_block * stride_block +
                                (size_t) kv_head_idx * stride_head +
                                (size_t) token_in_block * stride_token;
        float qk = 0.0f;
        for (int dim = lane; dim < head_dim; dim += 32) {
            qk += q_shared[dim] * paged_cache_value<block_t, quantized>(kv_cache, k_offset, dim);
        }
        qk = paged_warp_reduce_sum(qk);
        if (lane == 0) {
            score_row[token] = qk;
        }
    }
}

template<typename block_t, bool quantized>
static __global__ void paged_attention_value_kernel(
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
        const size_t max_context,
        float * __restrict__ scores,
        float * __restrict__ out) {
    extern __shared__ float smem[];

    const int head_idx   = blockIdx.x;
    const int seq_idx    = blockIdx.y;
    const int token_idx  = blockIdx.z;
    const int tid        = threadIdx.x;
    const int n_heads    = gridDim.x;
    const int head_dim   = blockDim.x;
    const int kv_head_idx = head_idx / (n_heads / n_heads_kv);
    const int seq_start   = batch_offsets[seq_idx];
    const int num_tokens  = batch_lens[seq_idx];
    if (token_idx >= num_tokens) {
        return;
    }

    const int token_batch_idx = seq_start + token_idx;
    const int q_pos = context_lens[seq_idx] - num_tokens + token_idx;
    float * score_row = scores + ((size_t) token_batch_idx * n_heads + head_idx) * max_context;

    float local_max = -FLT_MAX;
    for (int token = tid; token <= q_pos; token += head_dim) {
        local_max = fmaxf(local_max, score_row[token]);
    }
    const float qk_max = paged_block_reduce_max(local_max, smem, tid, head_dim);

    float local_sum = 0.0f;
    for (int token = tid; token <= q_pos; token += head_dim) {
        const float weight = __expf(score_row[token] - qk_max);
        score_row[token] = weight;
        local_sum += weight;
    }
    const float exp_sum = paged_block_reduce_sum(local_sum, smem, tid, head_dim);

    float acc = 0.0f;
    for (int token = 0; token <= q_pos; ++token) {
        const int bid            = token / block_size;
        const int token_in_block = token % block_size;
        const int physical_block = block_table[seq_idx * max_blocks + bid];
        const size_t v_offset = (size_t) physical_block * stride_block +
                                (size_t) (n_heads_kv + kv_head_idx) * stride_head +
                                (size_t) token_in_block * stride_token;
        acc += score_row[token] * paged_cache_value<block_t, quantized>(kv_cache, v_offset, tid);
    }

    const size_t out_idx = (size_t) token_batch_idx * n_heads * head_dim +
                           (size_t) head_idx * head_dim + tid;
    out[out_idx] = acc / (exp_sum + 1e-6f);
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
    const int token_idx  = blockIdx.z;
    if (token_idx >= num_tokens) {
        return;
    }

    const int token_batch_idx = seq_start + token_idx;
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
    const int token_idx      = blockIdx.z;
    if (token_idx >= num_new_tokens) {
        return;
    }
    const int token_batch_idx = seq_start + token_idx;
    const int ctx_len         = context_lens[seq_idx];
    const int q_pos           = (ctx_len - num_new_tokens) + token_idx;
    const size_t q_offset     = (size_t) token_batch_idx * n_heads * head_dim + (size_t) head_idx * head_dim;
    const float q_val        = q[q_offset + tid] * scale;
    float qk_max  = -FLT_MAX;
    float exp_sum = 0.0f;
    float acc     = 0.0f;

    for (int token = 0; token <= q_pos; ++token) {
        const int bid            = token / block_size;
        const int token_in_block = token % block_size;
        const int physical_block = block_table[seq_idx * max_blocks + bid];
        const size_t k_offset    = (size_t) physical_block * stride_block + (size_t) kv_head_idx * stride_head +
                                   (size_t) token_in_block * stride_token;
        const size_t v_offset    = (size_t) physical_block * stride_block +
                                   (size_t) (n_heads_kv + kv_head_idx) * stride_head +
                                   (size_t) token_in_block * stride_token;
        const block_t * k_row    = reinterpret_cast<const block_t *>(cache + k_offset);
        const block_t * v_row    = reinterpret_cast<const block_t *>(cache + v_offset);
        const float qk           = paged_block_reduce_sum(q_val * paged_dequantize_value(k_row, tid), smem, tid, head_dim);
        const float qk_max_new   = fmaxf(qk_max, qk);
        const float exp_old      = __expf(qk_max - qk_max_new);
        const float exp_new      = __expf(qk - qk_max_new);
        exp_sum = exp_sum * exp_old + exp_new;
        acc     = acc * exp_old + exp_new * paged_dequantize_value(v_row, tid);
        qk_max  = qk_max_new;
    }

    const size_t out_idx = (size_t) token_batch_idx * n_heads * head_dim +
                           (size_t) head_idx * head_dim + tid;
    out[out_idx] = acc / (exp_sum + 1e-6f);
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
    const int n_tokens        = q->ne[2];

    GGML_ASSERT(head_dim % QK == 0 && "paged quantized KV head_dim must align to its quantization block size");
    GGML_ASSERT(head_dim % 32 == 0 && head_dim <= 1024);
    GGML_ASSERT(n_heads != 0 && n_heads_kv != 0 && n_heads % n_heads_kv == 0);
    GGML_ASSERT(n_tokens <= 65535 && "paged CUDA token batch exceeds grid.z limit");

    const size_t stride_token = kv_cache->nb[1];
    const size_t stride_head  = kv_cache->nb[2];
    const size_t stride_block = kv_cache->nb[3];
    const size_t write_smem   = (size_t) head_dim * sizeof(float);

    paged_attention_quant_write_kernel<block_t, QK><<<dim3(n_heads_kv, n_seq, n_tokens), dim3(head_dim), write_smem, ctx.stream()>>>(
        (const float *) k_new->data, (const float *) v_new->data, kv_cache->data,
        (const int *) write_slots->data, (const int *) batch_offsets->data, (const int *) batch_lens->data,
        stride_token, stride_head, stride_block, n_heads_kv, block_size);

    const size_t reduce_smem = (size_t) ((head_dim + 31) / 32) * sizeof(float);
    if (n_tokens <= PAGED_ATTN_PARALLEL_MAX_TOKENS) {
        const size_t max_context = (size_t) max_blocks * block_size;
        ggml_cuda_pool_alloc<float> scores(ctx.pool(), (size_t) n_tokens * n_heads * max_context);
        paged_attention_score_kernel<block_t, true><<<
            dim3(n_heads, n_seq, n_tokens * PAGED_ATTN_CONTEXT_CTAS), dim3(head_dim),
            (size_t) head_dim * sizeof(float), ctx.stream()>>>(
                (const float *) q->data, kv_cache->data, (const int *) block_table->data,
                (const int *) context_lens->data, (const int *) batch_offsets->data, (const int *) batch_lens->data,
                stride_token, stride_head, stride_block, n_heads_kv, block_size, max_blocks, scale,
                max_context, scores.ptr);
        paged_attention_value_kernel<block_t, true><<<
            dim3(n_heads, n_seq, n_tokens), dim3(head_dim), reduce_smem, ctx.stream()>>>(
                kv_cache->data, (const int *) block_table->data, (const int *) context_lens->data,
                (const int *) batch_offsets->data, (const int *) batch_lens->data,
                stride_token, stride_head, stride_block, n_heads_kv, block_size, max_blocks,
                max_context, scores.ptr, (float *) dst->data);
    } else {
        paged_attention_quant_decode_kernel<block_t, QK><<<
            dim3(n_heads, n_seq, n_tokens), dim3(head_dim), reduce_smem, ctx.stream()>>>(
                (const float *) q->data, kv_cache->data, (const int *) block_table->data,
                (const int *) context_lens->data, (const int *) batch_offsets->data, (const int *) batch_lens->data,
                stride_token, stride_head, stride_block, n_heads_kv, block_size, max_blocks, scale,
                (float *) dst->data);
    }
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
    const int n_tokens   = q->ne[2];

    GGML_ASSERT(n_heads != 0 && "n_head cannot be 0.");
    GGML_ASSERT(n_heads_kv != 0 && "n_heads_kv cannot be 0.");
    GGML_ASSERT(head_dim <= 1024 && "head_dim exceeds maximum supported (1024)");
    GGML_ASSERT(head_dim % 32 == 0 && "paged CUDA head_dim must be a multiple of 32");
    GGML_ASSERT(n_heads % n_heads_kv == 0 && "n_heads must be divisible by n_heads_kv");
    GGML_ASSERT(n_tokens <= 65535 && "paged CUDA token batch exceeds grid.z limit");

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
    dim3 grid_dims(n_heads, n_seq, n_tokens);  // one block per head, sequence, and token

    // Write kernel - Grid (n_heads_kv, n_seq), Block (head_dim)
    paged_attention_write_kernel<<<dim3(n_heads_kv, n_seq, n_tokens), dim3(head_dim), 0, ctx.stream()>>>(
        (const float *) k_new->data, (const float *) v_new->data, (half *) kv_cache->data,
        (const int *) write_slots->data, (const int *) batch_offsets->data, (const int *) batch_lens->data,
        stride_token, stride_head, stride_block, n_heads_kv, block_size);

    const size_t reduce_smem = (size_t) ((head_dim + 31) / 32) * sizeof(float);
    if (n_tokens <= PAGED_ATTN_PARALLEL_MAX_TOKENS) {
        const size_t max_context = (size_t) max_blocks * block_size;
        ggml_cuda_pool_alloc<float> scores(ctx.pool(), (size_t) n_tokens * n_heads * max_context);
        paged_attention_score_kernel<half, false><<<
            dim3(n_heads, n_seq, n_tokens * PAGED_ATTN_CONTEXT_CTAS), block_dims,
            (size_t) head_dim * sizeof(float), ctx.stream()>>>(
                (const float *) q->data, kv_cache->data, (const int *) block_table->data,
                (const int *) context_lens->data, (const int *) batch_offsets->data, (const int *) batch_lens->data,
                stride_token, stride_head, stride_block, n_heads_kv, block_size, max_blocks, scale,
                max_context, scores.ptr);
        paged_attention_value_kernel<half, false><<<grid_dims, block_dims, reduce_smem, ctx.stream()>>>(
            kv_cache->data, (const int *) block_table->data, (const int *) context_lens->data,
            (const int *) batch_offsets->data, (const int *) batch_lens->data,
            stride_token, stride_head, stride_block, n_heads_kv, block_size, max_blocks,
            max_context, scores.ptr, (float *) dst->data);
    } else {
        paged_attention_decode_kernel<<<grid_dims, block_dims, reduce_smem, ctx.stream()>>>(
            (const float *) q->data, (const half *) kv_cache->data, (const int *) block_table->data,
            (const int *) context_lens->data, (const int *) batch_offsets->data, (const int *) batch_lens->data,
            stride_token, stride_head, stride_block, n_heads_kv, block_size, max_blocks, scale,
            (float *) dst->data);
    }
}
