#include "llama-kv-cache-paged.h"

#include "llama-impl.h"

#include <algorithm>
#include <numeric>

//
// llama_kv_cache_paged
//

llama_kv_cache_paged::llama_kv_cache_paged(uint32_t head_dim,
                                           uint32_t n_heads_kv,
                                           uint32_t block_size,
                                           uint32_t n_layers,
                                           uint32_t n_ubatch,
                                           uint32_t n_seq_max,
                                           const layer_filter_cb & filter_attn) :
    kv_type(GGML_TYPE_F16),
    head_dim(head_dim),
    n_heads_kv(n_heads_kv),
    block_size(block_size),
    n_layers(n_layers),
    n_ubatch(n_ubatch),
    n_seq_max(n_seq_max),
    filter_attn(filter_attn),
    num_gpu_blocks(0),
    num_cpu_blocks(0),
    initial_num_gpu_blocks(0),
    gpu_watermark_num_blocks(0),
    block_bytes(0),
    allow_dynamic_spill(false),
    cpu_backend(nullptr) {}

void llama_kv_cache_paged::init(const std::vector<ggml_backend_t> & layer_backends,
                                const std::vector<ggml_backend_t> & candidate_backends,
                                ggml_backend_t backend_cpu,
                                enum ggml_type type,
                                uint32_t       n_gpu_blocks,
                                uint32_t       n_cpu_blocks,
                                float          watermark,
                                uint32_t       initial_gpu_blocks,
                                bool           dynamic_spill) {
    GGML_ASSERT(backend_cpu && "backend_cpu is nullptr");

    GGML_ASSERT(n_gpu_blocks && "n_gpu_blocks need to be greater than 0.");
    GGML_ASSERT(n_cpu_blocks && "n_cpu_blocks need to be greater than 0.");

    LLAMA_LOG_INFO(
        "%s: initializing paged KV cache. n_gpu_blocks=%d, n_cpu_blocks=%d, block_size=%d, watermark=%0.2f, initial_gpu_blocks=%d, dynamic_spill=%d\n",
        __func__, n_gpu_blocks, n_cpu_blocks, block_size, watermark, initial_gpu_blocks, dynamic_spill);
    num_gpu_blocks = n_gpu_blocks;
    num_cpu_blocks = n_cpu_blocks;
    allow_dynamic_spill = dynamic_spill;
    initial_num_gpu_blocks = dynamic_spill ? std::max((uint32_t) 1, std::min(initial_gpu_blocks, n_gpu_blocks))
                                           : n_gpu_blocks;
    gpu_watermark_num_blocks = std::ceil(num_gpu_blocks * watermark);
    kv_type        = type;
    cpu_backend    = backend_cpu;
    GGML_ASSERT(kv_type == GGML_TYPE_F16 || kv_type == GGML_TYPE_Q4_0 || kv_type == GGML_TYPE_Q8_0);
    GGML_ASSERT(head_dim % ggml_blck_size(kv_type) == 0 && "paged KV head_dim must align to the quantization block size");
    const size_t row_bytes = ggml_row_size(kv_type, head_dim);
    block_bytes = (size_t) 2 * block_size * n_heads_kv * row_bytes;

    LLAMA_LOG_INFO(
        "%s: type=%s, bytes_per_row=%zu, bytes_per_block=%zu\n",
        __func__, ggml_type_name(kv_type), row_bytes, block_bytes);

    kv_gpu_layers.clear();
    kv_gpu_layers.resize(n_layers);
    layer_cpu_indices.assign(n_layers, -1);
    attn_layer_ids.clear();
    kv_cpu_layers.clear();
    spill_backends.clear();
    free_gpu_ids.resize(num_gpu_blocks);
    std::iota(free_gpu_ids.begin(), free_gpu_ids.end(), 0);
    gpu_block_ref_counts.assign(num_gpu_blocks, 0);
    for (ggml_backend_t backend : candidate_backends) {
        if (backend != nullptr && std::find(spill_backends.begin(), spill_backends.end(), backend) == spill_backends.end()) {
            spill_backends.push_back(backend);
        }
    }

    for (uint32_t il = 0; il < n_layers; ++il) {
        if (filter_attn && !filter_attn(il)) {
            continue;
        }

        ggml_backend_t layer_backend = il < layer_backends.size() ? layer_backends[il] : backend_cpu;
        GGML_ASSERT(layer_backend && "paged KV layer backend is nullptr");
        ggml_backend_t candidate_backend = layer_backend;
        for (ggml_backend_t backend : spill_backends) {
            if (backend != layer_backend) {
                candidate_backend = backend;
                break;
            }
        }

        auto & storage = kv_gpu_layers[il];
        storage.original_backend  = layer_backend;
        storage.candidate_backend = candidate_backend;
        storage.backend           = layer_backend;

        GGML_ASSERT(reset_layer_storage(storage, layer_backend, initial_num_gpu_blocks, 0) &&
                    "Failed to allocate paged KV layer buffer");
        LLAMA_LOG_INFO(
            "%s: layer %u backend=%s candidate=%s blocks=%u buffer=%8.2f MiB\n",
            __func__, il,
            ggml_backend_dev_name(ggml_backend_get_device(storage.backend)),
            ggml_backend_dev_name(ggml_backend_get_device(storage.candidate_backend)),
            storage.capacity,
            ggml_backend_buffer_get_size(storage.buf.get()) / 1024.0 / 1024.0);

        layer_cpu_indices[il] = kv_cpu_layers.size();
        attn_layer_ids.push_back(il);
    }

    // Set up CPU context and tensor (for swapping)
    struct ggml_init_params cpu_params;
    cpu_params.mem_size           = ggml_tensor_overhead() * 5 * attn_layer_ids.size();
    cpu_params.mem_buffer         = NULL;
    cpu_params.no_alloc           = true;
    cpu_ctx.reset(ggml_init(cpu_params));
    GGML_ASSERT(cpu_ctx && "Failed to initialize CPU KV cache context");
    for (size_t i = 0; i < attn_layer_ids.size(); ++i) {
        kv_cpu_layers.push_back(create_layer_tensor(cpu_ctx.get(), type, n_cpu_blocks));
    }

    // Allocate on the CPU backend (using pinned memory for faster PCIe transfer)
    cpu_buf.reset(ggml_backend_alloc_ctx_tensors(cpu_ctx.get(), backend_cpu));
    GGML_ASSERT(cpu_buf && "Failed to allocate CPU KV cache buffer");
    ggml_backend_buffer_clear(cpu_buf.get(), 0);
    for (uint32_t i = 0; i < kv_cpu_layers.size(); ++i) {
        GGML_ASSERT(kv_cpu_layers[i]->buffer && "CPU layer tensor has null buffer");
    }

    // Setting up our block accountant
    block_manager.init(n_gpu_blocks, n_cpu_blocks, watermark);
}

bool llama_kv_cache_paged::has_free_gpu_blocks(uint32_t num_requested_blocks, bool ignore_watermark) const {
    const size_t curr_free_gpus = free_gpu_ids.size();
    if (!ignore_watermark && curr_free_gpus < gpu_watermark_num_blocks) {
        return false;
    }
    return ignore_watermark ? num_requested_blocks <= curr_free_gpus
                            : num_requested_blocks <= (curr_free_gpus - gpu_watermark_num_blocks);
}

llama_block_ids llama_kv_cache_paged::preview_gpu_blocks(uint32_t num_blocks) const {
    if (num_blocks > free_gpu_ids.size()) {
        return {};
    }
    return llama_block_ids(free_gpu_ids.begin(), free_gpu_ids.begin() + num_blocks);
}

llama_block_ids llama_kv_cache_paged::checkout_gpu_blocks(uint32_t num_blocks) {
    llama_block_ids new_ids = preview_gpu_blocks(num_blocks);
    if (new_ids.size() != num_blocks) {
        return {};
    }

    free_gpu_ids.erase(free_gpu_ids.begin(), free_gpu_ids.begin() + num_blocks);
    for (const uint32_t id : new_ids) {
        gpu_block_ref_counts[id] += 1;
    }
    return new_ids;
}

void llama_kv_cache_paged::release_gpu_blocks(const llama_block_ids & freed_blocks) {
    for (const uint32_t id : freed_blocks) {
        GGML_ASSERT(id < gpu_block_ref_counts.size());
        GGML_ASSERT(gpu_block_ref_counts[id] > 0);
        gpu_block_ref_counts[id] -= 1;
        if (gpu_block_ref_counts[id] == 0) {
            free_gpu_ids.insert(std::lower_bound(free_gpu_ids.begin(), free_gpu_ids.end(), id), id);
        }
    }
}

uint32_t llama_kv_cache_paged::required_gpu_capacity() const {
    for (uint32_t i = num_gpu_blocks; i > 0; --i) {
        if (gpu_block_ref_counts[i - 1] > 0) {
            return i;
        }
    }
    return 0;
}

struct ggml_tensor * llama_kv_cache_paged::get_cpu_tensor(int layer_idx) const {
    GGML_ASSERT(layer_idx >= 0 && layer_idx < (int32_t) layer_cpu_indices.size());
    GGML_ASSERT(layer_cpu_indices[layer_idx] >= 0 && "layer not in CPU paged KV cache");
    return kv_cpu_layers[layer_cpu_indices[layer_idx]];
}

struct ggml_tensor * llama_kv_cache_paged::create_layer_tensor(struct ggml_context * ctx, enum ggml_type type, uint32_t num_blocks) const {
    return ggml_new_tensor_4d(ctx, type, head_dim, block_size, 2 * n_heads_kv, num_blocks);
}

bool llama_kv_cache_paged::reset_layer_storage(layer_storage & storage,
                                               ggml_backend_t  target_backend,
                                               uint32_t        num_blocks,
                                               size_t          bytes_to_copy) {
    GGML_ASSERT(target_backend && "paged KV target backend is nullptr");
    ggml_backend_t old_backend = storage.backend;
    const uint32_t old_capacity = storage.capacity;

    struct ggml_init_params params = {};
    params.mem_size   = ggml_tensor_overhead() * 5;
    params.mem_buffer = NULL;
    params.no_alloc   = true;

    ggml_context_ptr new_ctx { ggml_init(params) };
    if (!new_ctx) {
        return false;
    }

    struct ggml_tensor * new_tensor = create_layer_tensor(new_ctx.get(), kv_type, num_blocks);
    if (!new_tensor) {
        return false;
    }

    ggml_backend_buffer_ptr new_buf;
    std::vector<uint8_t> staging;
    size_t copy_bytes = 0;

    if (storage.tensor != nullptr && target_backend == storage.backend) {
        copy_bytes = std::min(bytes_to_copy, ggml_nbytes(storage.tensor));
        staging.resize(copy_bytes);
        if (copy_bytes > 0) {
            ggml_backend_synchronize(storage.backend);
            ggml_backend_tensor_get(storage.tensor, staging.data(), 0, copy_bytes);
        }

        // Avoid a transient double allocation: under WDDM it can evict otherwise fitting
        // CUDA buffers to system memory and leave decode permanently bandwidth-bound.
        storage.tensor = nullptr;
        storage.buf.reset();
        storage.ctx.reset();
        new_buf.reset(ggml_backend_alloc_ctx_tensors(new_ctx.get(), target_backend));

        if (!new_buf) {
            new_ctx.reset(ggml_init(params));
            new_tensor = new_ctx ? create_layer_tensor(new_ctx.get(), kv_type, old_capacity) : nullptr;
            new_buf.reset(new_tensor ? ggml_backend_alloc_ctx_tensors(new_ctx.get(), old_backend) : nullptr);
            GGML_ASSERT(new_buf && "failed to restore paged KV after in-place growth failure");
            ggml_backend_buffer_clear(new_buf.get(), 0);
            if (copy_bytes > 0) {
                ggml_backend_tensor_set(new_tensor, staging.data(), 0, copy_bytes);
            }
            storage.ctx      = std::move(new_ctx);
            storage.buf      = std::move(new_buf);
            storage.backend  = old_backend;
            storage.tensor   = new_tensor;
            storage.capacity = old_capacity;
            ++storage_generation;
            return false;
        }
    } else {
        new_buf.reset(ggml_backend_alloc_ctx_tensors(new_ctx.get(), target_backend));
    }
    if (!new_buf) {
        return false;
    }

    ggml_backend_buffer_clear(new_buf.get(), 0);

    if (storage.tensor != nullptr && bytes_to_copy > 0) {
        copy_bytes = std::min({ bytes_to_copy, ggml_nbytes(storage.tensor), ggml_nbytes(new_tensor) });
        if (copy_bytes > 0) {
            ggml_backend_synchronize(storage.backend);
            ggml_backend_synchronize(target_backend);
            staging.resize(copy_bytes);
            ggml_backend_tensor_get(storage.tensor, staging.data(), 0, copy_bytes);
        }
    }
    if (copy_bytes > 0) {
        ggml_backend_tensor_set(new_tensor, staging.data(), 0, copy_bytes);
    }

    storage.ctx      = std::move(new_ctx);
    storage.buf      = std::move(new_buf);
    storage.backend  = target_backend;
    storage.tensor   = new_tensor;
    storage.capacity = num_blocks;
    ++storage_generation;
    if (old_backend != nullptr) {
        LLAMA_LOG_INFO("%s: paged KV moved %s:%u -> %s:%u blocks\n", __func__,
            ggml_backend_dev_name(ggml_backend_get_device(old_backend)), old_capacity,
            ggml_backend_dev_name(ggml_backend_get_device(target_backend)), num_blocks);
    }
    return true;
}

bool llama_kv_cache_paged::ensure_physical_capacity(uint32_t required_blocks, bool rebalance) {
    if (required_blocks == 0) {
        return true;
    }

    const uint32_t growth_step = std::max((uint32_t) 1, initial_num_gpu_blocks);
    const uint32_t target_blocks = std::min(num_gpu_blocks,
        ((required_blocks + growth_step - 1) / growth_step) * growth_step);
    if (target_blocks <= initial_num_gpu_blocks && !rebalance) {
        return true;
    }

    // ponytail: the initial pool is the existing hardware calibration knob;
    // expose a separate VRAM budget only if another device needs independent tuning.
    const uint32_t original_budget_blocks = std::min(num_gpu_blocks,
        initial_num_gpu_blocks * 2);
    const size_t max_original_layers = std::min(attn_layer_ids.size(),
        (size_t) ((uint64_t) original_budget_blocks * attn_layer_ids.size() / target_blocks));

    for (size_t layer_index = 0; layer_index < attn_layer_ids.size(); ++layer_index) {
        const uint32_t il = attn_layer_ids[layer_index];
        auto & storage = kv_gpu_layers[il];
        const bool can_spill = allow_dynamic_spill && storage.candidate_backend != nullptr &&
                               storage.candidate_backend != storage.backend;
        const bool spill_first = (rebalance || target_blocks > original_budget_blocks) && can_spill &&
                                 storage.backend == storage.original_backend &&
                                 layer_index >= max_original_layers;
        if (storage.capacity >= target_blocks && !spill_first) {
            continue;
        }

        std::vector<ggml_backend_t> candidates;
        if (spill_first) {
            LLAMA_LOG_INFO("%s: spilling layer %u before growth (%zu/%zu layers remain on original backend)\n",
                __func__, il, max_original_layers, attn_layer_ids.size());
        }
        if (spill_first) {
            candidates.push_back(storage.candidate_backend);
        }
        candidates.push_back(storage.backend);
        if (can_spill && !spill_first) {
            candidates.push_back(storage.candidate_backend);
        }
        for (ggml_backend_t backend : spill_backends) {
            if (backend != nullptr && std::find(candidates.begin(), candidates.end(), backend) == candidates.end()) {
                candidates.push_back(backend);
            }
        }
        if (std::find(candidates.begin(), candidates.end(), storage.original_backend) == candidates.end()) {
            candidates.push_back(storage.original_backend);
        }

        const uint32_t new_capacity = std::max(storage.capacity, target_blocks);
        const size_t bytes_to_copy = (size_t) std::min(required_gpu_capacity(), storage.capacity) * block_bytes;
        bool grown = false;
        for (ggml_backend_t candidate : candidates) {
            if (reset_layer_storage(storage, candidate, new_capacity, bytes_to_copy)) {
                grown = true;
                break;
            }
        }
        if (!grown) {
            LLAMA_LOG_WARN("%s: unable to grow layer %u to %u blocks\n", __func__, il, new_capacity);
            return false;
        }
    }

    return true;
}

void llama_kv_cache_paged::maybe_restore_initial_storage() {
    if (!allow_dynamic_spill) {
        return;
    }

    const uint32_t required_blocks = required_gpu_capacity();
    if (required_blocks > initial_num_gpu_blocks) {
        return;
    }

    for (const uint32_t il : attn_layer_ids) {
        auto & storage = kv_gpu_layers[il];
        if (storage.backend == storage.original_backend && storage.capacity == initial_num_gpu_blocks) {
            continue;
        }

        const size_t bytes_to_copy = std::min(required_blocks, storage.capacity) * block_bytes;
        if (!reset_layer_storage(storage, storage.original_backend, initial_num_gpu_blocks, bytes_to_copy)) {
            LLAMA_LOG_WARN("%s: unable to restore layer %u to its original backend\n", __func__, il);
        }
    }
}

bool llama_kv_cache_paged::allocate(int32_t num_tokens, llama_sequence_group & group) {
    uint32_t curr_block_count     = group.block_table.size();
    uint32_t total_num_tokens     = group.n_prompt + group.n_decoded + num_tokens;
    uint32_t num_requested_blocks = std::ceil((float) total_num_tokens / block_size) - curr_block_count;
    LLAMA_LOG_DEBUG("%s: curr_block_count=%d, total_num_tokens=%d, num_requested_blocks=%d\n", __func__,
                    curr_block_count, total_num_tokens, num_requested_blocks);

    if (num_requested_blocks == 0) {
        if (!allow_dynamic_spill || num_tokens > 4) {
            return true;
        }
        maybe_restore_initial_storage();
        return ensure_physical_capacity(required_gpu_capacity(), /*rebalance=*/true);
    }

    const bool ignore_watermark = !group.block_table.empty();
    if (!has_free_gpu_blocks(num_requested_blocks, ignore_watermark)) {
        LLAMA_LOG_DEBUG("%s: insufficient GPU blocks. Requested: %d.\n", __func__, num_requested_blocks);
        return false;
    }

    llama_block_ids preview_ids = preview_gpu_blocks(num_requested_blocks);
    GGML_ASSERT(preview_ids.size() == num_requested_blocks);

    uint32_t required_blocks = required_gpu_capacity();
    if (!preview_ids.empty()) {
        required_blocks = std::max(required_blocks, *std::max_element(preview_ids.begin(), preview_ids.end()) + 1);
    }
    if (!ensure_physical_capacity(required_blocks, allow_dynamic_spill && num_tokens <= 4)) {
        return false;
    }

    llama_block_ids new_ids = checkout_gpu_blocks(num_requested_blocks);
    GGML_ASSERT(new_ids.size() == num_requested_blocks);
    concat_block_ids(group.block_table, new_ids);
    LLAMA_LOG_DEBUG("%s: successfully allocated %d.\n", __func__, num_requested_blocks);
    return true;
}

bool llama_kv_cache_paged::prepare_batch(const llama_batch & batch) {
    if (batch.n_tokens <= 0) {
        return false;
    }

    // ponytail: llama-server is configured with --parallel 1; keep this
    // fallback narrow until the scheduler owns the server request lifecycle.
    const llama_seq_id seq_id = batch.seq_id[0][0];
    for (int32_t i = 0; i < batch.n_tokens; ++i) {
        if (batch.n_seq_id[i] != 1 || batch.seq_id[i][0] != seq_id) {
            LLAMA_LOG_ERROR("%s: server-side paged KV currently supports one sequence per batch\n", __func__);
            return false;
        }
    }

    const llama_pos previous_max = seq_pos_max(seq_id);
    const llama_pos first_pos = batch.pos[0];
    if (previous_max >= 0 && first_pos != previous_max + 1) {
        LLAMA_LOG_ERROR("%s: non-contiguous sequence positions for seq %d: previous_max=%d first=%d\n",
                        __func__, seq_id, previous_max, first_pos);
        return false;
    }

    for (int32_t i = 1; i < batch.n_tokens; ++i) {
        if (batch.pos[i] != batch.pos[i - 1] + 1) {
            LLAMA_LOG_ERROR("%s: server-side paged KV requires contiguous positions\n", __func__);
            return false;
        }
    }

    auto & group = regular_groups[seq_id];
    group.request_id = seq_id;
    group.n_prompt = 0;
    group.n_decoded = previous_max >= 0 ? previous_max + 1 : 0;

    const size_t old_size = group.block_table.size();
    if (allocate(batch.n_tokens, group)) {
        return true;
    }

    if (group.block_table.size() > old_size) {
        llama_block_ids rollback(group.block_table.begin() + old_size, group.block_table.end());
        release_gpu_blocks(rollback);
        group.block_table.resize(old_size);
    }
    if (old_size == 0) {
        regular_groups.erase(seq_id);
    }
    return false;
}

bool llama_kv_cache_paged::build_batch_info(const llama_ubatch & ubatch, llama_paged_batch_info & info) const {
    if (ubatch.n_tokens == 0 || ubatch.n_seqs_unq != 1 || ubatch.n_pos < 1) {
        return false;
    }

    const llama_seq_id seq_id = ubatch.seq_id_unq[0];
    const auto it = regular_groups.find(seq_id);
    if (it == regular_groups.end() || it->second.block_table.empty()) {
        return false;
    }

    const auto & block_table = it->second.block_table;
    regular_write_slots.resize(ubatch.n_tokens);
    regular_block_table.assign(num_gpu_blocks, 0);
    std::copy(block_table.begin(), block_table.end(), regular_block_table.begin());
    regular_context_lens.assign(1, 0);
    regular_batch_offsets.assign(1, 0);
    regular_batch_lens.assign(1, ubatch.n_tokens);

    llama_pos max_pos = -1;
    for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
        // Qwen uses 4-position M-RoPE; the first position plane is the
        // sequential token position used by the paged KV block table.
        const llama_pos pos = ubatch.pos[i];
        const uint32_t block = pos / block_size;
        if (block >= block_table.size()) {
            return false;
        }
        regular_write_slots[i] = block_table[block] * block_size + (pos % block_size);
        max_pos = std::max(max_pos, pos);
    }

    regular_context_lens[0] = max_pos + 1;
    info.n_blocks_per_seq = num_gpu_blocks;
    info.n_seq            = 1;
    info.n_tokens         = ubatch.n_tokens;
    info.write_slots      = regular_write_slots.data();
    info.block_table      = regular_block_table.data();
    info.context_lens     = regular_context_lens.data();
    info.batch_offsets     = regular_batch_offsets.data();
    info.batch_lens        = regular_batch_lens.data();
    return true;
}

void llama_kv_cache_paged::commit_batch(const llama_ubatch & ubatch) {
    if (ubatch.n_tokens == 0 || ubatch.n_seqs_unq != 1) {
        return;
    }

    const llama_seq_id seq_id = ubatch.seq_id_unq[0];
    auto & range = sequence_positions[seq_id];
    range.min = range.min < 0 ? ubatch.pos[0] : range.min;
    range.max = ubatch.pos[ubatch.n_tokens - 1];
}

void llama_kv_cache_paged::free_blocks(llama_sequence_group & group) {
    if (group.block_table.empty()) {
        return;
    }

    llama_block_ids blocks_to_free_gpu;
    llama_block_ids blocks_to_free_cpu;

    for (uint32_t block_id : group.block_table) {
        if (block_manager.is_gpu(block_id)) {
            blocks_to_free_gpu.push_back(block_id);
        } else {
            blocks_to_free_cpu.push_back(block_id);
        }
    }

    if (!blocks_to_free_gpu.empty()) {
        release_gpu_blocks(blocks_to_free_gpu);
    }
    if (!blocks_to_free_cpu.empty()) {
        block_manager.release_cpu_blocks(blocks_to_free_cpu);
    }

    group.block_table.clear();
    sequence_positions.erase(group.request_id);
    if (required_gpu_capacity() <= initial_num_gpu_blocks) {
        maybe_restore_initial_storage();
    }
}

void llama_kv_cache_paged::do_block_copy(const llama_block_ids & src_ids,
                                         const llama_block_ids & new_ids,
                                         bool                    to_gpu) {
    const uint32_t num_blocks = src_ids.size();
    LLAMA_LOG_DEBUG("%s: num_blocks_size=%d, new_ids_size=%ld\n", __func__, num_blocks, new_ids.size());
    GGML_ASSERT(num_blocks == new_ids.size() && "src_ids and new_ids do not have the same size.");

    GGML_ASSERT(kv_cpu_layers.size() == attn_layer_ids.size() && "CPU and attention layer counts differ.");

    // Buffer on HOST to faciliate block data transfer
    // Note: an optimization would be to use views and async copies. Beware of
    // memory overhead heurisitcs.
    std::vector<uint8_t> staging(block_bytes);

    for (uint32_t il = 0; il < attn_layer_ids.size(); ++il) {
        const int32_t layer_id = attn_layer_ids[il];
        struct ggml_tensor * gpu_main = kv_gpu_layers[layer_id].tensor;
        struct ggml_tensor * cpu_main = get_cpu_tensor(layer_id);
        struct ggml_tensor * src_main = to_gpu ? cpu_main : gpu_main;
        struct ggml_tensor * dst_main = to_gpu ? gpu_main : cpu_main;

        ggml_backend_t src_backend = to_gpu ? cpu_backend : kv_gpu_layers[layer_id].backend;
        ggml_backend_t dst_backend = to_gpu ? kv_gpu_layers[layer_id].backend : cpu_backend;
        ggml_backend_synchronize(src_backend);
        ggml_backend_synchronize(dst_backend);

        for (uint32_t i = 0; i < num_blocks; ++i) {
            const uint32_t src_global = src_ids[i];
            const uint32_t dst_global = new_ids[i];

            // GPU and CPu blocks may differ (usually CPU < GPU)
            // We substract the diffence to calculate where the local starts before we calculate offsets
            const uint32_t src_local = to_gpu ? src_global - num_gpu_blocks : src_global;
            const uint32_t dst_local = to_gpu ? dst_global : dst_global - num_gpu_blocks;

            const size_t src_offset = (size_t) src_local * block_bytes;
            const size_t dst_offset = (size_t) dst_local * block_bytes;

            // Put src tensor into HOST staging buffer
            ggml_backend_tensor_get(src_main, staging.data(), src_offset, block_bytes);
            // Put tensor from HOST staging into dst tensor
            ggml_backend_tensor_set(dst_main, staging.data(), dst_offset, block_bytes);
        }
    }
}

bool llama_kv_cache_paged::swap_in(llama_sequence_group & group) {
    const uint32_t num_blocks = group.block_table.size();
    if (num_blocks == 0) {
        return true;
    }

    // A potential optimization to reduce thrashing is to have a heuristic to check if
    // if we can continue decoding after swap_in.
    if (!has_free_gpu_blocks(num_blocks, /*ignore_watermark=*/true)) {
        return false;
    }

    llama_block_ids preview_ids = preview_gpu_blocks(num_blocks);
    if (preview_ids.size() != num_blocks) {
        return false;
    }

    uint32_t required_blocks = required_gpu_capacity();
    if (!preview_ids.empty()) {
        required_blocks = std::max(required_blocks, *std::max_element(preview_ids.begin(), preview_ids.end()) + 1);
    }
    if (!ensure_physical_capacity(required_blocks)) {
        return false;
    }

    llama_block_ids new_ids = checkout_gpu_blocks(num_blocks);
    do_block_copy(group.block_table, new_ids, /*to_gpu=*/true);

    free_blocks(group);
    group.block_table = new_ids;
    return true;
}

bool llama_kv_cache_paged::swap_out(llama_sequence_group & group) {
    const uint32_t num_blocks = group.block_table.size();
    if (num_blocks == 0) {
        return true;
    }

    if (!block_manager.has_free_cpu_blocks(num_blocks)) {
        return false;
    }

    llama_block_ids new_ids = block_manager.checkout_cpu_blocks(num_blocks);
    do_block_copy(group.block_table, new_ids, /*to_gpu=*/false);

    free_blocks(group);
    group.block_table = new_ids;
    return true;
}

void llama_kv_cache_paged::set_paged_batch_info(const llama_paged_batch_info * info) {
    last_paged_info = info;
}

uint32_t llama_kv_cache_paged::get_num_gpu_blocks() const {
    return num_gpu_blocks;
}

void llama_kv_cache_paged::concat_block_ids(llama_block_ids &       to_block_table,
                                            const llama_block_ids & from_block_table) {
    to_block_table.insert(to_block_table.end(), from_block_table.begin(), from_block_table.end());
}

// llama_memory_i

llama_memory_context_ptr llama_kv_cache_paged::init_batch(llama_batch_allocr & balloc,
                                                          uint32_t             n_ubatch,
                                                          bool /*embd_all*/) {
    do {
        balloc.split_reset();

        std::vector<llama_ubatch> ubatches;
        while (true) {
            auto ubatch = balloc.split_simple(n_ubatch);
            if (ubatch.n_tokens == 0) {
                break;
            }
            ubatches.push_back(std::move(ubatch));
        }

        // Failed to find a suitable split
        if (balloc.get_n_used() < balloc.get_n_tokens()) {
            break;
        }

        const bool scheduler_batch = last_paged_info != nullptr;
        if (!scheduler_batch && !prepare_batch(balloc.get_batch())) {
            break;
        }

        auto ctx = std::make_unique<llama_kv_cache_paged_context>(this, std::move(ubatches));
        if (scheduler_batch) {
            ctx->set_batch_data(*last_paged_info);
        } else {
            ctx->set_batch_data(ctx->get_ubatch());
        }
        return ctx;
    } while (false);

    return std::make_unique<llama_kv_cache_paged_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
}

// Used by llama_context scheduler to dry-run
llama_memory_context_ptr llama_kv_cache_paged::init_full() {
    LLAMA_LOG_DEBUG("%s: reserving graph for n_ubatch=%d, n_seq_max=%d, num_gpu_blocks=%d\n", __func__, n_ubatch,
                    n_seq_max, num_gpu_blocks);

    // Create a "dummy" ubatch that represents the maximum capacity
    // of the system to let the scheduler reserve enough space for metadata.
    llama_ubatch ubatch = {};
    ubatch.n_tokens     = n_ubatch;   // maximum tokens
    ubatch.n_seqs       = n_seq_max;  // maximum sequences
    ubatch.n_pos        = 1;

    std::vector<llama_ubatch> ubatches = { ubatch };

    auto ctx = std::make_unique<llama_kv_cache_paged_context>(this, ubatches);

    ctx->set_batch_size(n_seq_max);       // maximum possible sequences
    ctx->set_n_tokens(n_ubatch);          // representative token count
    ctx->set_max_blocks(num_gpu_blocks);  // every block could theoretically belong to one seq

    return ctx;
}

llama_memory_context_ptr llama_kv_cache_paged::init_update(llama_context * /*lctx*/, bool /*optimize*/) {
    return std::make_unique<llama_kv_cache_paged_context>(LLAMA_MEMORY_STATUS_NO_UPDATE);
}

struct ggml_tensor * llama_kv_cache_paged::get_kv_tensor(int layer_idx) const {
    GGML_ASSERT(layer_idx >= 0 && layer_idx < (int32_t) kv_gpu_layers.size());
    GGML_ASSERT(kv_gpu_layers[layer_idx].tensor != nullptr && "layer not in paged KV cache");
    return kv_gpu_layers[layer_idx].tensor;
}

void llama_kv_cache_paged::clear(bool /*data*/) {
    for (auto & entry : regular_groups) {
        if (!entry.second.block_table.empty()) {
            llama_block_ids released_gpu;
            llama_block_ids released_cpu;
            for (const uint32_t block : entry.second.block_table) {
                (block_manager.is_gpu(block) ? released_gpu : released_cpu).push_back(block);
            }
            if (!released_gpu.empty()) {
                release_gpu_blocks(released_gpu);
            }
            if (!released_cpu.empty()) {
                block_manager.release_cpu_blocks(released_cpu);
            }
        }
    }
    regular_groups.clear();
    sequence_positions.clear();
    maybe_restore_initial_storage();
}

bool llama_kv_cache_paged::seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos /*p1*/) {
    auto it = regular_groups.find(seq_id);
    if (it != regular_groups.end()) {
        auto & group = it->second;
        if (p0 <= 0) {
            if (!group.block_table.empty()) {
                free_blocks(group);
            }
            regular_groups.erase(it);
            sequence_positions.erase(seq_id);
            return true;
        }

        const size_t keep_blocks = std::min(group.block_table.size(),
                                            (size_t) (p0 + block_size - 1) / block_size);
        llama_block_ids released(group.block_table.begin() + keep_blocks, group.block_table.end());
        if (!released.empty()) {
            llama_block_ids released_gpu;
            llama_block_ids released_cpu;
            for (const uint32_t block : released) {
                (block_manager.is_gpu(block) ? released_gpu : released_cpu).push_back(block);
            }
            if (!released_gpu.empty()) {
                release_gpu_blocks(released_gpu);
            }
            if (!released_cpu.empty()) {
                block_manager.release_cpu_blocks(released_cpu);
            }
            group.block_table.resize(keep_blocks);
        }

        auto pos_it = sequence_positions.find(seq_id);
        if (pos_it != sequence_positions.end()) {
            pos_it->second.max = std::min(pos_it->second.max, p0 - 1);
        }
    }
    return true;
}

llama_pos llama_kv_cache_paged::seq_pos_min(llama_seq_id seq_id) const {
    auto it = sequence_positions.find(seq_id);
    return (it != sequence_positions.end()) ? it->second.min : -1;
}

llama_pos llama_kv_cache_paged::seq_pos_max(llama_seq_id seq_id) const {
    auto it = sequence_positions.find(seq_id);
    return (it != sequence_positions.end()) ? it->second.max : -1;
}

std::map<ggml_backend_buffer_type_t, size_t> llama_kv_cache_paged::memory_breakdown() const {
    std::map<ggml_backend_buffer_type_t, size_t> breakdown;

    for (const auto & storage : kv_gpu_layers) {
        if (storage.tensor != nullptr) {
            breakdown[ggml_backend_buffer_get_type(storage.tensor->buffer)] += ggml_nbytes(storage.tensor);
        }
    }
    for (const auto * kv_cpu : kv_cpu_layers) {
        breakdown[ggml_backend_buffer_get_type(kv_cpu->buffer)] += ggml_nbytes(kv_cpu);
    }
    return breakdown;
}

void llama_kv_cache_paged::set_seq_min_pos(llama_seq_id seq_id, llama_pos new_min) {
    sequence_positions[seq_id].min = new_min;
}

void llama_kv_cache_paged::set_seq_max_pos(llama_seq_id seq_id, llama_pos new_max) {
    sequence_positions[seq_id].max = new_max;
}

// llama_kv_cache_paged_context

void llama_kv_cache_paged_context::set_batch_data(const llama_paged_batch_info & info) {
    owned_write_slots.assign(info.write_slots, info.write_slots + info.n_tokens);
    owned_block_table.assign(info.block_table, info.block_table + info.n_seq * info.n_blocks_per_seq);
    owned_context_lens.assign(info.context_lens, info.context_lens + info.n_seq);
    owned_batch_offsets.assign(info.batch_offsets, info.batch_offsets + info.n_seq);
    owned_batch_lens.assign(info.batch_lens, info.batch_lens + info.n_seq);
    paged_write_slots   = owned_write_slots.data();
    paged_block_table   = owned_block_table.data();
    paged_context_lens  = owned_context_lens.data();
    paged_batch_offsets = owned_batch_offsets.data();
    paged_batch_lens    = owned_batch_lens.data();
    n_tokens            = info.n_tokens;
    max_blocks          = info.n_blocks_per_seq;
    batch_size          = info.n_seq;
}

void llama_kv_cache_paged_context::set_batch_data(const llama_ubatch & ubatch) {
    llama_paged_batch_info info = {};
    GGML_ASSERT(manager && manager->build_batch_info(ubatch, info));
    set_batch_data(info);
}

bool llama_kv_cache_paged_context::next() {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);
    manager->commit_batch(ubatches[i_cur]);
    if (++i_cur >= ubatches.size()) {
        return false;
    }
    set_batch_data(ubatches[i_cur]);
    return true;
}

bool llama_kv_cache_paged_context::apply() {
    // Nothing to do for paged KV cache, return true to allow for execution
    return true;
}

const llama_ubatch & llama_kv_cache_paged_context::get_ubatch() const {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);
    return ubatches[i_cur];
}

struct ggml_tensor * llama_kv_cache_paged_context::get_k(int layer_idx) const {
    GGML_ASSERT(manager && "manager has not been initialized.");
    return manager->get_kv_tensor(layer_idx);
}

struct ggml_tensor * llama_kv_cache_paged_context::get_v(int layer_idx) const {
    GGML_ASSERT(manager && "manager has not been initialized.");
    return manager->get_kv_tensor(layer_idx);
}

int32_t llama_kv_cache_paged_context::get_n_tokens() const {
    return n_tokens;
}

int32_t llama_kv_cache_paged_context::get_batch_size() const {
    return batch_size;
}

int32_t llama_kv_cache_paged_context::get_max_blocks() const {
    return max_blocks;
}

uint64_t llama_kv_cache_paged_context::get_storage_generation() const {
    GGML_ASSERT(manager && "manager has not been initialized.");
    return manager->get_storage_generation();
}

int32_t * llama_kv_cache_paged_context::get_write_slots() const {
    return paged_write_slots;
}

int32_t * llama_kv_cache_paged_context::get_block_table() const {
    return paged_block_table;
}

int32_t * llama_kv_cache_paged_context::get_context_lens() const {
    return paged_context_lens;
}

int32_t * llama_kv_cache_paged_context::get_batch_offsets() const {
    return paged_batch_offsets;
}

int32_t * llama_kv_cache_paged_context::get_batch_lens() const {
    return paged_batch_lens;
}

void llama_kv_cache_paged_context::set_n_tokens(int32_t new_n_tokens) {
    n_tokens = new_n_tokens;
}

void llama_kv_cache_paged_context::set_batch_size(int32_t new_batch_size) {
    batch_size = new_batch_size;
}

void llama_kv_cache_paged_context::set_max_blocks(int32_t new_max_blocks) {
    max_blocks = new_max_blocks;
}
