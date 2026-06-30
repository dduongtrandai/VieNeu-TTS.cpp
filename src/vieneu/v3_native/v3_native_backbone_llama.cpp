#include "v3_native_backbone_llama.h"
#include <iostream>
#include <cstring>
#include <stdexcept>
#include <cstdlib>
#include <algorithm>

bool V3NativeBackbone::backend_initialized_ = false;

namespace {

int env_int_or_default(const char* name, int fallback) {
    const char* value = std::getenv(name);
    if (!value || !*value) {
        return (std::max)(1, fallback);
    }
    char* end = nullptr;
    const long parsed = std::strtol(value, &end, 10);
    if (end == value || parsed <= 0) {
        return (std::max)(1, fallback);
    }
    return static_cast<int>((std::min)(parsed, 256L));
}

} // namespace

V3NativeBackbone::V3NativeBackbone() {}

V3NativeBackbone::~V3NativeBackbone() {
    free_resources();
}

bool V3NativeBackbone::initialize(const V3BackboneParams& params) {
    free_resources();

    if (!backend_initialized_) {
        // Quiet logging
        llama_log_set([](enum ggml_log_level level, const char * text, void * user_data) {
            (void)user_data;
            if (level >= GGML_LOG_LEVEL_ERROR && text) {
                fputs(text, stderr);
            }
        }, nullptr);
        llama_backend_init();
        backend_initialized_ = true;
    }

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = params.n_gpu_layers;

    model_ = llama_model_load_from_file(params.model_path.c_str(), model_params);
    if (!model_) {
        std::cerr << "[V3NativeBackbone] Failed to load model from " << params.model_path << std::endl;
        return false;
    }

    hidden_size_ = llama_model_n_embd(model_);
    const int n_threads = env_int_or_default("VIENEU_BACKBONE_THREADS", params.n_threads);
    const int n_threads_batch = env_int_or_default("VIENEU_BACKBONE_BATCH_THREADS", params.n_threads_batch > 0 ? params.n_threads_batch : n_threads);

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = 2048;
    ctx_params.n_threads = n_threads;
    ctx_params.n_threads_batch = n_threads_batch;
    ctx_params.embeddings = true; // Enable embeddings extraction
    ctx_params.no_perf = true;

    ctx_ = llama_init_from_model(model_, ctx_params);
    if (!ctx_) {
        std::cerr << "[V3NativeBackbone] Failed to initialize context." << std::endl;
        llama_model_free(model_);
        model_ = nullptr;
        return false;
    }

    prefill_capacity_ = ctx_params.n_ctx;
    prefill_batch_ = llama_batch_init(prefill_capacity_, hidden_size_, 1);
    prefill_batch_initialized_ = true;
    decode_batch_ = llama_batch_init(1, hidden_size_, 1);
    decode_batch_initialized_ = true;
    decoded_pos_ = 0;
    return true;
}

void V3NativeBackbone::free_resources() {
    if (prefill_batch_initialized_) {
        llama_batch_free(prefill_batch_);
        prefill_batch_ = {};
        prefill_batch_initialized_ = false;
    }
    if (decode_batch_initialized_) {
        llama_batch_free(decode_batch_);
        decode_batch_ = {};
        decode_batch_initialized_ = false;
    }
    if (ctx_) {
        llama_free(ctx_);
        ctx_ = nullptr;
    }
    if (model_) {
        llama_model_free(model_);
        model_ = nullptr;
    }
    decoded_pos_ = 0;
    prefill_capacity_ = 0;
}

bool V3NativeBackbone::prefill(const std::vector<float>& embeds, std::vector<float>& out_hidden) {
    if (!ctx_ || embeds.empty()) return false;

    const int32_t n_tokens = static_cast<int32_t>(embeds.size() / hidden_size_);
    if (n_tokens <= 0) return false;
    if (!prefill_batch_initialized_ || n_tokens > prefill_capacity_) {
        std::cerr << "[V3NativeBackbone] Prefill batch capacity is insufficient." << std::endl;
        return false;
    }

    // Reset KV cache and decoded position
    clear_kv_cache();

    // Copy input embeddings
    std::memcpy(prefill_batch_.embd, embeds.data(), embeds.size() * sizeof(float));

    for (int32_t i = 0; i < n_tokens; ++i) {
        prefill_batch_.pos[i] = i;
        prefill_batch_.n_seq_id[i] = 1;
        prefill_batch_.seq_id[i][0] = 0;
        prefill_batch_.logits[i] = (i == n_tokens - 1); // request logits/embedding output for last token only
    }
    prefill_batch_.n_tokens = n_tokens;

    int res = llama_decode(ctx_, prefill_batch_);

    if (res != 0) {
        std::cerr << "[V3NativeBackbone] Prefill failed with code: " << res << std::endl;
        return false;
    }

    decoded_pos_ = n_tokens;

    // Retrieve last hidden state from context output embeddings
    float* emb_ptr = llama_get_embeddings_ith(ctx_, -1);
    if (!emb_ptr) {
        // Fallback to first if -1 is null
        emb_ptr = llama_get_embeddings_ith(ctx_, 0);
    }
    if (!emb_ptr) {
        std::cerr << "[V3NativeBackbone] Prefill failed to retrieve output embeddings." << std::endl;
        return false;
    }

    out_hidden.resize(static_cast<size_t>(hidden_size_));
    std::memcpy(out_hidden.data(), emb_ptr, hidden_size_ * sizeof(float));
    return true;
}

bool V3NativeBackbone::decode_step(const std::vector<float>& slot_embed, std::vector<float>& out_hidden) {
    if (!ctx_ || !decode_batch_initialized_ || slot_embed.size() != static_cast<size_t>(hidden_size_)) return false;

    std::memcpy(decode_batch_.embd, slot_embed.data(), slot_embed.size() * sizeof(float));

    decode_batch_.pos[0] = decoded_pos_;
    decode_batch_.n_seq_id[0] = 1;
    decode_batch_.seq_id[0][0] = 0;
    decode_batch_.logits[0] = true;
    decode_batch_.n_tokens = 1;

    int res = llama_decode(ctx_, decode_batch_);

    if (res != 0) {
        std::cerr << "[V3NativeBackbone] Decode step failed with code: " << res << std::endl;
        return false;
    }

    decoded_pos_ += 1;

    float* emb_ptr = llama_get_embeddings_ith(ctx_, 0);
    if (!emb_ptr) {
        std::cerr << "[V3NativeBackbone] Decode step failed to retrieve output embeddings." << std::endl;
        return false;
    }

    out_hidden.resize(static_cast<size_t>(hidden_size_));
    std::memcpy(out_hidden.data(), emb_ptr, hidden_size_ * sizeof(float));
    return true;
}

void V3NativeBackbone::clear_kv_cache() {
    if (ctx_) {
        llama_memory_t mem = llama_get_memory(ctx_);
        if (mem) {
            llama_memory_clear(mem, true);
        }
    }
    decoded_pos_ = 0;
}
