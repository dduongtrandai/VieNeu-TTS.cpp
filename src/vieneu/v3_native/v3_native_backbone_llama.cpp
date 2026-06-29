#include "v3_native_backbone_llama.h"
#include <iostream>
#include <cstring>
#include <stdexcept>

bool V3NativeBackbone::backend_initialized_ = false;

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

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = 2048;
    ctx_params.n_threads = params.n_threads;
    ctx_params.n_threads_batch = params.n_threads;
    ctx_params.embeddings = true; // Enable embeddings extraction
    ctx_params.no_perf = true;

    ctx_ = llama_init_from_model(model_, ctx_params);
    if (!ctx_) {
        std::cerr << "[V3NativeBackbone] Failed to initialize context." << std::endl;
        llama_model_free(model_);
        model_ = nullptr;
        return false;
    }

    hidden_size_ = llama_model_n_embd(model_);
    decoded_pos_ = 0;
    return true;
}

void V3NativeBackbone::free_resources() {
    if (ctx_) {
        llama_free(ctx_);
        ctx_ = nullptr;
    }
    if (model_) {
        llama_model_free(model_);
        model_ = nullptr;
    }
    decoded_pos_ = 0;
}

bool V3NativeBackbone::prefill(const std::vector<float>& embeds, std::vector<float>& out_hidden) {
    if (!ctx_ || embeds.empty()) return false;

    const int32_t n_tokens = static_cast<int32_t>(embeds.size() / hidden_size_);
    if (n_tokens <= 0) return false;

    // Reset KV cache and decoded position
    clear_kv_cache();

    // Use llama_batch_init with embd = hidden_size_ to enable input embeddings
    llama_batch batch = llama_batch_init(n_tokens, hidden_size_, 1);
    
    // Copy input embeddings
    std::memcpy(batch.embd, embeds.data(), embeds.size() * sizeof(float));

    for (int32_t i = 0; i < n_tokens; ++i) {
        batch.pos[i] = i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = (i == n_tokens - 1); // request logits/embedding output for last token only
    }
    batch.n_tokens = n_tokens;

    int res = llama_decode(ctx_, batch);
    llama_batch_free(batch);

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
    if (!ctx_ || slot_embed.size() != static_cast<size_t>(hidden_size_)) return false;

    llama_batch batch = llama_batch_init(1, hidden_size_, 1);
    std::memcpy(batch.embd, slot_embed.data(), slot_embed.size() * sizeof(float));

    batch.pos[0] = decoded_pos_;
    batch.n_seq_id[0] = 1;
    batch.seq_id[0][0] = 0;
    batch.logits[0] = true;
    batch.n_tokens = 1;

    int res = llama_decode(ctx_, batch);
    llama_batch_free(batch);

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
