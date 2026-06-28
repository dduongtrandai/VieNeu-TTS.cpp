#include "llama_backend.h"
#include <iostream>
#include <stdexcept>
#include <cstdlib>
#include <cstdio>

bool LlamaBackend::backend_initialized = false;

static void quiet_llama_log_callback(enum ggml_log_level level, const char * text, void * user_data) {
    (void)user_data;
    if (level >= GGML_LOG_LEVEL_ERROR && text) {
        fputs(text, stderr);
    }
}

LlamaBackend::LlamaBackend() {}

LlamaBackend::~LlamaBackend() {
    free_resources();
}

bool LlamaBackend::initialize(const LlamaBackendParams& params) {
    free_resources();

    n_ctx_val = params.n_ctx;

    if (!backend_initialized) {
        const char * verbose = std::getenv("VIENEU_TTS_VERBOSE");
        if (!verbose || std::string(verbose) != "1") {
            llama_log_set(quiet_llama_log_callback, nullptr);
        }
        ggml_backend_load_all();
        backend_initialized = true;
    }

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = params.n_gpu_layers;

    model = llama_model_load_from_file(params.model_path.c_str(), model_params);
    if (!model) {
        return false;
    }

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = params.n_ctx;
    ctx_params.n_threads = params.n_threads;
    ctx_params.n_threads_batch = params.n_threads;
    ctx_params.flash_attn_type = params.flash_attn ? LLAMA_FLASH_ATTN_TYPE_ENABLED : LLAMA_FLASH_ATTN_TYPE_DISABLED;
    ctx_params.no_perf = true;

    ctx = llama_init_from_model(model, ctx_params);
    if (!ctx) {
        llama_model_free(model);
        model = nullptr;
        return false;
    }

    return true;
}

void LlamaBackend::free_resources() {
    if (ctx) {
        llama_free(ctx);
        ctx = nullptr;
    }
    if (model) {
        llama_model_free(model);
        model = nullptr;
    }
}

std::vector<llama_token> LlamaBackend::tokenize(const std::string& text, bool add_special) const {
    if (!model) return {};

    const struct llama_vocab * vocab = llama_model_get_vocab(model);
    std::vector<llama_token> tokens(text.size() + 4);
    int n_tokens = llama_tokenize(vocab, text.c_str(), (int)text.size(), tokens.data(), (int)tokens.size(), add_special, true);
    if (n_tokens < 0) {
        tokens.resize(-n_tokens);
        n_tokens = llama_tokenize(vocab, text.c_str(), (int)text.size(), tokens.data(), (int)tokens.size(), add_special, true);
    }
    tokens.resize(n_tokens);
    return tokens;
}

std::string LlamaBackend::token_to_piece(llama_token token) const {
    if (!model) return "";

    const struct llama_vocab * vocab = llama_model_get_vocab(model);
    char buf[128];
    int n = llama_token_to_piece(vocab, token, buf, sizeof(buf), 0, true);
    if (n < 0) {
        return "";
    }
    return std::string(buf, n);
}

bool LlamaBackend::decode(const std::vector<llama_token>& tokens, int32_t seq_id, bool is_first) {
    if (tokens.empty() || !ctx) return true;

    if (is_first) {
        decoded_pos = 0;
        clear_kv_cache();
    }

    llama_batch batch = llama_batch_init((int32_t)tokens.size(), 0, 1);
    for (size_t i = 0; i < tokens.size(); ++i) {
        batch.token[i] = tokens[i];
        batch.pos[i] = decoded_pos + (int32_t)i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = seq_id;
        batch.logits[i] = (i == tokens.size() - 1);
    }
    batch.n_tokens = (int32_t)tokens.size();

    int res = llama_decode(ctx, batch);
    llama_batch_free(batch);

    if (res != 0) {
        return false;
    }

    decoded_pos += (int32_t)tokens.size();
    return true;
}

llama_token LlamaBackend::sample(float temperature, int top_k) {
    if (!ctx) return LLAMA_TOKEN_NULL;

    auto sparams = llama_sampler_chain_default_params();
    llama_sampler * smpl = llama_sampler_chain_init(sparams);
    if (temperature <= 0.0f) {
        llama_sampler_chain_add(smpl, llama_sampler_init_greedy());
    } else {
        llama_sampler_chain_add(smpl, llama_sampler_init_top_k(top_k));
        llama_sampler_chain_add(smpl, llama_sampler_init_top_p(0.95f, 1));
        llama_sampler_chain_add(smpl, llama_sampler_init_temp(temperature));
        llama_sampler_chain_add(smpl, llama_sampler_init_dist(42));
    }

    llama_token next_token = llama_sampler_sample(smpl, ctx, -1);
    llama_sampler_free(smpl);

    return next_token;
}

void LlamaBackend::clear_kv_cache() {
    if (ctx) {
        llama_memory_t mem = llama_get_memory(ctx);
        if (mem) {
            llama_memory_clear(mem, true);
        }
    }
    decoded_pos = 0;
}

bool LlamaBackend::is_eog(llama_token token) const {
    if (!model) return true;
    const struct llama_vocab * vocab = llama_model_get_vocab(model);
    return llama_vocab_is_eog(vocab, token);
}

llama_token LlamaBackend::get_eog_token() const {
    if (!model) return LLAMA_TOKEN_NULL;
    const struct llama_vocab * vocab = llama_model_get_vocab(model);
    return llama_vocab_eos(vocab);
}
