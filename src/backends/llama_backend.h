#ifndef LLAMA_BACKEND_H
#define LLAMA_BACKEND_H

#include <string>
#include <vector>
#include "llama.h"

struct LlamaBackendParams {
    std::string model_path;
    int n_ctx = 2048;
    int n_threads = 4;
    int n_gpu_layers = 0;
    bool flash_attn = false;
    bool mlock = true;
};

class LlamaBackend {
public:
    LlamaBackend();
    ~LlamaBackend();

    bool initialize(const LlamaBackendParams& params);
    void free_resources();

    std::vector<llama_token> tokenize(const std::string& text, bool add_special) const;
    std::string token_to_piece(llama_token token) const;

    bool decode(const std::vector<llama_token>& tokens, int32_t seq_id, bool is_first);
    llama_token sample(float temperature, int top_k);
    void clear_kv_cache();

    bool is_eog(llama_token token) const;
    llama_token get_eog_token() const;

private:
    llama_model* model = nullptr;
    llama_context* ctx = nullptr;
    llama_sampler* sampler = nullptr;
    
    int32_t decoded_pos = 0;
    int n_ctx_val = 2048;

    static bool backend_initialized;
};

#endif // LLAMA_BACKEND_H
