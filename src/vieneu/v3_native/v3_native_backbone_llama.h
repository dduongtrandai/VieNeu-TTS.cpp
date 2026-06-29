#ifndef V3_NATIVE_BACKBONE_LLAMA_H
#define V3_NATIVE_BACKBONE_LLAMA_H

#include <string>
#include <vector>
#include "llama.h"

struct V3BackboneParams {
    std::string model_path;
    int n_threads = 4;
    int n_gpu_layers = 0;
};

class V3NativeBackbone {
public:
    V3NativeBackbone();
    ~V3NativeBackbone();

    bool initialize(const V3BackboneParams& params);
    void free_resources();

    // Run backbone prefill on the prompt embeddings of shape [n_tokens, hidden_size]
    // Returns the final hidden state of shape [hidden_size]
    bool prefill(const std::vector<float>& embeds, std::vector<float>& out_hidden);

    // Run a single decode step on a single slot/frame embedding of shape [hidden_size]
    // Returns the next hidden state of shape [hidden_size]
    bool decode_step(const std::vector<float>& slot_embed, std::vector<float>& out_hidden);

    void clear_kv_cache();

private:
    llama_model* model_ = nullptr;
    llama_context* ctx_ = nullptr;
    int32_t decoded_pos_ = 0;
    int hidden_size_ = 768;

    static bool backend_initialized_;
};

#endif // V3_NATIVE_BACKBONE_LLAMA_H
