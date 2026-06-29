#include "v3_native_acoustic_ggml.h"
#include "ggml.h"
#include "ggml-cpu.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <iostream>

namespace {

int get_ggml_threads() {
    const char* val = std::getenv("OMP_NUM_THREADS");
    if (!val || !*val) {
        return 4;
    }
    char* end = nullptr;
    const long parsed = std::strtol(val, &end, 10);
    if (end == val || parsed <= 0) {
        return 4;
    }
    return static_cast<int>(std::min<long>(parsed, 32));
}

bool env_flag_enabled(const char* name) {
    const char* value = std::getenv(name);
    if (!value || !*value) {
        return false;
    }
    std::string normalized(value);
    std::transform(normalized.begin(), normalized.end(), normalized.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return normalized == "1" || normalized == "true" || normalized == "yes" || normalized == "on";
}

class GgmlLinear {
public:
    GgmlLinear() = default;
    ~GgmlLinear() { release(); }

    void initialize(ggml_backend_t backend, const float* weight, int out_dim, int in_dim) {
        release();
        (void)backend;
        thread_count_ = get_ggml_threads();
        in_dim_ = in_dim;
        out_dim_ = out_dim;

        const size_t ctx_size = 1024 * 1024 + static_cast<size_t>(in_dim_) * out_dim_ * sizeof(float) +
                                static_cast<size_t>(in_dim_ + out_dim_) * sizeof(float);
        ggml_init_params params = {
            /* .mem_size   = */ ctx_size,
            /* .mem_buffer = */ nullptr,
            /* .no_alloc   = */ false,
        };
        ctx_ = ggml_init(params);
        if (!ctx_) {
            throw std::runtime_error("Failed to initialize GGML context for linear layer.");
        }

        weight_ = ggml_new_tensor_2d(ctx_, GGML_TYPE_F32, in_dim_, out_dim_);
        input_ = ggml_new_tensor_1d(ctx_, GGML_TYPE_F32, in_dim_);
        output_ = ggml_mul_mat(ctx_, weight_, input_);
        graph_ = ggml_new_graph(ctx_);
        ggml_build_forward_expand(graph_, output_);

        std::memcpy(weight_->data, weight, static_cast<size_t>(in_dim_) * out_dim_ * sizeof(float));
        plan_ = ggml_graph_plan(graph_, thread_count_, nullptr);
        work_data_.resize(plan_.work_size);
        plan_.work_data = work_data_.empty() ? nullptr : work_data_.data();
    }

    void run(const float* input, std::vector<float>& output) {
        output.resize(static_cast<size_t>(out_dim_));
        std::memcpy(input_->data, input, static_cast<size_t>(in_dim_) * sizeof(float));
        const ggml_status status = ggml_graph_compute(graph_, &plan_);
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("GGML graph compute failed.");
        }
        std::memcpy(output.data(), output_->data, static_cast<size_t>(out_dim_) * sizeof(float));
    }

private:
    void release() {
        if (ctx_) {
            ggml_free(ctx_);
            ctx_ = nullptr;
        }
        weight_ = nullptr;
        input_ = nullptr;
        output_ = nullptr;
        graph_ = nullptr;
        in_dim_ = 0;
        out_dim_ = 0;
        thread_count_ = 1;
        work_data_.clear();
    }

    ggml_context* ctx_ = nullptr;
    ggml_tensor* weight_ = nullptr;
    ggml_tensor* input_ = nullptr;
    ggml_tensor* output_ = nullptr;
    ggml_cgraph* graph_ = nullptr;
    ggml_cplan plan_{};
    std::vector<uint8_t> work_data_;
    int in_dim_ = 0;
    int out_dim_ = 0;
    int thread_count_ = 1;
};

class GgmlFfnOp {
public:
    GgmlFfnOp() = default;
    ~GgmlFfnOp() { release(); }

    void initialize(ggml_backend_t backend,
                    const float* gate_weight,
                    const float* up_weight,
                    const float* down_weight,
                    int hidden_dim,
                    int intermediate_dim) {
        release();
        (void)backend;
        thread_count_ = get_ggml_threads();
        hidden_dim_ = hidden_dim;
        intermediate_dim_ = intermediate_dim;

        const size_t weight_bytes =
            2 * static_cast<size_t>(intermediate_dim_) * hidden_dim_ * sizeof(float) +
            static_cast<size_t>(hidden_dim_) * intermediate_dim_ * sizeof(float);
        const size_t activation_bytes =
            static_cast<size_t>(hidden_dim_ + 4 * intermediate_dim_) * sizeof(float);
        ggml_init_params params = {
            /* .mem_size   = */ 1024 * 1024 + weight_bytes + activation_bytes,
            /* .mem_buffer = */ nullptr,
            /* .no_alloc   = */ false,
        };
        ctx_ = ggml_init(params);
        if (!ctx_) {
            throw std::runtime_error("failed to initialize ggml context for ffn.");
        }

        gate_weight_ = ggml_new_tensor_2d(ctx_, GGML_TYPE_F32, hidden_dim_, intermediate_dim_);
        up_weight_ = ggml_new_tensor_2d(ctx_, GGML_TYPE_F32, hidden_dim_, intermediate_dim_);
        down_weight_ = ggml_new_tensor_2d(ctx_, GGML_TYPE_F32, intermediate_dim_, hidden_dim_);
        input_ = ggml_new_tensor_1d(ctx_, GGML_TYPE_F32, hidden_dim_);

        ggml_tensor* gate = ggml_mul_mat(ctx_, gate_weight_, input_);
        ggml_tensor* up = ggml_mul_mat(ctx_, up_weight_, input_);
        ggml_tensor* activated = ggml_silu(ctx_, gate);
        ggml_tensor* fused = ggml_mul(ctx_, activated, up);
        output_ = ggml_mul_mat(ctx_, down_weight_, fused);

        graph_ = ggml_new_graph(ctx_);
        ggml_build_forward_expand(graph_, output_);

        std::memcpy(gate_weight_->data, gate_weight, static_cast<size_t>(intermediate_dim_) * hidden_dim_ * sizeof(float));
        std::memcpy(up_weight_->data, up_weight, static_cast<size_t>(intermediate_dim_) * hidden_dim_ * sizeof(float));
        std::memcpy(down_weight_->data, down_weight, static_cast<size_t>(hidden_dim_) * intermediate_dim_ * sizeof(float));
        plan_ = ggml_graph_plan(graph_, thread_count_, nullptr);
        work_data_.resize(plan_.work_size);
        plan_.work_data = work_data_.empty() ? nullptr : work_data_.data();
    }

    void run(const float* input, std::vector<float>& output) {
        output.resize(static_cast<size_t>(hidden_dim_));
        std::memcpy(input_->data, input, static_cast<size_t>(hidden_dim_) * sizeof(float));
        const ggml_status status = ggml_graph_compute(graph_, &plan_);
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("ggml graph compute failed for ffn.");
        }
        std::memcpy(output.data(), output_->data, static_cast<size_t>(hidden_dim_) * sizeof(float));
    }

private:
    void release() {
        if (ctx_) {
            ggml_free(ctx_);
            ctx_ = nullptr;
        }
        gate_weight_ = nullptr;
        up_weight_ = nullptr;
        down_weight_ = nullptr;
        input_ = nullptr;
        output_ = nullptr;
        graph_ = nullptr;
        hidden_dim_ = 0;
        intermediate_dim_ = 0;
        thread_count_ = 1;
        work_data_.clear();
    }

    ggml_context* ctx_ = nullptr;
    ggml_tensor* gate_weight_ = nullptr;
    ggml_tensor* up_weight_ = nullptr;
    ggml_tensor* down_weight_ = nullptr;
    ggml_tensor* input_ = nullptr;
    ggml_tensor* output_ = nullptr;
    ggml_cgraph* graph_ = nullptr;
    ggml_cplan plan_{};
    std::vector<uint8_t> work_data_;
    int hidden_dim_ = 0;
    int intermediate_dim_ = 0;
    int thread_count_ = 1;
};

} // namespace

struct V3NativeAcoustic::Impl {
    const V3NativeConfig& config;
    const V3NativeAssets& assets;
    V3NativeSampler& sampler;

    ggml_backend_t backend = nullptr;
    
    struct LayerCache {
        std::vector<float> k;
        std::vector<float> v;
    };

    struct LayerOps {
        GgmlLinear qkv;
        GgmlLinear o_proj;
        GgmlLinear ff_gate;
        GgmlLinear ff_up;
        GgmlLinear ff_down;
        GgmlFfnOp ffn;
        bool fuse_ffn = false;
    };

    std::vector<LayerOps> layer_ops;
    std::vector<LayerCache> caches;

    std::vector<float> token;
    std::vector<float> hidden;
    std::vector<float> slot0;
    std::vector<float> logits;
    std::vector<float> text_logits;
    std::vector<float> x;
    std::vector<float> normed;
    std::vector<float> qkv_val;
    std::vector<float> q;
    std::vector<float> new_k;
    std::vector<float> new_v;
    std::vector<float> attn_out;
    std::vector<float> proj;
    std::vector<float> gate;
    std::vector<float> up;
    std::vector<float> down;
    std::vector<float> scores;

    Impl(const V3NativeConfig& cfg, const V3NativeAssets& ast, V3NativeSampler& smpl)
        : config(cfg), assets(ast), sampler(smpl) {
        backend = ggml_backend_cpu_init();
        if (!backend) {
            throw std::runtime_error("Failed to initialize GGML CPU backend.");
        }
        ggml_backend_cpu_set_n_threads(backend, get_ggml_threads());
    }

    ~Impl() {
        layer_ops.clear();
        if (backend) {
            ggml_backend_free(backend);
            backend = nullptr;
        }
    }

    static float silu(float val) {
        return val / (1.0f + std::exp(-val));
    }

    static void rms_norm(const float* input, const float* weight, int dim, float eps, float* out) {
        double sum_sq = 0.0;
        for (int i = 0; i < dim; ++i) {
            sum_sq += static_cast<double>(input[i]) * input[i];
        }
        const float scale = 1.0f / std::sqrt(static_cast<float>(sum_sq / dim) + eps);
        for (int i = 0; i < dim; ++i) {
            out[i] = input[i] * scale * weight[i];
        }
    }

    void initialize_ops() {
        const auto& w = assets.acoustic_weights();
        const int H = config.hidden_size;
        const int I = config.local_intermediate_size;
        const bool fuse = env_flag_enabled("VIENEU_GGML_FUSE_FFN");
        layer_ops.clear();
        layer_ops.resize(w.layers.size());
        for (size_t i = 0; i < w.layers.size(); ++i) {
            const auto& wl = w.layers[i];
            LayerOps& ops = layer_ops[i];
            ops.fuse_ffn = fuse;
            ops.qkv.initialize(backend, wl.qkv.data(), 3 * H, H);
            ops.o_proj.initialize(backend, wl.o_proj.data(), H, H);
            if (fuse) {
                ops.ffn.initialize(backend, wl.ff_gate.data(), wl.ff_up.data(), wl.ff_down.data(), H, I);
            } else {
                ops.ff_gate.initialize(backend, wl.ff_gate.data(), I, H);
                ops.ff_up.initialize(backend, wl.ff_up.data(), I, H);
                ops.ff_down.initialize(backend, wl.ff_down.data(), H, I);
            }
        }
    }

    void cached_step(const std::vector<float>& input, const std::vector<int>& positions, std::vector<float>& output) {
        const auto& w = assets.acoustic_weights();
        const int H = config.hidden_size;
        const int nH = config.local_num_attention_heads;
        const int hd = H / nH;
        const int I = config.local_intermediate_size;
        const int S = static_cast<int>(positions.size());
        const float inv_sqrt_hd = 1.0f / std::sqrt(static_cast<float>(hd));

        x = input;
        for (int s = 0; s < S; ++s) {
            const int pos = positions[static_cast<size_t>(s)];
            const float* pe = w.slot_pos_emb.data() + static_cast<size_t>(pos) * H;
            float* dst = x.data() + static_cast<size_t>(s) * H;
            for (int i = 0; i < H; ++i) {
                dst[i] += pe[i];
            }
        }

        normed.resize(static_cast<size_t>(S * H));
        q.resize(static_cast<size_t>(S * H));
        new_k.resize(static_cast<size_t>(S * H));
        new_v.resize(static_cast<size_t>(S * H));
        attn_out.resize(static_cast<size_t>(S * H));

        for (int layer = 0; layer < config.local_num_hidden_layers; ++layer) {
            const auto& wl = w.layers[static_cast<size_t>(layer)];
            auto& ops = layer_ops[static_cast<size_t>(layer)];
            auto& cache = caches[static_cast<size_t>(layer)];
            const int past = static_cast<int>(cache.k.size() / static_cast<size_t>(H));

            for (int s = 0; s < S; ++s) {
                rms_norm(
                    x.data() + static_cast<size_t>(s) * H,
                    wl.norm1.data(),
                    H,
                    config.rms_norm_eps,
                    normed.data() + static_cast<size_t>(s) * H);

                ops.qkv.run(normed.data() + static_cast<size_t>(s) * H, qkv_val);
                std::copy(qkv_val.begin(), qkv_val.begin() + H, q.begin() + static_cast<size_t>(s) * H);
                std::copy(qkv_val.begin() + H, qkv_val.begin() + 2 * H, new_k.begin() + static_cast<size_t>(s) * H);
                std::copy(qkv_val.begin() + 2 * H, qkv_val.end(), new_v.begin() + static_cast<size_t>(s) * H);

                for (int head = 0; head < nH; ++head) {
                    rms_norm(
                        q.data() + static_cast<size_t>(s * H + head * hd),
                        wl.q_norm.data(),
                        hd,
                        config.rms_norm_eps,
                        q.data() + static_cast<size_t>(s * H + head * hd));
                    rms_norm(
                        new_k.data() + static_cast<size_t>(s * H + head * hd),
                        wl.k_norm.data(),
                        hd,
                        config.rms_norm_eps,
                        new_k.data() + static_cast<size_t>(s * H + head * hd));
                }
            }

            cache.k.insert(cache.k.end(), new_k.begin(), new_k.begin() + static_cast<size_t>(S * H));
            cache.v.insert(cache.v.end(), new_v.begin(), new_v.begin() + static_cast<size_t>(S * H));

            std::fill(attn_out.begin(), attn_out.end(), 0.0f);
            for (int s = 0; s < S; ++s) {
                const int attend_count = past + s + 1;
                scores.resize(static_cast<size_t>(attend_count));
                for (int head = 0; head < nH; ++head) {
                    const float* q_ptr = q.data() + static_cast<size_t>(s * H + head * hd);
                    float max_score = -std::numeric_limits<float>::infinity();
                    for (int t = 0; t < attend_count; ++t) {
                        const float* k_ptr = cache.k.data() + static_cast<size_t>(t * H + head * hd);
                        float dot = 0.0f;
                        for (int d = 0; d < hd; ++d) {
                            dot += q_ptr[d] * k_ptr[d];
                        }
                        const float score = static_cast<float>(dot) * inv_sqrt_hd;
                        scores[static_cast<size_t>(t)] = score;
                        if (score > max_score) {
                            max_score = score;
                        }
                    }

                    double denom = 0.0;
                    for (float& score : scores) {
                        score = std::exp(score - max_score);
                        denom += score;
                    }
                    if (denom <= 0.0 || !std::isfinite(denom)) {
                        denom = 1.0;
                    }

                    float* out_ptr = attn_out.data() + static_cast<size_t>(s * H + head * hd);
                    for (int t = 0; t < attend_count; ++t) {
                        const float prob = scores[static_cast<size_t>(t)] / static_cast<float>(denom);
                        const float* v_ptr = cache.v.data() + static_cast<size_t>(t * H + head * hd);
                        for (int d = 0; d < hd; ++d) {
                            out_ptr[d] += prob * v_ptr[d];
                        }
                    }
                }
            }

            for (int s = 0; s < S; ++s) {
                ops.o_proj.run(attn_out.data() + static_cast<size_t>(s) * H, proj);
                float* x_ptr = x.data() + static_cast<size_t>(s) * H;
                for (int i = 0; i < H; ++i) {
                    x_ptr[i] += proj[static_cast<size_t>(i)];
                }

                rms_norm(x_ptr, wl.norm2.data(), H, config.rms_norm_eps, normed.data() + static_cast<size_t>(s) * H);
                if (ops.fuse_ffn) {
                    ops.ffn.run(normed.data() + static_cast<size_t>(s) * H, down);
                } else {
                    ops.ff_gate.run(normed.data() + static_cast<size_t>(s) * H, gate);
                    ops.ff_up.run(normed.data() + static_cast<size_t>(s) * H, up);
                    for (int i = 0; i < I; ++i) {
                        up[static_cast<size_t>(i)] *= silu(gate[static_cast<size_t>(i)]);
                    }
                    ops.ff_down.run(up.data(), down);
                }
                for (int i = 0; i < H; ++i) {
                    x_ptr[i] += down[static_cast<size_t>(i)];
                }
            }
        }

        output.resize(static_cast<size_t>(S * H));
        for (int s = 0; s < S; ++s) {
            rms_norm(
                x.data() + static_cast<size_t>(s) * H,
                w.final_norm.data(),
                H,
                config.rms_norm_eps,
                output.data() + static_cast<size_t>(s) * H);
        }
    }

    int64_t sample_channel(int ch, const float* vec, float temp, int top_k, float top_p, float rep_pen, std::vector<std::unordered_set<int>>& history) {
        const float* head = assets.audio_emb().data() + static_cast<int64_t>(ch) * config.audio_vocab_size * config.hidden_size;
        
        // Compute logits
        matvec_native(vec, head, config.hidden_size, config.audio_vocab_size, logits);
        
        std::unordered_set<int>* prev = history.empty() ? nullptr : &history[static_cast<size_t>(ch)];
        const int64_t code = sampler.sample_logits(logits, temp, top_k, top_p, rep_pen, prev);
        if (prev) {
            prev->insert(static_cast<int>(code));
        }
        return code;
    }
};

V3NativeAcoustic::V3NativeAcoustic(const V3NativeConfig& config, const V3NativeAssets& assets, V3NativeSampler& sampler)
    : impl_(std::make_unique<Impl>(config, assets, sampler)) {}

V3NativeAcoustic::~V3NativeAcoustic() = default;

bool V3NativeAcoustic::initialize(std::string& error) {
    try {
        impl_->initialize_ops();
        return true;
    } catch (const std::exception& e) {
        error = e.what();
        return false;
    }
}

bool V3NativeAcoustic::generate_frame(
    const std::vector<float>& h,
    float temperature,
    int top_k,
    float top_p,
    float repetition_penalty,
    std::vector<std::unordered_set<int>>& history,
    std::vector<int64_t>& codes,
    bool& eos,
    std::string& error) {
    try {
        const int H = impl_->config.hidden_size;
        impl_->caches.assign(static_cast<size_t>(impl_->config.local_num_hidden_layers), Impl::LayerCache{});
        
        impl_->token.resize(static_cast<size_t>(2 * H));
        std::copy(h.begin(), h.begin() + H, impl_->token.begin());
        const float* sgs = impl_->assets.text_emb().data() + impl_->config.speech_generation_start_token_id * H;
        std::copy(sgs, sgs + H, impl_->token.begin() + H);

        impl_->cached_step(impl_->token, {0, 1}, impl_->hidden);
        impl_->slot0.assign(impl_->hidden.begin(), impl_->hidden.begin() + H);

        codes.clear();
        codes.reserve(static_cast<size_t>(impl_->config.n_vq));
        codes.push_back(impl_->sample_channel(0, impl_->hidden.data() + H, temperature, top_k, top_p, repetition_penalty, history));

        for (int ch = 1; ch < impl_->config.n_vq; ++ch) {
            const int64_t prev_code = codes.back();
            const float* emb = impl_->assets.audio_emb().data() + (static_cast<int64_t>(ch - 1) * impl_->config.audio_vocab_size + prev_code) * H;
            impl_->token.assign(emb, emb + H);
            impl_->cached_step(impl_->token, {ch + 1}, impl_->hidden);
            codes.push_back(impl_->sample_channel(ch, impl_->hidden.data(), temperature, top_k, top_p, repetition_penalty, history));
        }

        matvec_native(impl_->slot0.data(), impl_->assets.text_emb().data(), H, impl_->config.text_vocab_size, impl_->text_logits);
        eos = static_cast<int>(std::distance(impl_->text_logits.begin(), std::max_element(impl_->text_logits.begin(), impl_->text_logits.end()))) ==
              impl_->config.speech_generation_end_token_id;

        return true;
    } catch (const std::exception& e) {
        error = e.what();
        return false;
    }
}
