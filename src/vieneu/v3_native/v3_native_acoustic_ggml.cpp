#include "v3_native_acoustic_ggml.h"
#include "ggml.h"
#include "ggml-cpu.h"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <limits>
#include <stdexcept>
#include <iostream>
#if defined(_OPENMP)
#include <omp.h>
#endif

namespace {

int get_ggml_threads(int fallback_threads) {
    const char* val = std::getenv("OMP_NUM_THREADS");
    if (!val || !*val) {
        return (std::max)(1, fallback_threads);
    }
    char* end = nullptr;
    const long parsed = std::strtol(val, &end, 10);
    if (end == val || parsed <= 0) {
        return (std::max)(1, fallback_threads);
    }
    return static_cast<int>(std::min<long>(parsed, 32));
}

bool env_flag_enabled(const char* name, bool default_value = false) {
    const char* value = std::getenv(name);
    if (!value || !*value) {
        return default_value;
    }
    std::string normalized(value);
    std::transform(normalized.begin(), normalized.end(), normalized.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    if (normalized == "0" || normalized == "false" || normalized == "no" || normalized == "off") {
        return false;
    }
    return normalized == "1" || normalized == "true" || normalized == "yes" || normalized == "on";
}

bool use_direct_linear_backend() {
    if (env_flag_enabled("VIENEU_ACOUSTIC_DIRECT_LINEAR", false)) {
        return true;
    }
    return !env_flag_enabled("VIENEU_ACOUSTIC_GGML_LINEAR", true);
}

using BenchClock = std::chrono::steady_clock;

struct ScopedBenchTimer {
    bool enabled = false;
    BenchClock::time_point start{};
    double* total_ms = nullptr;

    ScopedBenchTimer(bool enabled_, double& total)
        : enabled(enabled_), total_ms(&total) {
        if (enabled) {
            start = BenchClock::now();
        }
    }

    ~ScopedBenchTimer() {
        if (enabled && total_ms) {
            const auto end = BenchClock::now();
            *total_ms += std::chrono::duration<double, std::milli>(end - start).count();
        }
    }
};

struct AcousticBenchStats {
    double reset_cache_ms = 0.0;
    double slot_pos_ms = 0.0;
    double norm1_ms = 0.0;
    double qkv_ms = 0.0;
    double qk_norm_ms = 0.0;
    double cache_write_ms = 0.0;
    double attn_ms = 0.0;
    double o_proj_ms = 0.0;
    double norm2_ms = 0.0;
    double ffn_ms = 0.0;
    double residual_ms = 0.0;
    double final_norm_ms = 0.0;
    double cached_step_ms = 0.0;
    double sample_head_matvec_ms = 0.0;
    double sample_head_ggml_ms = 0.0;
    double sample_select_ms = 0.0;
    double eos_head_matvec_ms = 0.0;
    double eos_head_ggml_ms = 0.0;
    double frame_total_ms = 0.0;
    int cached_step_calls = 0;
    int cached_step_tokens = 0;
    int sample_calls = 0;
    int frame_calls = 0;
};

void linear_matvec_cpu(const float* weight, const float* input, int out_dim, int in_dim, int n_threads, float* output) {
    const bool use_parallel = n_threads > 1 && out_dim >= 64;
#if defined(_OPENMP)
#pragma omp parallel for if(use_parallel) num_threads(n_threads) schedule(static)
#endif
    for (int o = 0; o < out_dim; ++o) {
        const float* row = weight + static_cast<size_t>(o) * in_dim;
        float sum = 0.0f;
#if defined(_MSC_VER)
#pragma loop(ivdep)
#elif defined(__GNUC__) || defined(__clang__)
#pragma GCC ivdep
#endif
        for (int i = 0; i < in_dim; ++i) {
            sum += row[i] * input[i];
        }
        output[o] = sum;
    }
}

class GgmlLinear {
public:
    GgmlLinear() = default;
    ~GgmlLinear() { release(); }

    void initialize(ggml_backend_t backend, const float* weight, int out_dim, int in_dim, int n_threads) {
        release();
        (void)backend;
        thread_count_ = get_ggml_threads(n_threads);
        in_dim_ = in_dim;
        out_dim_ = out_dim;
        direct_weight_ = weight;
        use_direct_ = use_direct_linear_backend();
        if (use_direct_) {
            return;
        }

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
        if (use_direct_) {
            linear_matvec_cpu(direct_weight_, input, out_dim_, in_dim_, thread_count_, output.data());
            return;
        }
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
        direct_weight_ = nullptr;
        use_direct_ = true;
    }

    ggml_context* ctx_ = nullptr;
    ggml_tensor* weight_ = nullptr;
    ggml_tensor* input_ = nullptr;
    ggml_tensor* output_ = nullptr;
    ggml_cgraph* graph_ = nullptr;
    ggml_cplan plan_{};
    std::vector<uint8_t> work_data_;
    const float* direct_weight_ = nullptr;
    int in_dim_ = 0;
    int out_dim_ = 0;
    int thread_count_ = 1;
    bool use_direct_ = true;
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
                    int intermediate_dim,
                    int n_threads) {
        release();
        (void)backend;
        thread_count_ = get_ggml_threads(n_threads);
        hidden_dim_ = hidden_dim;
        intermediate_dim_ = intermediate_dim;
        gate_weight_direct_ = gate_weight;
        up_weight_direct_ = up_weight;
        down_weight_direct_ = down_weight;
        use_q8_ = env_flag_enabled("VIENEU_ACOUSTIC_Q8_FFN", true);
        const int64_t q8_block = ggml_blck_size(GGML_TYPE_Q8_0);
        if (use_q8_ && (hidden_dim_ % q8_block != 0 || intermediate_dim_ % q8_block != 0)) {
            use_q8_ = false;
        }

        use_direct_ = !use_q8_ && use_direct_linear_backend();
        if (use_direct_) {
            gate_direct_.resize(static_cast<size_t>(intermediate_dim_));
            up_direct_.resize(static_cast<size_t>(intermediate_dim_));
            fused_direct_.resize(static_cast<size_t>(intermediate_dim_));
            return;
        }

        const ggml_type weight_type = use_q8_ ? GGML_TYPE_Q8_0 : GGML_TYPE_F32;
        const size_t weight_bytes =
            2 * ggml_row_size(weight_type, hidden_dim_) * static_cast<size_t>(intermediate_dim_) +
            ggml_row_size(weight_type, intermediate_dim_) * static_cast<size_t>(hidden_dim_);
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

        gate_weight_ = ggml_new_tensor_2d(ctx_, weight_type, hidden_dim_, intermediate_dim_);
        up_weight_ = ggml_new_tensor_2d(ctx_, weight_type, hidden_dim_, intermediate_dim_);
        down_weight_ = ggml_new_tensor_2d(ctx_, weight_type, intermediate_dim_, hidden_dim_);
        input_ = ggml_new_tensor_1d(ctx_, GGML_TYPE_F32, hidden_dim_);

        ggml_tensor* gate = ggml_mul_mat(ctx_, gate_weight_, input_);
        ggml_tensor* up = ggml_mul_mat(ctx_, up_weight_, input_);
        ggml_tensor* activated = ggml_silu(ctx_, gate);
        ggml_tensor* fused = ggml_mul(ctx_, activated, up);
        output_ = ggml_mul_mat(ctx_, down_weight_, fused);

        graph_ = ggml_new_graph(ctx_);
        ggml_build_forward_expand(graph_, output_);

        if (use_q8_) {
            ggml_quantize_chunk(GGML_TYPE_Q8_0, gate_weight, gate_weight_->data, 0, intermediate_dim_, hidden_dim_, nullptr);
            ggml_quantize_chunk(GGML_TYPE_Q8_0, up_weight, up_weight_->data, 0, intermediate_dim_, hidden_dim_, nullptr);
            ggml_quantize_chunk(GGML_TYPE_Q8_0, down_weight, down_weight_->data, 0, hidden_dim_, intermediate_dim_, nullptr);
        } else {
            std::memcpy(gate_weight_->data, gate_weight, static_cast<size_t>(intermediate_dim_) * hidden_dim_ * sizeof(float));
            std::memcpy(up_weight_->data, up_weight, static_cast<size_t>(intermediate_dim_) * hidden_dim_ * sizeof(float));
            std::memcpy(down_weight_->data, down_weight, static_cast<size_t>(hidden_dim_) * intermediate_dim_ * sizeof(float));
        }
        plan_ = ggml_graph_plan(graph_, thread_count_, nullptr);
        work_data_.resize(plan_.work_size);
        plan_.work_data = work_data_.empty() ? nullptr : work_data_.data();
    }

    void run(const float* input, std::vector<float>& output) {
        output.resize(static_cast<size_t>(hidden_dim_));
        if (use_direct_) {
            linear_matvec_cpu(gate_weight_direct_, input, intermediate_dim_, hidden_dim_, thread_count_, gate_direct_.data());
            linear_matvec_cpu(up_weight_direct_, input, intermediate_dim_, hidden_dim_, thread_count_, up_direct_.data());
            for (int i = 0; i < intermediate_dim_; ++i) {
                fused_direct_[static_cast<size_t>(i)] = up_direct_[static_cast<size_t>(i)] * (gate_direct_[static_cast<size_t>(i)] / (1.0f + std::exp(-gate_direct_[static_cast<size_t>(i)])));
            }
            linear_matvec_cpu(down_weight_direct_, fused_direct_.data(), hidden_dim_, intermediate_dim_, thread_count_, output.data());
            return;
        }
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
        gate_weight_direct_ = nullptr;
        up_weight_direct_ = nullptr;
        down_weight_direct_ = nullptr;
        gate_direct_.clear();
        up_direct_.clear();
        fused_direct_.clear();
        use_q8_ = false;
        use_direct_ = true;
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
    const float* gate_weight_direct_ = nullptr;
    const float* up_weight_direct_ = nullptr;
    const float* down_weight_direct_ = nullptr;
    std::vector<float> gate_direct_;
    std::vector<float> up_direct_;
    std::vector<float> fused_direct_;
    int hidden_dim_ = 0;
    int intermediate_dim_ = 0;
    int thread_count_ = 1;
    bool use_q8_ = false;
    bool use_direct_ = true;
};

} // namespace

struct V3NativeAcoustic::Impl {
    const V3NativeConfig& config;
    const V3NativeAssets& assets;
    V3NativeSampler& sampler;
    int n_threads = 4;

    ggml_backend_t backend = nullptr;
    
    struct LayerCache {
        std::vector<float> k;
        std::vector<float> v;
        int used = 0;

        void initialize(int max_tokens, int hidden_size) {
            const size_t values = static_cast<size_t>(max_tokens) * hidden_size;
            k.resize(values);
            v.resize(values);
            used = 0;
        }

        void reset() {
            used = 0;
        }
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
    std::vector<GgmlLinear> audio_head_ops;
    GgmlLinear text_head_op;
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
    bool benchmark_enabled = false;
    bool use_ggml_heads = true;
    AcousticBenchStats bench;

    Impl(const V3NativeConfig& cfg, const V3NativeAssets& ast, V3NativeSampler& smpl, int threads)
        : config(cfg), assets(ast), sampler(smpl), n_threads((std::max)(1, threads)) {
        benchmark_enabled = env_flag_enabled("VIENEU_V3_NATIVE_BENCHMARK", false);
        backend = ggml_backend_cpu_init();
        if (!backend) {
            throw std::runtime_error("Failed to initialize GGML CPU backend.");
        }
        ggml_backend_cpu_set_n_threads(backend, get_ggml_threads(n_threads));
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
        const bool fuse = env_flag_enabled("VIENEU_GGML_FUSE_FFN", true);
        use_ggml_heads = env_flag_enabled("VIENEU_ACOUSTIC_GGML_HEADS", true);
        layer_ops.clear();
        layer_ops.resize(w.layers.size());
        for (size_t i = 0; i < w.layers.size(); ++i) {
            const auto& wl = w.layers[i];
            LayerOps& ops = layer_ops[i];
            ops.fuse_ffn = fuse;
            ops.qkv.initialize(backend, wl.qkv.data(), 3 * H, H, n_threads);
            ops.o_proj.initialize(backend, wl.o_proj.data(), H, H, n_threads);
            if (fuse) {
                ops.ffn.initialize(backend, wl.ff_gate.data(), wl.ff_up.data(), wl.ff_down.data(), H, I, n_threads);
            } else {
                ops.ff_gate.initialize(backend, wl.ff_gate.data(), I, H, n_threads);
                ops.ff_up.initialize(backend, wl.ff_up.data(), I, H, n_threads);
                ops.ff_down.initialize(backend, wl.ff_down.data(), H, I, n_threads);
            }
        }
        if (use_ggml_heads) {
            audio_head_ops.clear();
            audio_head_ops.resize(static_cast<size_t>(config.n_vq));
            for (int ch = 0; ch < config.n_vq; ++ch) {
                const float* head = assets.audio_emb().data() +
                    (static_cast<int64_t>(ch) * config.audio_vocab_size) * config.hidden_size;
                audio_head_ops[static_cast<size_t>(ch)].initialize(
                    backend,
                    head,
                    config.audio_vocab_size,
                    config.hidden_size,
                    n_threads);
            }
            text_head_op.initialize(
                backend,
                assets.text_emb().data(),
                config.text_vocab_size,
                config.hidden_size,
                n_threads);
        }

        const int max_local_tokens = config.n_vq + 2;
        caches.resize(static_cast<size_t>(config.local_num_hidden_layers));
        for (LayerCache& cache : caches) {
            cache.initialize(max_local_tokens, H);
        }

        token.reserve(static_cast<size_t>(2 * H));
        hidden.reserve(static_cast<size_t>(2 * H));
        slot0.resize(static_cast<size_t>(H));
        logits.reserve(static_cast<size_t>(config.audio_vocab_size));
        text_logits.reserve(static_cast<size_t>(config.text_vocab_size));
        x.reserve(static_cast<size_t>(2 * H));
        normed.reserve(static_cast<size_t>(2 * H));
        qkv_val.reserve(static_cast<size_t>(3 * H));
        q.reserve(static_cast<size_t>(2 * H));
        new_k.reserve(static_cast<size_t>(2 * H));
        new_v.reserve(static_cast<size_t>(2 * H));
        attn_out.reserve(static_cast<size_t>(2 * H));
        proj.reserve(static_cast<size_t>(H));
        gate.reserve(static_cast<size_t>(I));
        up.reserve(static_cast<size_t>(I));
        down.reserve(static_cast<size_t>(H));
        scores.reserve(static_cast<size_t>(max_local_tokens));
    }

    void reset_caches() {
        ScopedBenchTimer timer(benchmark_enabled, bench.reset_cache_ms);
        for (LayerCache& cache : caches) {
            cache.reset();
        }
    }

    void cached_step(const std::vector<float>& input, const int* positions, int S, std::vector<float>& output) {
        ScopedBenchTimer step_timer(benchmark_enabled, bench.cached_step_ms);
        if (benchmark_enabled) {
            bench.cached_step_calls += 1;
            bench.cached_step_tokens += S;
        }
        const auto& w = assets.acoustic_weights();
        const int H = config.hidden_size;
        const int nH = config.local_num_attention_heads;
        const int hd = H / nH;
        const int I = config.local_intermediate_size;
        const float inv_sqrt_hd = 1.0f / std::sqrt(static_cast<float>(hd));

        x.resize(static_cast<size_t>(S * H));
        std::copy(input.begin(), input.begin() + static_cast<size_t>(S * H), x.begin());
        {
            ScopedBenchTimer timer(benchmark_enabled, bench.slot_pos_ms);
            for (int s = 0; s < S; ++s) {
                const int pos = positions[static_cast<size_t>(s)];
                const float* pe = w.slot_pos_emb.data() + static_cast<size_t>(pos) * H;
                float* dst = x.data() + static_cast<size_t>(s) * H;
                for (int i = 0; i < H; ++i) {
                    dst[i] += pe[i];
                }
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
            const int past = cache.used;

            for (int s = 0; s < S; ++s) {
                {
                    ScopedBenchTimer timer(benchmark_enabled, bench.norm1_ms);
                    rms_norm(
                        x.data() + static_cast<size_t>(s) * H,
                        wl.norm1.data(),
                        H,
                        config.rms_norm_eps,
                        normed.data() + static_cast<size_t>(s) * H);
                }

                {
                    ScopedBenchTimer timer(benchmark_enabled, bench.qkv_ms);
                    ops.qkv.run(normed.data() + static_cast<size_t>(s) * H, qkv_val);
                }
                std::copy(qkv_val.begin(), qkv_val.begin() + H, q.begin() + static_cast<size_t>(s) * H);
                std::copy(qkv_val.begin() + H, qkv_val.begin() + 2 * H, new_k.begin() + static_cast<size_t>(s) * H);
                std::copy(qkv_val.begin() + 2 * H, qkv_val.end(), new_v.begin() + static_cast<size_t>(s) * H);

                {
                    ScopedBenchTimer timer(benchmark_enabled, bench.qk_norm_ms);
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
            }

            const int new_used = past + S;
            {
                ScopedBenchTimer timer(benchmark_enabled, bench.cache_write_ms);
                const size_t required_cache_values = static_cast<size_t>(new_used) * H;
                if (cache.k.size() < required_cache_values) {
                    cache.k.resize(required_cache_values);
                    cache.v.resize(required_cache_values);
                }
                for (int s = 0; s < S; ++s) {
                    const size_t dst = static_cast<size_t>(past + s) * H;
                    const size_t src = static_cast<size_t>(s) * H;
                    std::copy(new_k.begin() + src, new_k.begin() + src + H, cache.k.begin() + dst);
                    std::copy(new_v.begin() + src, new_v.begin() + src + H, cache.v.begin() + dst);
                }
            }
            cache.used = new_used;

            {
                ScopedBenchTimer timer(benchmark_enabled, bench.attn_ms);
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
            }

            for (int s = 0; s < S; ++s) {
                {
                    ScopedBenchTimer timer(benchmark_enabled, bench.o_proj_ms);
                    ops.o_proj.run(attn_out.data() + static_cast<size_t>(s) * H, proj);
                }
                float* x_ptr = x.data() + static_cast<size_t>(s) * H;
                {
                    ScopedBenchTimer timer(benchmark_enabled, bench.residual_ms);
                    for (int i = 0; i < H; ++i) {
                        x_ptr[i] += proj[static_cast<size_t>(i)];
                    }
                }

                {
                    ScopedBenchTimer timer(benchmark_enabled, bench.norm2_ms);
                    rms_norm(x_ptr, wl.norm2.data(), H, config.rms_norm_eps, normed.data() + static_cast<size_t>(s) * H);
                }
                if (ops.fuse_ffn) {
                    ScopedBenchTimer timer(benchmark_enabled, bench.ffn_ms);
                    ops.ffn.run(normed.data() + static_cast<size_t>(s) * H, down);
                } else {
                    ScopedBenchTimer timer(benchmark_enabled, bench.ffn_ms);
                    ops.ff_gate.run(normed.data() + static_cast<size_t>(s) * H, gate);
                    ops.ff_up.run(normed.data() + static_cast<size_t>(s) * H, up);
                    for (int i = 0; i < I; ++i) {
                        up[static_cast<size_t>(i)] *= silu(gate[static_cast<size_t>(i)]);
                    }
                    ops.ff_down.run(up.data(), down);
                }
                {
                    ScopedBenchTimer timer(benchmark_enabled, bench.residual_ms);
                    for (int i = 0; i < H; ++i) {
                        x_ptr[i] += down[static_cast<size_t>(i)];
                    }
                }
            }
        }

        output.resize(static_cast<size_t>(S * H));
        {
            ScopedBenchTimer timer(benchmark_enabled, bench.final_norm_ms);
            for (int s = 0; s < S; ++s) {
                rms_norm(
                    x.data() + static_cast<size_t>(s) * H,
                    w.final_norm.data(),
                    H,
                    config.rms_norm_eps,
                    output.data() + static_cast<size_t>(s) * H);
            }
        }
    }

    int64_t sample_channel(int ch, const float* vec, float temp, int top_k, float top_p, float rep_pen, std::vector<V3RepetitionHistory>& history) {
        if (benchmark_enabled) {
            bench.sample_calls += 1;
        }
        if (use_ggml_heads) {
            ScopedBenchTimer timer(benchmark_enabled, bench.sample_head_ggml_ms);
            audio_head_ops[static_cast<size_t>(ch)].run(vec, logits);
        } else {
            const float* head = assets.audio_emb_t().data() + static_cast<int64_t>(ch) * config.hidden_size * config.audio_vocab_size;
            ScopedBenchTimer timer(benchmark_enabled, bench.sample_head_matvec_ms);
            matvec_transposed_native(vec, head, config.hidden_size, config.audio_vocab_size, logits);
        }
        
        V3RepetitionHistory* prev = history.empty() ? nullptr : &history[static_cast<size_t>(ch)];
        int64_t code = 0;
        {
            ScopedBenchTimer timer(benchmark_enabled, bench.sample_select_ms);
            code = sampler.sample_logits(logits, temp, top_k, top_p, rep_pen, prev);
        }
        if (prev) {
            prev->add(static_cast<int32_t>(code));
        }
        return code;
    }
};

V3NativeAcoustic::V3NativeAcoustic(const V3NativeConfig& config, const V3NativeAssets& assets, V3NativeSampler& sampler, int n_threads)
    : impl_(std::make_unique<Impl>(config, assets, sampler, n_threads)) {}

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

void V3NativeAcoustic::reset_benchmark_stats() {
    if (impl_) {
        impl_->bench = AcousticBenchStats{};
    }
}

void V3NativeAcoustic::print_benchmark_stats() const {
    if (!impl_ || !impl_->benchmark_enabled) {
        return;
    }

    const AcousticBenchStats& b = impl_->bench;
    const auto avg = [](double total, int calls) {
        return calls > 0 ? total / static_cast<double>(calls) : 0.0;
    };
    const auto pct = [&](double value) {
        return b.frame_total_ms > 0.0 ? (100.0 * value / b.frame_total_ms) : 0.0;
    };
    const double measured_parts =
        b.reset_cache_ms + b.cached_step_ms + b.sample_head_matvec_ms +
        b.sample_head_ggml_ms + b.sample_select_ms + b.eos_head_matvec_ms +
        b.eos_head_ggml_ms;

    std::cerr << "[V3NativeAcoustic] Detailed benchmark\n"
              << "  frames: calls=" << b.frame_calls
              << ", total=" << b.frame_total_ms << " ms"
              << ", avg=" << avg(b.frame_total_ms, b.frame_calls) << " ms/frame\n"
              << "  reset_cache: total=" << b.reset_cache_ms << " ms"
              << " (" << pct(b.reset_cache_ms) << "%)\n"
              << "  cached_step: total=" << b.cached_step_ms << " ms"
              << " (" << pct(b.cached_step_ms) << "%)"
              << ", calls=" << b.cached_step_calls
              << ", tokens=" << b.cached_step_tokens
              << ", avg_call=" << avg(b.cached_step_ms, b.cached_step_calls) << " ms\n"
              << "    slot_pos=" << b.slot_pos_ms << " ms"
              << ", norm1=" << b.norm1_ms << " ms"
              << ", qkv=" << b.qkv_ms << " ms"
              << ", qk_norm=" << b.qk_norm_ms << " ms\n"
              << "    cache_write=" << b.cache_write_ms << " ms"
              << ", attn=" << b.attn_ms << " ms"
              << ", o_proj=" << b.o_proj_ms << " ms"
              << ", norm2=" << b.norm2_ms << " ms\n"
              << "    ffn=" << b.ffn_ms << " ms"
              << ", residual=" << b.residual_ms << " ms"
              << ", final_norm=" << b.final_norm_ms << " ms\n"
              << "  sample_head_matvec: total=" << b.sample_head_matvec_ms << " ms"
              << " (" << pct(b.sample_head_matvec_ms) << "%)"
              << ", calls=" << b.sample_calls
              << ", avg=" << avg(b.sample_head_matvec_ms, b.sample_calls) << " ms\n"
              << "  sample_head_ggml: total=" << b.sample_head_ggml_ms << " ms"
              << " (" << pct(b.sample_head_ggml_ms) << "%)"
              << ", calls=" << b.sample_calls
              << ", avg=" << avg(b.sample_head_ggml_ms, b.sample_calls) << " ms\n"
              << "  sample_select: total=" << b.sample_select_ms << " ms"
              << " (" << pct(b.sample_select_ms) << "%)"
              << ", calls=" << b.sample_calls
              << ", avg=" << avg(b.sample_select_ms, b.sample_calls) << " ms\n"
              << "  eos_head_matvec: total=" << b.eos_head_matvec_ms << " ms"
              << " (" << pct(b.eos_head_matvec_ms) << "%)"
              << ", avg=" << avg(b.eos_head_matvec_ms, b.frame_calls) << " ms/frame\n"
              << "  eos_head_ggml: total=" << b.eos_head_ggml_ms << " ms"
              << " (" << pct(b.eos_head_ggml_ms) << "%)"
              << ", avg=" << avg(b.eos_head_ggml_ms, b.frame_calls) << " ms/frame\n"
              << "  unaccounted/overhead: " << (b.frame_total_ms - measured_parts) << " ms\n";
}

bool V3NativeAcoustic::generate_frame(
    const std::vector<float>& h,
    float temperature,
    int top_k,
    float top_p,
    float repetition_penalty,
    std::vector<V3RepetitionHistory>& history,
    std::vector<int64_t>& codes,
    bool& eos,
    std::string& error) {
    try {
        ScopedBenchTimer frame_timer(impl_->benchmark_enabled, impl_->bench.frame_total_ms);
        if (impl_->benchmark_enabled) {
            impl_->bench.frame_calls += 1;
        }
        const int H = impl_->config.hidden_size;
        impl_->reset_caches();
        
        impl_->token.resize(static_cast<size_t>(2 * H));
        std::copy(h.begin(), h.begin() + H, impl_->token.begin());
        const float* sgs = impl_->assets.text_emb().data() + impl_->config.speech_generation_start_token_id * H;
        std::copy(sgs, sgs + H, impl_->token.begin() + H);

        const int initial_positions[2] = {0, 1};
        impl_->cached_step(impl_->token, initial_positions, 2, impl_->hidden);
        std::copy(impl_->hidden.begin(), impl_->hidden.begin() + H, impl_->slot0.begin());

        codes.clear();
        codes.reserve(static_cast<size_t>(impl_->config.n_vq));
        codes.push_back(impl_->sample_channel(0, impl_->hidden.data() + H, temperature, top_k, top_p, repetition_penalty, history));

        for (int ch = 1; ch < impl_->config.n_vq; ++ch) {
            const int64_t prev_code = codes.back();
            const float* emb = impl_->assets.audio_emb().data() + (static_cast<int64_t>(ch - 1) * impl_->config.audio_vocab_size + prev_code) * H;
            impl_->token.resize(static_cast<size_t>(H));
            std::copy(emb, emb + H, impl_->token.begin());
            const int step_position = ch + 1;
            impl_->cached_step(impl_->token, &step_position, 1, impl_->hidden);
            codes.push_back(impl_->sample_channel(ch, impl_->hidden.data(), temperature, top_k, top_p, repetition_penalty, history));
        }

        {
            if (impl_->use_ggml_heads) {
                ScopedBenchTimer timer(impl_->benchmark_enabled, impl_->bench.eos_head_ggml_ms);
                impl_->text_head_op.run(impl_->slot0.data(), impl_->text_logits);
            } else {
                ScopedBenchTimer timer(impl_->benchmark_enabled, impl_->bench.eos_head_matvec_ms);
                matvec_transposed_native(impl_->slot0.data(), impl_->assets.text_emb_t().data(), H, impl_->config.text_vocab_size, impl_->text_logits);
            }
        }
        eos = static_cast<int>(std::distance(impl_->text_logits.begin(), std::max_element(impl_->text_logits.begin(), impl_->text_logits.end()))) ==
              impl_->config.speech_generation_end_token_id;

        return true;
    } catch (const std::exception& e) {
        error = e.what();
        return false;
    }
}
