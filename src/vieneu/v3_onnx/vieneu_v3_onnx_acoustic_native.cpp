#include "../vieneu_v3_onnx.h"
#include "vieneu_v3_onnx_internal.h"

#include "ggml.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
#if defined(_OPENMP)
#include <omp.h>
#endif

namespace {

int ggml_thread_count_from_env(int fallback_threads) {
    const char* value = std::getenv("OMP_NUM_THREADS");
    if (!value || !*value) {
        return (std::max)(1, fallback_threads);
    }
    char* end = nullptr;
    const long parsed = std::strtol(value, &end, 10);
    if (end == value || parsed <= 0) {
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

class GgmlLinearOp {
public:
    GgmlLinearOp() = default;

    GgmlLinearOp(ggml_backend_t backend, const float* weight, int out_dim, int in_dim, int n_threads, const char* name) {
        initialize(backend, weight, out_dim, in_dim, n_threads, name);
    }

    GgmlLinearOp(const GgmlLinearOp&) = delete;
    GgmlLinearOp& operator=(const GgmlLinearOp&) = delete;

    GgmlLinearOp(GgmlLinearOp&& other) noexcept {
        move_from(other);
    }

    GgmlLinearOp& operator=(GgmlLinearOp&& other) noexcept {
        if (this != &other) {
            release();
            move_from(other);
        }
        return *this;
    }

    ~GgmlLinearOp() {
        release();
    }

    void initialize(ggml_backend_t backend, const float* weight, int out_dim, int in_dim, int n_threads, const char* name) {
        release();
        if (!backend) {
            throw std::runtime_error("ggml backend is not initialized");
        }
        (void) backend;
        thread_count_ = ggml_thread_count_from_env(n_threads);
        in_dim_ = in_dim;
        out_dim_ = out_dim;
        name_ = name ? name : "linear";
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
            throw std::runtime_error("failed to initialize ggml context for " + name_);
        }

        // PyTorch stores Linear weight as [out_dim, in_dim]. ggml_mul_mat expects
        // the contiguous dimension to be input columns, so [in_dim, out_dim] maps directly.
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
            throw std::runtime_error("ggml graph compute failed for " + name_ + ": " + ggml_status_to_string(status));
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
        name_.clear();
        direct_weight_ = nullptr;
        use_direct_ = true;
    }

    void move_from(GgmlLinearOp& other) noexcept {
        ctx_ = other.ctx_;
        weight_ = other.weight_;
        input_ = other.input_;
        output_ = other.output_;
        graph_ = other.graph_;
        plan_ = other.plan_;
        work_data_ = std::move(other.work_data_);
        if (!work_data_.empty()) {
            plan_.work_data = work_data_.data();
        }
        direct_weight_ = other.direct_weight_;
        in_dim_ = other.in_dim_;
        out_dim_ = other.out_dim_;
        thread_count_ = other.thread_count_;
        name_ = std::move(other.name_);
        use_direct_ = other.use_direct_;

        other.ctx_ = nullptr;
        other.weight_ = nullptr;
        other.input_ = nullptr;
        other.output_ = nullptr;
        other.graph_ = nullptr;
        other.in_dim_ = 0;
        other.out_dim_ = 0;
        other.thread_count_ = 1;
        other.work_data_.clear();
        other.direct_weight_ = nullptr;
        other.use_direct_ = true;
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
    std::string name_;
    bool use_direct_ = true;
};

class GgmlFfnOp {
public:
    GgmlFfnOp() = default;

    GgmlFfnOp(const GgmlFfnOp&) = delete;
    GgmlFfnOp& operator=(const GgmlFfnOp&) = delete;

    GgmlFfnOp(GgmlFfnOp&& other) noexcept {
        move_from(other);
    }

    GgmlFfnOp& operator=(GgmlFfnOp&& other) noexcept {
        if (this != &other) {
            release();
            move_from(other);
        }
        return *this;
    }

    ~GgmlFfnOp() {
        release();
    }

    void initialize(ggml_backend_t backend,
                    const float* gate_weight,
                    const float* up_weight,
                    const float* down_weight,
                    int hidden_dim,
                    int intermediate_dim,
                    int n_threads,
                    const char* name) {
        release();
        if (!backend) {
            throw std::runtime_error("ggml backend is not initialized");
        }
        (void) backend;
        thread_count_ = ggml_thread_count_from_env(n_threads);
        hidden_dim_ = hidden_dim;
        intermediate_dim_ = intermediate_dim;
        name_ = name ? name : "ffn";
        gate_weight_direct_ = gate_weight;
        up_weight_direct_ = up_weight;
        down_weight_direct_ = down_weight;
        use_direct_ = use_direct_linear_backend();
        if (use_direct_) {
            gate_direct_.resize(static_cast<size_t>(intermediate_dim_));
            up_direct_.resize(static_cast<size_t>(intermediate_dim_));
            fused_direct_.resize(static_cast<size_t>(intermediate_dim_));
            return;
        }

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
            throw std::runtime_error("failed to initialize ggml context for " + name_);
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
        if (use_direct_) {
            linear_matvec_cpu(gate_weight_direct_, input, intermediate_dim_, hidden_dim_, thread_count_, gate_direct_.data());
            linear_matvec_cpu(up_weight_direct_, input, intermediate_dim_, hidden_dim_, thread_count_, up_direct_.data());
            for (int i = 0; i < intermediate_dim_; ++i) {
                const float gate = gate_direct_[static_cast<size_t>(i)];
                fused_direct_[static_cast<size_t>(i)] = up_direct_[static_cast<size_t>(i)] * (gate / (1.0f + std::exp(-gate)));
            }
            linear_matvec_cpu(down_weight_direct_, fused_direct_.data(), hidden_dim_, intermediate_dim_, thread_count_, output.data());
            return;
        }
        std::memcpy(input_->data, input, static_cast<size_t>(hidden_dim_) * sizeof(float));
        const ggml_status status = ggml_graph_compute(graph_, &plan_);
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("ggml graph compute failed for " + name_ + ": " + ggml_status_to_string(status));
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
        name_.clear();
        gate_weight_direct_ = nullptr;
        up_weight_direct_ = nullptr;
        down_weight_direct_ = nullptr;
        gate_direct_.clear();
        up_direct_.clear();
        fused_direct_.clear();
        use_direct_ = true;
    }

    void move_from(GgmlFfnOp& other) noexcept {
        ctx_ = other.ctx_;
        gate_weight_ = other.gate_weight_;
        up_weight_ = other.up_weight_;
        down_weight_ = other.down_weight_;
        input_ = other.input_;
        output_ = other.output_;
        graph_ = other.graph_;
        plan_ = other.plan_;
        work_data_ = std::move(other.work_data_);
        if (!work_data_.empty()) {
            plan_.work_data = work_data_.data();
        }
        gate_weight_direct_ = other.gate_weight_direct_;
        up_weight_direct_ = other.up_weight_direct_;
        down_weight_direct_ = other.down_weight_direct_;
        gate_direct_ = std::move(other.gate_direct_);
        up_direct_ = std::move(other.up_direct_);
        fused_direct_ = std::move(other.fused_direct_);
        hidden_dim_ = other.hidden_dim_;
        intermediate_dim_ = other.intermediate_dim_;
        thread_count_ = other.thread_count_;
        name_ = std::move(other.name_);
        use_direct_ = other.use_direct_;

        other.ctx_ = nullptr;
        other.gate_weight_ = nullptr;
        other.up_weight_ = nullptr;
        other.down_weight_ = nullptr;
        other.input_ = nullptr;
        other.output_ = nullptr;
        other.graph_ = nullptr;
        other.hidden_dim_ = 0;
        other.intermediate_dim_ = 0;
        other.thread_count_ = 1;
        other.work_data_.clear();
        other.gate_weight_direct_ = nullptr;
        other.up_weight_direct_ = nullptr;
        other.down_weight_direct_ = nullptr;
        other.gate_direct_.clear();
        other.up_direct_.clear();
        other.fused_direct_.clear();
        other.use_direct_ = true;
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
    std::string name_;
    bool use_direct_ = true;
};

} // namespace

class VieneuV3OnnxEngine::NativeAcousticExecutor final : public VieneuV3OnnxEngine::AcousticExecutor {
public:
    explicit NativeAcousticExecutor(VieneuV3OnnxEngine& engine) : engine_(engine) {
        backend_ = ggml_backend_cpu_init();
        if (!backend_) {
            throw std::runtime_error("failed to initialize ggml CPU backend");
        }
        ggml_backend_cpu_set_n_threads(backend_, ggml_thread_count_from_env(engine_.threads_to_use_));
        fuse_ffn_ = env_flag_enabled("VIENEU_GGML_FUSE_FFN", true);
        initialize_ops();
    }

    ~NativeAcousticExecutor() override {
        layer_ops_.clear();
        if (backend_) {
            ggml_backend_free(backend_);
            backend_ = nullptr;
        }
    }

    const char* backend_name() const override {
        return "ggml";
    }

    bool generate_frame(const std::vector<float>& h,
                        float temperature,
                        int top_k,
                        float top_p,
                        float repetition_penalty,
                        std::vector<V3RepetitionHistory>& history,
                        std::vector<int64_t>& codes,
                        bool& eos,
                        std::string& error) override {
        const auto frame_start = engine_.benchmark_enabled_ ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
        try {
            generate_frame_impl(h, temperature, top_k, top_p, repetition_penalty, history, codes, eos);
            if (engine_.benchmark_enabled_) {
                const auto frame_end = std::chrono::steady_clock::now();
                engine_.benchmark_stats_.acoustic_frame_ms += std::chrono::duration<double, std::milli>(frame_end - frame_start).count();
                engine_.benchmark_stats_.acoustic_frame_calls += 1;
            }
            return true;
        } catch (const std::exception& e) {
            if (engine_.benchmark_enabled_) {
                const auto frame_end = std::chrono::steady_clock::now();
                engine_.benchmark_stats_.acoustic_frame_ms += std::chrono::duration<double, std::milli>(frame_end - frame_start).count();
                engine_.benchmark_stats_.acoustic_frame_calls += 1;
            }
            error = std::string("VieNeu v3 native acoustic frame failed: ") + e.what();
            return false;
        }
    }

private:
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
        GgmlLinearOp qkv;
        GgmlLinearOp o_proj;
        GgmlLinearOp ff_gate;
        GgmlLinearOp ff_up;
        GgmlLinearOp ff_down;
        GgmlFfnOp ffn;
    };

    VieneuV3OnnxEngine& engine_;
    ggml_backend_t backend_ = nullptr;
    bool fuse_ffn_ = false;
    std::vector<LayerOps> layer_ops_;
    std::vector<LayerCache> caches_;
    std::vector<float> token_;
    std::vector<float> hidden_;
    std::vector<float> slot0_;
    std::vector<float> logits_;
    std::vector<float> text_logits_;
    std::vector<float> x_;
    std::vector<float> normed_;
    std::vector<float> qkv_;
    std::vector<float> q_;
    std::vector<float> new_k_;
    std::vector<float> new_v_;
    std::vector<float> attn_out_;
    std::vector<float> proj_;
    std::vector<float> gate_;
    std::vector<float> up_;
    std::vector<float> down_;
    std::vector<float> scores_;

    static float silu(float x) {
        return x / (1.0f + std::exp(-x));
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
        const auto& cfg = engine_.config_;
        const auto& weights = engine_.acoustic_weights_;
        const int H = cfg.hidden_size;
        const int I = cfg.local_intermediate_size;
        const int n_threads = engine_.threads_to_use_;
        layer_ops_.clear();
        layer_ops_.reserve(weights.layers.size());
        for (size_t i = 0; i < weights.layers.size(); ++i) {
            const AcousticLayerWeights& w = weights.layers[i];
            const std::string prefix = "acoustic.layer." + std::to_string(i) + ".";
            LayerOps ops;
            ops.qkv.initialize(backend_, w.qkv.data(), 3 * H, H, n_threads, (prefix + "qkv").c_str());
            ops.o_proj.initialize(backend_, w.o_proj.data(), H, H, n_threads, (prefix + "o_proj").c_str());
            if (fuse_ffn_) {
                ops.ffn.initialize(backend_, w.ff_gate.data(), w.ff_up.data(), w.ff_down.data(), H, I, n_threads, (prefix + "ffn").c_str());
            } else {
                ops.ff_gate.initialize(backend_, w.ff_gate.data(), I, H, n_threads, (prefix + "ff_gate").c_str());
                ops.ff_up.initialize(backend_, w.ff_up.data(), I, H, n_threads, (prefix + "ff_up").c_str());
                ops.ff_down.initialize(backend_, w.ff_down.data(), H, I, n_threads, (prefix + "ff_down").c_str());
            }
            layer_ops_.push_back(std::move(ops));
        }

        const int max_local_tokens = cfg.n_vq + 2;
        caches_.resize(static_cast<size_t>(cfg.local_num_hidden_layers));
        for (LayerCache& cache : caches_) {
            cache.initialize(max_local_tokens, H);
        }

        token_.reserve(static_cast<size_t>(2 * H));
        hidden_.reserve(static_cast<size_t>(2 * H));
        slot0_.resize(static_cast<size_t>(H));
        logits_.reserve(static_cast<size_t>(cfg.audio_vocab_size));
        text_logits_.reserve(static_cast<size_t>(cfg.text_vocab_size));
        x_.reserve(static_cast<size_t>(2 * H));
        normed_.reserve(static_cast<size_t>(2 * H));
        qkv_.reserve(static_cast<size_t>(3 * H));
        q_.reserve(static_cast<size_t>(2 * H));
        new_k_.reserve(static_cast<size_t>(2 * H));
        new_v_.reserve(static_cast<size_t>(2 * H));
        attn_out_.reserve(static_cast<size_t>(2 * H));
        proj_.reserve(static_cast<size_t>(H));
        gate_.reserve(static_cast<size_t>(I));
        up_.reserve(static_cast<size_t>(I));
        down_.reserve(static_cast<size_t>(H));
        scores_.reserve(static_cast<size_t>(max_local_tokens));
    }

    void reset_caches() {
        for (LayerCache& cache : caches_) {
            cache.reset();
        }
    }

    void cached_step(const std::vector<float>& input, const int* positions, int S, std::vector<float>& output) {
        const auto& cfg = engine_.config_;
        const auto& weights = engine_.acoustic_weights_;
        const int H = cfg.hidden_size;
        const int nH = cfg.local_num_attention_heads;
        const int hd = H / nH;
        const int I = cfg.local_intermediate_size;
        const float inv_sqrt_hd = 1.0f / std::sqrt(static_cast<float>(hd));

        x_.resize(static_cast<size_t>(S * H));
        std::copy(input.begin(), input.begin() + static_cast<size_t>(S * H), x_.begin());
        for (int s = 0; s < S; ++s) {
            const int pos = positions[static_cast<size_t>(s)];
            if (pos < 0 || pos > cfg.n_vq) {
                throw std::runtime_error("native acoustic position id is out of range");
            }
            const float* pe = weights.slot_pos_emb.data() + static_cast<size_t>(pos) * H;
            float* dst = x_.data() + static_cast<size_t>(s) * H;
            for (int i = 0; i < H; ++i) {
                dst[i] += pe[i];
            }
        }

        normed_.resize(static_cast<size_t>(S * H));
        q_.resize(static_cast<size_t>(S * H));
        new_k_.resize(static_cast<size_t>(S * H));
        new_v_.resize(static_cast<size_t>(S * H));
        attn_out_.resize(static_cast<size_t>(S * H));

        for (int layer = 0; layer < cfg.local_num_hidden_layers; ++layer) {
            const AcousticLayerWeights& w = weights.layers[static_cast<size_t>(layer)];
            LayerOps& ops = layer_ops_[static_cast<size_t>(layer)];
            LayerCache& cache = caches_[static_cast<size_t>(layer)];
            const int past = cache.used;

            for (int s = 0; s < S; ++s) {
                rms_norm(
                    x_.data() + static_cast<size_t>(s) * H,
                    w.norm1.data(),
                    H,
                    cfg.rms_norm_eps,
                    normed_.data() + static_cast<size_t>(s) * H);

                ops.qkv.run(normed_.data() + static_cast<size_t>(s) * H, qkv_);
                std::copy(qkv_.begin(), qkv_.begin() + H, q_.begin() + static_cast<size_t>(s) * H);
                std::copy(qkv_.begin() + H, qkv_.begin() + 2 * H, new_k_.begin() + static_cast<size_t>(s) * H);
                std::copy(qkv_.begin() + 2 * H, qkv_.end(), new_v_.begin() + static_cast<size_t>(s) * H);

                for (int head = 0; head < nH; ++head) {
                    rms_norm(
                        q_.data() + static_cast<size_t>(s * H + head * hd),
                        w.q_norm.data(),
                        hd,
                        cfg.rms_norm_eps,
                        q_.data() + static_cast<size_t>(s * H + head * hd));
                    rms_norm(
                        new_k_.data() + static_cast<size_t>(s * H + head * hd),
                        w.k_norm.data(),
                        hd,
                        cfg.rms_norm_eps,
                        new_k_.data() + static_cast<size_t>(s * H + head * hd));
                }
            }

            const int new_used = past + S;
            const size_t required_cache_values = static_cast<size_t>(new_used) * H;
            if (cache.k.size() < required_cache_values) {
                cache.k.resize(required_cache_values);
                cache.v.resize(required_cache_values);
            }
            for (int s = 0; s < S; ++s) {
                const size_t dst = static_cast<size_t>(past + s) * H;
                const size_t src = static_cast<size_t>(s) * H;
                std::copy(new_k_.begin() + src, new_k_.begin() + src + H, cache.k.begin() + dst);
                std::copy(new_v_.begin() + src, new_v_.begin() + src + H, cache.v.begin() + dst);
            }
            cache.used = new_used;

            std::fill(attn_out_.begin(), attn_out_.end(), 0.0f);
            for (int s = 0; s < S; ++s) {
                const int attend_count = past + s + 1;
                scores_.resize(static_cast<size_t>(attend_count));
                for (int head = 0; head < nH; ++head) {
                    const float* q_ptr = q_.data() + static_cast<size_t>(s * H + head * hd);
                    float max_score = -std::numeric_limits<float>::infinity();
                    for (int t = 0; t < attend_count; ++t) {
                        const float* k_ptr = cache.k.data() + static_cast<size_t>(t * H + head * hd);
                        float dot = 0.0f;
                        for (int d = 0; d < hd; ++d) {
                            dot += q_ptr[d] * k_ptr[d];
                        }
                        const float score = static_cast<float>(dot) * inv_sqrt_hd;
                        scores_[static_cast<size_t>(t)] = score;
                        if (score > max_score) {
                            max_score = score;
                        }
                    }

                    double denom = 0.0;
                    for (float& score : scores_) {
                        score = std::exp(score - max_score);
                        denom += score;
                    }
                    if (denom <= 0.0 || !std::isfinite(denom)) {
                        denom = 1.0;
                    }

                    float* out_ptr = attn_out_.data() + static_cast<size_t>(s * H + head * hd);
                    for (int t = 0; t < attend_count; ++t) {
                        const float prob = scores_[static_cast<size_t>(t)] / static_cast<float>(denom);
                        const float* v_ptr = cache.v.data() + static_cast<size_t>(t * H + head * hd);
                        for (int d = 0; d < hd; ++d) {
                            out_ptr[d] += prob * v_ptr[d];
                        }
                    }
                }
            }

            for (int s = 0; s < S; ++s) {
                ops.o_proj.run(attn_out_.data() + static_cast<size_t>(s) * H, proj_);
                float* x_ptr = x_.data() + static_cast<size_t>(s) * H;
                for (int i = 0; i < H; ++i) {
                    x_ptr[i] += proj_[static_cast<size_t>(i)];
                }

                rms_norm(x_ptr, w.norm2.data(), H, cfg.rms_norm_eps, normed_.data() + static_cast<size_t>(s) * H);
                if (fuse_ffn_) {
                    ops.ffn.run(normed_.data() + static_cast<size_t>(s) * H, down_);
                } else {
                    ops.ff_gate.run(normed_.data() + static_cast<size_t>(s) * H, gate_);
                    ops.ff_up.run(normed_.data() + static_cast<size_t>(s) * H, up_);
                    for (int i = 0; i < I; ++i) {
                        up_[static_cast<size_t>(i)] *= silu(gate_[static_cast<size_t>(i)]);
                    }
                    ops.ff_down.run(up_.data(), down_);
                }
                for (int i = 0; i < H; ++i) {
                    x_ptr[i] += down_[static_cast<size_t>(i)];
                }
            }
        }

        output.resize(static_cast<size_t>(S * H));
        for (int s = 0; s < S; ++s) {
            rms_norm(
                x_.data() + static_cast<size_t>(s) * H,
                weights.final_norm.data(),
                H,
                cfg.rms_norm_eps,
                output.data() + static_cast<size_t>(s) * H);
        }
    }

    int64_t sample_channel(int ch,
                           const float* vec,
                           float temperature,
                           int top_k,
                           float top_p,
                           float repetition_penalty,
                           std::vector<V3RepetitionHistory>& history) {
        const float* head = engine_.audio_emb_t_.data.data() +
            static_cast<int64_t>(ch) * engine_.audio_emb_t_.dim1 * engine_.audio_emb_t_.dim2;
        matvec_transposed(vec, head, engine_.audio_emb_t_.dim1, engine_.audio_emb_t_.dim2, logits_);
        V3RepetitionHistory* prev = history.empty() ? nullptr : &history[static_cast<size_t>(ch)];
        const int64_t code = engine_.sample_logits(logits_, temperature, top_k, top_p, repetition_penalty, prev);
        if (prev) {
            prev->add(static_cast<int32_t>(code));
        }
        return code;
    }

    void generate_frame_impl(const std::vector<float>& h,
                             float temperature,
                             int top_k,
                             float top_p,
                             float repetition_penalty,
                             std::vector<V3RepetitionHistory>& history,
                             std::vector<int64_t>& codes,
                             bool& eos) {
        const auto& cfg = engine_.config_;
        const int H = cfg.hidden_size;
        if (!engine_.acoustic_weights_.loaded) {
            throw std::runtime_error("native acoustic weights are not loaded");
        }
        if (static_cast<int>(h.size()) < H) {
            throw std::runtime_error("native acoustic input hidden state is too small");
        }

        reset_caches();
        token_.resize(static_cast<size_t>(2 * H));
        std::copy(h.begin(), h.begin() + H, token_.begin());
        const float* sgs = engine_.text_emb_.data.data() + cfg.speech_generation_start_token_id * engine_.text_emb_.cols;
        std::copy(sgs, sgs + H, token_.begin() + H);

        const int initial_positions[2] = {0, 1};
        cached_step(token_, initial_positions, 2, hidden_);
        std::copy(hidden_.begin(), hidden_.begin() + H, slot0_.begin());

        codes.clear();
        codes.reserve(static_cast<size_t>(cfg.n_vq));
        codes.push_back(sample_channel(0, hidden_.data() + H, temperature, top_k, top_p, repetition_penalty, history));

        for (int ch = 1; ch < cfg.n_vq; ++ch) {
            const int64_t prev_code = codes.back();
            if (prev_code < 0 || prev_code >= engine_.audio_emb_.dim1) {
                throw std::runtime_error("native acoustic sampled code is out of range");
            }
            const float* emb = engine_.audio_emb_.data.data() +
                (static_cast<int64_t>(ch - 1) * engine_.audio_emb_.dim1 + prev_code) * engine_.audio_emb_.dim2;
            token_.resize(static_cast<size_t>(H));
            std::copy(emb, emb + H, token_.begin());
            const int step_position = ch + 1;
            cached_step(token_, &step_position, 1, hidden_);
            codes.push_back(sample_channel(ch, hidden_.data(), temperature, top_k, top_p, repetition_penalty, history));
        }

        matvec_transposed(slot0_.data(), engine_.text_emb_t_.data.data(), engine_.text_emb_t_.rows, engine_.text_emb_t_.cols, text_logits_);
        eos = static_cast<int>(std::distance(text_logits_.begin(), std::max_element(text_logits_.begin(), text_logits_.end()))) ==
              cfg.speech_generation_end_token_id;
    }
};

bool VieneuV3OnnxEngine::initialize_native_acoustic_executor(std::string& error) {
    const std::string weights_path = join_path(join_path(model_dir_, "acoustic"), "vieneu_acoustic_weights.npz");
    if (!load_acoustic_weights(weights_path, error)) {
        return false;
    }
    try {
        acoustic_executor_ = std::make_unique<NativeAcousticExecutor>(*this);
        return true;
    } catch (const std::exception& e) {
        error = std::string("Failed to initialize VieNeu v3 ggml acoustic executor: ") + e.what();
        return false;
    }
}
