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
#include <unordered_set>
#include <vector>

namespace {

int ggml_thread_count_from_env() {
    const char* value = std::getenv("OMP_NUM_THREADS");
    if (!value || !*value) {
        return 4;
    }
    char* end = nullptr;
    const long parsed = std::strtol(value, &end, 10);
    if (end == value || parsed <= 0) {
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

class GgmlLinearOp {
public:
    GgmlLinearOp() = default;

    GgmlLinearOp(ggml_backend_t backend, const float* weight, int out_dim, int in_dim, const char* name) {
        initialize(backend, weight, out_dim, in_dim, name);
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

    void initialize(ggml_backend_t backend, const float* weight, int out_dim, int in_dim, const char* name) {
        release();
        if (!backend) {
            throw std::runtime_error("ggml backend is not initialized");
        }
        (void) backend;
        thread_count_ = ggml_thread_count_from_env();
        in_dim_ = in_dim;
        out_dim_ = out_dim;
        name_ = name ? name : "linear";

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
        in_dim_ = other.in_dim_;
        out_dim_ = other.out_dim_;
        thread_count_ = other.thread_count_;
        name_ = std::move(other.name_);

        other.ctx_ = nullptr;
        other.weight_ = nullptr;
        other.input_ = nullptr;
        other.output_ = nullptr;
        other.graph_ = nullptr;
        other.in_dim_ = 0;
        other.out_dim_ = 0;
        other.thread_count_ = 1;
        other.work_data_.clear();
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
    std::string name_;
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
                    const char* name) {
        release();
        if (!backend) {
            throw std::runtime_error("ggml backend is not initialized");
        }
        (void) backend;
        thread_count_ = ggml_thread_count_from_env();
        hidden_dim_ = hidden_dim;
        intermediate_dim_ = intermediate_dim;
        name_ = name ? name : "ffn";

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
        hidden_dim_ = other.hidden_dim_;
        intermediate_dim_ = other.intermediate_dim_;
        thread_count_ = other.thread_count_;
        name_ = std::move(other.name_);

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
    std::string name_;
};

} // namespace

class VieneuV3OnnxEngine::NativeAcousticExecutor final : public VieneuV3OnnxEngine::AcousticExecutor {
public:
    explicit NativeAcousticExecutor(VieneuV3OnnxEngine& engine) : engine_(engine) {
        backend_ = ggml_backend_cpu_init();
        if (!backend_) {
            throw std::runtime_error("failed to initialize ggml CPU backend");
        }
        ggml_backend_cpu_set_n_threads(backend_, ggml_thread_count_from_env());
        fuse_ffn_ = env_flag_enabled("VIENEU_GGML_FUSE_FFN");
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
                        std::vector<std::unordered_set<int>>& history,
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
        layer_ops_.clear();
        layer_ops_.reserve(weights.layers.size());
        for (size_t i = 0; i < weights.layers.size(); ++i) {
            const AcousticLayerWeights& w = weights.layers[i];
            const std::string prefix = "acoustic.layer." + std::to_string(i) + ".";
            LayerOps ops;
            ops.qkv.initialize(backend_, w.qkv.data(), 3 * H, H, (prefix + "qkv").c_str());
            ops.o_proj.initialize(backend_, w.o_proj.data(), H, H, (prefix + "o_proj").c_str());
            if (fuse_ffn_) {
                ops.ffn.initialize(backend_, w.ff_gate.data(), w.ff_up.data(), w.ff_down.data(), H, I, (prefix + "ffn").c_str());
            } else {
                ops.ff_gate.initialize(backend_, w.ff_gate.data(), I, H, (prefix + "ff_gate").c_str());
                ops.ff_up.initialize(backend_, w.ff_up.data(), I, H, (prefix + "ff_up").c_str());
                ops.ff_down.initialize(backend_, w.ff_down.data(), H, I, (prefix + "ff_down").c_str());
            }
            layer_ops_.push_back(std::move(ops));
        }
    }

    void cached_step(const std::vector<float>& input, const std::vector<int>& positions, std::vector<float>& output) {
        const auto& cfg = engine_.config_;
        const auto& weights = engine_.acoustic_weights_;
        const int H = cfg.hidden_size;
        const int nH = cfg.local_num_attention_heads;
        const int hd = H / nH;
        const int I = cfg.local_intermediate_size;
        const int S = static_cast<int>(positions.size());
        const float inv_sqrt_hd = 1.0f / std::sqrt(static_cast<float>(hd));

        x_ = input;
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
            const int past = static_cast<int>(cache.k.size() / static_cast<size_t>(H));

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

            cache.k.insert(cache.k.end(), new_k_.begin(), new_k_.begin() + static_cast<size_t>(S * H));
            cache.v.insert(cache.v.end(), new_v_.begin(), new_v_.begin() + static_cast<size_t>(S * H));

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
                           std::vector<std::unordered_set<int>>& history) {
        const float* head = engine_.audio_emb_t_.data.data() +
            static_cast<int64_t>(ch) * engine_.audio_emb_t_.dim1 * engine_.audio_emb_t_.dim2;
        matvec_transposed(vec, head, engine_.audio_emb_t_.dim1, engine_.audio_emb_t_.dim2, logits_);
        std::unordered_set<int>* prev = history.empty() ? nullptr : &history[static_cast<size_t>(ch)];
        const int64_t code = engine_.sample_logits(logits_, temperature, top_k, top_p, repetition_penalty, prev);
        if (prev) {
            prev->insert(static_cast<int>(code));
        }
        return code;
    }

    void generate_frame_impl(const std::vector<float>& h,
                             float temperature,
                             int top_k,
                             float top_p,
                             float repetition_penalty,
                             std::vector<std::unordered_set<int>>& history,
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

        caches_.assign(static_cast<size_t>(cfg.local_num_hidden_layers), LayerCache{});
        token_.resize(static_cast<size_t>(2 * H));
        std::copy(h.begin(), h.begin() + H, token_.begin());
        const float* sgs = engine_.text_emb_.data.data() + cfg.speech_generation_start_token_id * engine_.text_emb_.cols;
        std::copy(sgs, sgs + H, token_.begin() + H);

        cached_step(token_, {0, 1}, hidden_);
        slot0_.assign(hidden_.begin(), hidden_.begin() + H);

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
            token_.assign(emb, emb + H);
            cached_step(token_, {ch + 1}, hidden_);
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
