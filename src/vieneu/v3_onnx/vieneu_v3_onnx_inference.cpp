#include "../vieneu_v3_onnx.h"
#include "vieneu_v3_onnx_internal.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <string>
#include <unordered_set>
#include <vector>
#include <stdexcept>

// --- VieneuV3OnnxEngine Inference Member Functions ---

VieneuV3OnnxEngine::PromptRows VieneuV3OnnxEngine::build_rows(
    const std::string& phonemes,
    const std::vector<int64_t>* ref_codes,
    int leading_token) const {
    const std::vector<int64_t> phone_ids = tokenizer_.encode(phonemes);
    const int64_t cols = config_.n_vq + 1;
    const int64_t text_rows = static_cast<int64_t>(phone_ids.size()) + 3;
    const int64_t ref_rows = ref_codes ? static_cast<int64_t>(ref_codes->size() / config_.n_vq) : 0;
    PromptRows rows;
    rows.rows = text_rows + ref_rows;
    rows.cols = cols;
    rows.data.assign(static_cast<size_t>(rows.rows * rows.cols), config_.audio_pad_token_id);
    rows.data[0] = leading_token;
    rows.data[cols] = config_.text_prompt_start_token_id;
    for (size_t i = 0; i < phone_ids.size(); ++i) {
        rows.data[static_cast<size_t>((static_cast<int64_t>(i) + 2) * cols)] = phone_ids[i];
    }
    rows.data[static_cast<size_t>((text_rows - 1) * cols)] = config_.text_prompt_end_token_id;
    if (ref_codes) {
        for (int64_t r = 0; r < ref_rows; ++r) {
            const int64_t dst_row = text_rows + r;
            rows.data[static_cast<size_t>(dst_row * cols)] = config_.audio_ref_slot_token_id;
            for (int ch = 0; ch < config_.n_vq; ++ch) {
                rows.data[static_cast<size_t>(dst_row * cols + ch + 1)] =
                    (*ref_codes)[static_cast<size_t>(r * config_.n_vq + ch)];
            }
        }
    }
    return rows;
}

std::vector<float> VieneuV3OnnxEngine::embed_rows(const PromptRows& rows) const {
    std::vector<float> embeds(static_cast<size_t>(rows.rows * config_.hidden_size), 0.0f);
    for (int64_t r = 0; r < rows.rows; ++r) {
        float* dst = embeds.data() + r * config_.hidden_size;
        const int64_t text_id = rows.data[static_cast<size_t>(r * rows.cols)];
        if (text_id >= 0 && text_id < text_emb_.rows) {
            const float* src = text_emb_.data.data() + text_id * text_emb_.cols;
            std::copy(src, src + config_.hidden_size, dst);
        }
        for (int ch = 0; ch < config_.n_vq; ++ch) {
            const int64_t id = rows.data[static_cast<size_t>(r * rows.cols + ch + 1)];
            if (id == config_.audio_pad_token_id || id < 0 || id >= audio_emb_.dim1) {
                continue;
            }
            const float* src = audio_emb_.data.data() +
                (static_cast<int64_t>(ch) * audio_emb_.dim1 + id) * audio_emb_.dim2;
            for (int h = 0; h < config_.hidden_size; ++h) {
                dst[h] += src[h];
            }
        }
    }
    return embeds;
}

bool VieneuV3OnnxEngine::acoustic_frame(
    const std::vector<float>& h,
    float temperature,
    int top_k,
    float top_p,
    float repetition_penalty,
    std::vector<std::unordered_set<int>>& history,
    std::vector<int64_t>& codes,
    bool& eos,
    std::string& error) {
    const auto frame_start = benchmark_enabled_ ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    try {
        const int H = config_.hidden_size;
        const int nH = config_.local_num_attention_heads;
        const int hd = H / nH;
        acoustic_token_.resize(static_cast<size_t>(2 * H));
        std::copy(h.begin(), h.begin() + H, acoustic_token_.begin());
        const float* sgs = text_emb_.data.data() + config_.speech_generation_start_token_id * text_emb_.cols;
        std::copy(sgs, sgs + H, acoustic_token_.begin() + H);
        std::array<int64_t, 2> pos = {0, 1};
        acoustic_empty_.clear();
        std::array<int64_t, 4> empty_shape = {1, nH, 0, hd};
        std::array<int64_t, 3> token_shape = {1, 2, H};
        std::array<int64_t, 2> pos_shape = {1, 2};

        auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        if (acoustic_io_.input_names.size() != 6 || acoustic_io_.output_names.size() != 5) {
            error = "VieNeu v3 acoustic ONNX signature mismatch: expected 6 inputs and 5 outputs.";
            return false;
        }
        acoustic_inputs_.clear();
        acoustic_inputs_.reserve(6);
        acoustic_inputs_.emplace_back(Ort::Value::CreateTensor<float>(mem, acoustic_token_.data(), acoustic_token_.size(), token_shape.data(), token_shape.size()));
        acoustic_inputs_.emplace_back(Ort::Value::CreateTensor<int64_t>(mem, pos.data(), pos.size(), pos_shape.data(), pos_shape.size()));
        acoustic_inputs_.emplace_back(Ort::Value::CreateTensor<float>(mem, acoustic_empty_.data(), 0, empty_shape.data(), empty_shape.size()));
        acoustic_inputs_.emplace_back(Ort::Value::CreateTensor<float>(mem, acoustic_empty_.data(), 0, empty_shape.data(), empty_shape.size()));
        acoustic_inputs_.emplace_back(Ort::Value::CreateTensor<float>(mem, acoustic_empty_.data(), 0, empty_shape.data(), empty_shape.size()));
        acoustic_inputs_.emplace_back(Ort::Value::CreateTensor<float>(mem, acoustic_empty_.data(), 0, empty_shape.data(), empty_shape.size()));
        auto out = acoustic_session_->Run(
            Ort::RunOptions{nullptr},
            acoustic_io_.input_ptrs.data(),
            acoustic_inputs_.data(),
            acoustic_inputs_.size(),
            acoustic_io_.output_ptrs.data(),
            acoustic_io_.output_ptrs.size());
        Ort::Value hidden_val = std::move(out[0]);
        Ort::Value pk0 = std::move(out[1]);
        Ort::Value pk1 = std::move(out[2]);
        Ort::Value pv0 = std::move(out[3]);
        Ort::Value pv1 = std::move(out[4]);

        const float* hidden_ptr = hidden_val.GetTensorData<float>();
        acoustic_slot0_.assign(hidden_ptr, hidden_ptr + H);

        auto sample_channel = [&](int ch, const float* vec) {
            const float* head = audio_emb_t_.data.data() + static_cast<int64_t>(ch) * audio_emb_t_.dim1 * audio_emb_t_.dim2;
            matvec_transposed(vec, head, audio_emb_t_.dim1, audio_emb_t_.dim2, acoustic_logits_);
            std::unordered_set<int>* prev = history.empty() ? nullptr : &history[static_cast<size_t>(ch)];
            int64_t code = sample_logits(acoustic_logits_, temperature, top_k, top_p, repetition_penalty, prev);
            if (prev) prev->insert(static_cast<int>(code));
            return code;
        };

        codes.clear();
        codes.reserve(static_cast<size_t>(config_.n_vq));
        codes.push_back(sample_channel(0, hidden_ptr + H));
        for (int ch = 1; ch < config_.n_vq; ++ch) {
            const float* emb = audio_emb_.data.data() +
                (static_cast<int64_t>(ch - 1) * audio_emb_.dim1 + codes.back()) * audio_emb_.dim2;
            int64_t step_pos = ch + 1;
            std::array<int64_t, 3> step_token_shape = {1, 1, H};
            std::array<int64_t, 2> step_pos_shape = {1, 1};
            acoustic_step_inputs_.clear();
            acoustic_step_inputs_.reserve(6);
            acoustic_step_inputs_.emplace_back(Ort::Value::CreateTensor<float>(mem, const_cast<float*>(emb), static_cast<size_t>(H), step_token_shape.data(), step_token_shape.size()));
            acoustic_step_inputs_.emplace_back(Ort::Value::CreateTensor<int64_t>(mem, &step_pos, 1, step_pos_shape.data(), step_pos_shape.size()));
            acoustic_step_inputs_.emplace_back(std::move(pk0));
            acoustic_step_inputs_.emplace_back(std::move(pk1));
            acoustic_step_inputs_.emplace_back(std::move(pv0));
            acoustic_step_inputs_.emplace_back(std::move(pv1));
            auto step_out = acoustic_session_->Run(
                Ort::RunOptions{nullptr},
                acoustic_io_.input_ptrs.data(),
                acoustic_step_inputs_.data(),
                acoustic_step_inputs_.size(),
                acoustic_io_.output_ptrs.data(),
                acoustic_io_.output_ptrs.size());
            hidden_val = std::move(step_out[0]);
            pk0 = std::move(step_out[1]);
            pk1 = std::move(step_out[2]);
            pv0 = std::move(step_out[3]);
            pv1 = std::move(step_out[4]);
            
            const float* step_hidden_ptr = hidden_val.GetTensorData<float>();
            codes.push_back(sample_channel(ch, step_hidden_ptr));
        }

        matvec_transposed(acoustic_slot0_.data(), text_emb_t_.data.data(), text_emb_t_.rows, text_emb_t_.cols, acoustic_text_logits_);
        eos = static_cast<int>(std::distance(acoustic_text_logits_.begin(), std::max_element(acoustic_text_logits_.begin(), acoustic_text_logits_.end()))) ==
              config_.speech_generation_end_token_id;
        if (benchmark_enabled_) {
            const auto frame_end = std::chrono::steady_clock::now();
            benchmark_stats_.acoustic_frame_ms += std::chrono::duration<double, std::milli>(frame_end - frame_start).count();
            benchmark_stats_.acoustic_frame_calls += 1;
        }
        return true;
    } catch (const std::exception& e) {
        if (benchmark_enabled_) {
            const auto frame_end = std::chrono::steady_clock::now();
            benchmark_stats_.acoustic_frame_ms += std::chrono::duration<double, std::milli>(frame_end - frame_start).count();
            benchmark_stats_.acoustic_frame_calls += 1;
        }
        error = std::string("VieNeu v3 acoustic frame failed: ") + e.what();
        return false;
    }
}
