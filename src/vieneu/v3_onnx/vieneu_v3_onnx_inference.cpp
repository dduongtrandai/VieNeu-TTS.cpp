#include "../vieneu_v3_onnx.h"
#include "vieneu_v3_onnx_internal.h"

#include <algorithm>
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
    try {
        const int H = config_.hidden_size;
        const int nH = config_.local_num_attention_heads;
        const int hd = H / nH;
        std::vector<float> token(static_cast<size_t>(2 * H));
        std::copy(h.begin(), h.begin() + H, token.begin());
        const float* sgs = text_emb_.data.data() + config_.speech_generation_start_token_id * text_emb_.cols;
        std::copy(sgs, sgs + H, token.begin() + H);
        std::vector<int64_t> pos = {0, 1};
        std::vector<float> empty;
        std::vector<int64_t> empty_shape = {1, nH, 0, hd};
        std::vector<int64_t> token_shape = {1, 2, H};
        std::vector<int64_t> pos_shape = {1, 2};

        auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        if (acoustic_io_.input_names.size() != 6 || acoustic_io_.output_names.size() != 5) {
            error = "VieNeu v3 acoustic ONNX signature mismatch: expected 6 inputs and 5 outputs.";
            return false;
        }
        std::vector<Ort::Value> inputs;
        inputs.emplace_back(Ort::Value::CreateTensor<float>(mem, token.data(), token.size(), token_shape.data(), token_shape.size()));
        inputs.emplace_back(Ort::Value::CreateTensor<int64_t>(mem, pos.data(), pos.size(), pos_shape.data(), pos_shape.size()));
        inputs.emplace_back(Ort::Value::CreateTensor<float>(mem, empty.data(), 0, empty_shape.data(), empty_shape.size()));
        inputs.emplace_back(Ort::Value::CreateTensor<float>(mem, empty.data(), 0, empty_shape.data(), empty_shape.size()));
        inputs.emplace_back(Ort::Value::CreateTensor<float>(mem, empty.data(), 0, empty_shape.data(), empty_shape.size()));
        inputs.emplace_back(Ort::Value::CreateTensor<float>(mem, empty.data(), 0, empty_shape.data(), empty_shape.size()));
        auto out = acoustic_session_->Run(
            Ort::RunOptions{nullptr},
            acoustic_io_.input_ptrs.data(),
            inputs.data(),
            inputs.size(),
            acoustic_io_.output_ptrs.data(),
            acoustic_io_.output_ptrs.size());
        TensorBlob hidden = copy_float_tensor(out[0]);
        TensorBlob pk0 = copy_float_tensor(out[1]);
        TensorBlob pk1 = copy_float_tensor(out[2]);
        TensorBlob pv0 = copy_float_tensor(out[3]);
        TensorBlob pv1 = copy_float_tensor(out[4]);
        std::vector<float> slot0(hidden.data.begin(), hidden.data.begin() + H);

        std::vector<float> logits;
        auto sample_channel = [&](int ch, const float* vec) {
            const float* head = audio_emb_t_.data.data() + static_cast<int64_t>(ch) * audio_emb_t_.dim1 * audio_emb_t_.dim2;
            matvec_transposed(vec, head, audio_emb_t_.dim1, audio_emb_t_.dim2, logits);
            std::unordered_set<int>* prev = history.empty() ? nullptr : &history[static_cast<size_t>(ch)];
            int64_t code = sample_logits(logits, temperature, top_k, top_p, repetition_penalty, prev);
            if (prev) prev->insert(static_cast<int>(code));
            return code;
        };

        codes.clear();
        codes.reserve(static_cast<size_t>(config_.n_vq));
        codes.push_back(sample_channel(0, hidden.data.data() + H));
        for (int ch = 1; ch < config_.n_vq; ++ch) {
            const float* emb = audio_emb_.data.data() +
                (static_cast<int64_t>(ch - 1) * audio_emb_.dim1 + codes.back()) * audio_emb_.dim2;
            std::vector<float> step_token(emb, emb + H);
            std::vector<int64_t> step_pos = {ch + 1};
            std::vector<int64_t> step_token_shape = {1, 1, H};
            std::vector<int64_t> step_pos_shape = {1, 1};
            std::vector<Ort::Value> step_inputs;
            step_inputs.emplace_back(Ort::Value::CreateTensor<float>(mem, step_token.data(), step_token.size(), step_token_shape.data(), step_token_shape.size()));
            step_inputs.emplace_back(Ort::Value::CreateTensor<int64_t>(mem, step_pos.data(), step_pos.size(), step_pos_shape.data(), step_pos_shape.size()));
            step_inputs.emplace_back(Ort::Value::CreateTensor<float>(mem, pk0.data.data(), pk0.data.size(), pk0.shape.data(), pk0.shape.size()));
            step_inputs.emplace_back(Ort::Value::CreateTensor<float>(mem, pk1.data.data(), pk1.data.size(), pk1.shape.data(), pk1.shape.size()));
            step_inputs.emplace_back(Ort::Value::CreateTensor<float>(mem, pv0.data.data(), pv0.data.size(), pv0.shape.data(), pv0.shape.size()));
            step_inputs.emplace_back(Ort::Value::CreateTensor<float>(mem, pv1.data.data(), pv1.data.size(), pv1.shape.data(), pv1.shape.size()));
            auto step_out = acoustic_session_->Run(
                Ort::RunOptions{nullptr},
                acoustic_io_.input_ptrs.data(),
                step_inputs.data(),
                step_inputs.size(),
                acoustic_io_.output_ptrs.data(),
                acoustic_io_.output_ptrs.size());
            copy_float_tensor_into(step_out[0], hidden);
            copy_float_tensor_into(step_out[1], pk0);
            copy_float_tensor_into(step_out[2], pk1);
            copy_float_tensor_into(step_out[3], pv0);
            copy_float_tensor_into(step_out[4], pv1);
            codes.push_back(sample_channel(ch, hidden.data.data()));
        }

        std::vector<float> text_logits;
        matvec_transposed(slot0.data(), text_emb_t_.data.data(), text_emb_t_.rows, text_emb_t_.cols, text_logits);
        eos = static_cast<int>(std::distance(text_logits.begin(), std::max_element(text_logits.begin(), text_logits.end()))) ==
              config_.speech_generation_end_token_id;
        return true;
    } catch (const std::exception& e) {
        error = std::string("VieNeu v3 acoustic frame failed: ") + e.what();
        return false;
    }
}
