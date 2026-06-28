#include "../vieneu_v3_onnx.h"
#include "vieneu_v3_onnx_internal.h"
#include "../vieneu.h"

#include <algorithm>
#include <cmath>
#include <mutex>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

// --- VieneuV3OnnxEngine Orchestrator Member Functions ---

bool VieneuV3OnnxEngine::initialize(const VieneuV3OnnxInit& init, std::string& error) {
    initialized_ = false;
    env_.reset();
    prefill_session_.reset();
    decode_session_.reset();
    acoustic_session_.reset();
    codec_decode_session_.reset();
    codec_encode_session_.reset();
    session_options_.reset();
    prefill_io_ = SessionIo{};
    decode_io_ = SessionIo{};
    acoustic_io_ = SessionIo{};
    codec_decode_io_ = SessionIo{};
    codec_encode_io_ = SessionIo{};
    codec_encode_path_.clear();
    voices_json_.clear();
    default_voice_id_.clear();
    voice_presets_.clear();
    rng_.seed(std::random_device{}());

    if (init.model_dir.empty() && init.onnx_dir.empty()) {
        error = "VieNeu v3 requires model_dir or onnx_dir.";
        return false;
    }
    if (init.codec_dir.empty()) {
        error = "VieNeu v3 requires codec_dir with MOSS ONNX codec files.";
        return false;
    }
    if (!validate_assets(init, error)) {
        return false;
    }

    const std::string config_path = init.config_path.empty() ? join_path(model_dir_, "config.json") : init.config_path;
    const std::string tokenizer_path = init.tokenizer_path.empty() ? join_path(model_dir_, "tokenizer.json") : init.tokenizer_path;
    if (!load_config(config_path, error) ||
        !load_heads_npz(join_path(onnx_dir_, "vieneu_v3_heads.npz"), error) ||
        !tokenizer_.load(tokenizer_path, error)) {
        return false;
    }

    env_ = std::make_shared<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "VieneuV3Onnx");
    session_options_ = std::make_unique<Ort::SessionOptions>();
    if (init.n_threads > 0) {
        session_options_->SetIntraOpNumThreads(init.n_threads);
    }
    session_options_->SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    if (!load_session(join_path(onnx_dir_, "vieneu_prefill.onnx"), prefill_session_, error) ||
        !load_session(join_path(onnx_dir_, "vieneu_decode_step.onnx"), decode_session_, error) ||
        !load_session(join_path(onnx_dir_, "vieneu_acoustic_cached.onnx"), acoustic_session_, error) ||
        !load_session(join_path(codec_dir_, "moss_audio_tokenizer_decode_full.onnx"), codec_decode_session_, error)) {
        return false;
    }
    cache_session_io(*prefill_session_, prefill_io_);
    cache_session_io(*decode_session_, decode_io_);
    cache_session_io(*acoustic_session_, acoustic_io_);
    cache_session_io(*codec_decode_session_, codec_decode_io_);

    if (!load_voices(init.voices_json_path, error)) {
        return false;
    }

    initialized_ = true;
    return true;
}

std::string VieneuV3OnnxEngine::phonemize_for_v3(const std::string& text) const {
    return VieneuProfile::phonemize(text);
}

bool VieneuV3OnnxEngine::synthesize_phonemes(
    const std::string& phonemes,
    const std::vector<int64_t>* ref_codes,
    int leading_token,
    const VieneuV3OnnxParams& params,
    std::vector<float>& out_audio,
    std::string& error) {
    out_audio.clear();
    const PromptRows rows = build_rows(phonemes, ref_codes, leading_token);
    std::vector<float> prompt_embeds = embed_rows(rows);
    std::vector<int64_t> prompt_shape = {1, rows.rows, config_.hidden_size};
    auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    std::lock_guard<std::mutex> lock(run_mutex_);
    try {
        const size_t expected_lm_outputs = static_cast<size_t>(1 + config_.num_hidden_layers * 2);
        if (prefill_io_.input_names.size() != 1 || prefill_io_.output_names.size() != expected_lm_outputs) {
            error = "VieNeu v3 prefill ONNX signature mismatch.";
            return false;
        }
        Ort::Value prompt_tensor = Ort::Value::CreateTensor<float>(mem, prompt_embeds.data(), prompt_embeds.size(), prompt_shape.data(), prompt_shape.size());
        auto pre = prefill_session_->Run(
            Ort::RunOptions{nullptr},
            prefill_io_.input_ptrs.data(),
            &prompt_tensor,
            1,
            prefill_io_.output_ptrs.data(),
            prefill_io_.output_ptrs.size());
        TensorBlob hidden_blob = copy_float_tensor(pre[0]);
        std::vector<TensorBlob> past_k;
        std::vector<TensorBlob> past_v;
        past_k.reserve(static_cast<size_t>(config_.num_hidden_layers));
        past_v.reserve(static_cast<size_t>(config_.num_hidden_layers));
        for (int i = 0; i < config_.num_hidden_layers; ++i) past_k.push_back(copy_float_tensor(pre[1 + i]));
        for (int i = 0; i < config_.num_hidden_layers; ++i) past_v.push_back(copy_float_tensor(pre[1 + config_.num_hidden_layers + i]));

        std::vector<float> h(static_cast<size_t>(config_.hidden_size));
        const int64_t last_offset = (rows.rows - 1) * config_.hidden_size;
        std::copy(hidden_blob.data.begin() + last_offset, hidden_blob.data.begin() + last_offset + config_.hidden_size, h.begin());

        const size_t expected_decode_inputs = static_cast<size_t>(2 + config_.num_hidden_layers * 2);
        if (decode_io_.input_names.size() != expected_decode_inputs || decode_io_.output_names.size() != expected_lm_outputs) {
            error = "VieNeu v3 decode-step ONNX signature mismatch.";
            return false;
        }

        std::vector<std::unordered_set<int>> history;
        if (std::fabs(params.repetition_penalty - 1.0f) > 1e-6f) {
            history.resize(static_cast<size_t>(config_.n_vq));
        }
        std::vector<int64_t> frames;
        const int max_frames = (std::max)(1, params.max_new_frames);
        frames.reserve(static_cast<size_t>(max_frames * config_.n_vq));
        for (int t = 0; t < max_frames; ++t) {
            std::vector<int64_t> codes;
            bool eos = false;
            if (!acoustic_frame(h, params.temperature, params.top_k, params.top_p, params.repetition_penalty, history, codes, eos, error)) {
                return false;
            }
            frames.insert(frames.end(), codes.begin(), codes.end());
            if (eos) {
                break;
            }

            PromptRows slot;
            slot.rows = 1;
            slot.cols = config_.n_vq + 1;
            slot.data.assign(static_cast<size_t>(slot.cols), config_.audio_pad_token_id);
            slot.data[0] = config_.speech_generation_start_token_id;
            for (int ch = 0; ch < config_.n_vq; ++ch) slot.data[static_cast<size_t>(ch + 1)] = codes[static_cast<size_t>(ch)];
            std::vector<float> se = embed_rows(slot);
            std::vector<int64_t> se_shape = {1, 1, config_.hidden_size};
            std::vector<int64_t> pos = {rows.rows + t};
            std::vector<int64_t> pos_shape = {1, 1};

            std::vector<Ort::Value> inputs;
            inputs.reserve(static_cast<size_t>(expected_decode_inputs));
            inputs.emplace_back(Ort::Value::CreateTensor<float>(mem, se.data(), se.size(), se_shape.data(), se_shape.size()));
            inputs.emplace_back(Ort::Value::CreateTensor<int64_t>(mem, pos.data(), pos.size(), pos_shape.data(), pos_shape.size()));
            for (auto& pk : past_k) inputs.emplace_back(Ort::Value::CreateTensor<float>(mem, pk.data.data(), pk.data.size(), pk.shape.data(), pk.shape.size()));
            for (auto& pv : past_v) inputs.emplace_back(Ort::Value::CreateTensor<float>(mem, pv.data.data(), pv.data.size(), pv.shape.data(), pv.shape.size()));
            auto dec = decode_session_->Run(
                Ort::RunOptions{nullptr},
                decode_io_.input_ptrs.data(),
                inputs.data(),
                inputs.size(),
                decode_io_.output_ptrs.data(),
                decode_io_.output_ptrs.size());
            const float* dec_hidden = dec[0].GetTensorData<float>();
            std::copy(dec_hidden, dec_hidden + config_.hidden_size, h.begin());
            for (int i = 0; i < config_.num_hidden_layers; ++i) copy_float_tensor_into(dec[1 + i], past_k[static_cast<size_t>(i)]);
            for (int i = 0; i < config_.num_hidden_layers; ++i) copy_float_tensor_into(dec[1 + config_.num_hidden_layers + i], past_v[static_cast<size_t>(i)]);
        }

        if (frames.empty()) {
            error = "VieNeu v3 synthesis produced no acoustic frames.";
            return false;
        }
        return decode_codes(frames, static_cast<int64_t>(frames.size() / config_.n_vq), out_audio, error);
    } catch (const std::exception& e) {
        error = std::string("VieNeu v3 synthesis failed: ") + e.what();
        return false;
    }
}

bool VieneuV3OnnxEngine::synthesize(const VieneuV3OnnxParams& params, std::vector<float>& out_audio, std::string& error) {
    out_audio.clear();
    if (!initialized_) {
        error = "VieNeu v3 ONNX engine is not initialized.";
        return false;
    }
    if (params.text.empty()) {
        error = "VieNeu v3 synthesis requires non-empty text.";
        return false;
    }

    int leading_token = config_.emotion_0_token_id;
    std::vector<int64_t> ref_codes;
    if (!params.ref_audio_path.empty()) {
        if (!encode_reference_audio(params.ref_audio_path, ref_codes, error)) {
            return false;
        }
    } else {
        VoicePreset preset;
        if (!resolve_voice_preset(params.voice_id, preset, error)) {
            return false;
        }
        if (preset.has_reserved_id) {
            leading_token = preset.reserved_id;
        }
        if (!preset.codes.empty()) {
            ref_codes = std::move(preset.codes);
        }
    }

    if (!ref_codes.empty() && ref_codes.size() % static_cast<size_t>(config_.n_vq) != 0) {
        error = "VieNeu v3 reference codes are not divisible by n_vq.";
        return false;
    }

    const std::vector<std::string> chunks = split_text_for_v3_chunks(params.text, params.max_chars);
    if (chunks.empty()) {
        error = "VieNeu v3 synthesis produced no text chunks.";
        return false;
    }

    const int silence_samples = static_cast<int>(std::lround(static_cast<double>(sample_rate()) * 0.15));
    for (size_t i = 0; i < chunks.size(); ++i) {
        const std::string phonemes = phonemize_for_v3(chunks[i]);
        std::vector<float> chunk_audio;
        if (!synthesize_phonemes(
                phonemes,
                ref_codes.empty() ? nullptr : &ref_codes,
                leading_token,
                params,
                chunk_audio,
                error)) {
            if (chunks.size() > 1) {
                error += " (chunk " + std::to_string(i + 1) + "/" + std::to_string(chunks.size()) + ")";
            }
            return false;
        }
        if (chunk_audio.empty()) {
            continue;
        }
        if (!out_audio.empty() && silence_samples > 0) {
            out_audio.insert(out_audio.end(), static_cast<size_t>(silence_samples), 0.0f);
        }
        out_audio.insert(out_audio.end(), chunk_audio.begin(), chunk_audio.end());
    }
    if (out_audio.empty()) {
        error = "VieNeu v3 synthesis produced empty audio.";
        return false;
    }
    return true;
}
