#include "../vieneu_v3_onnx.h"
#include "vieneu_v3_onnx_internal.h"
#include "../vieneu.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <iostream>
#include <limits>
#include <mutex>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>
#include <thread>

namespace {

std::string getenv_string(const char* name) {
    const char* value = std::getenv(name);
    return value ? std::string(value) : std::string();
}

std::string lowercase(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return value;
}

bool env_enabled(const char* name) {
    const std::string value = lowercase(getenv_string(name));
    return !value.empty() && value != "0" && value != "false" && value != "off" && value != "no";
}

int env_int(const char* name, int fallback) {
    const std::string value = getenv_string(name);
    if (value.empty()) {
        return fallback;
    }
    try {
        return std::stoi(value);
    } catch (...) {
        return fallback;
    }
}

bool append_requested_execution_provider(Ort::SessionOptions& options, std::string& error) {
    const std::string requested = lowercase(getenv_string("VIENEU_ORT_EP"));
    if (requested.empty() || requested == "cpu") {
        return true;
    }

    try {
        if (requested == "cuda") {
            Ort::CUDAProviderOptions cuda_options;
            const int device_id = env_int("VIENEU_ORT_CUDA_DEVICE_ID", 0);
            cuda_options.Update({{"device_id", std::to_string((std::max)(0, device_id))}});
            options.AppendExecutionProvider_CUDA_V2(*cuda_options);
            return true;
        }

        error = "Unsupported VIENEU_ORT_EP value: " + requested + " (supported: cpu, cuda).";
        return false;
    } catch (const std::exception& e) {
        if (env_enabled("VIENEU_ORT_EP_REQUIRED")) {
            error = "Failed to enable requested ONNX Runtime EP '" + requested + "': " + e.what();
            return false;
        }
        std::cerr << "[VieNeu v3] Failed to enable ONNX Runtime EP '" << requested
                  << "', falling back to CPU: " << e.what() << std::endl;
        return true;
    }
}

} // namespace

// --- VieneuV3OnnxEngine Orchestrator Member Functions ---

void VieneuV3OnnxEngine::reset_benchmark_stats() {
    benchmark_stats_ = BenchmarkStats{};
}

void VieneuV3OnnxEngine::print_benchmark_stats() const {
    if (!benchmark_enabled_) {
        return;
    }

    const auto avg = [](double total_ms, int64_t calls) -> double {
        return calls > 0 ? total_ms / static_cast<double>(calls) : 0.0;
    };

    std::cerr << "[VieNeu v3] Benchmark summary\n"
              << "  prefill: total=" << benchmark_stats_.prefill_ms << " ms"
              << ", calls=" << benchmark_stats_.prefill_calls
              << ", avg=" << avg(benchmark_stats_.prefill_ms, benchmark_stats_.prefill_calls) << " ms\n"
              << "  decode_step: total=" << benchmark_stats_.decode_step_ms << " ms"
              << ", calls=" << benchmark_stats_.decode_step_calls
              << ", avg=" << avg(benchmark_stats_.decode_step_ms, benchmark_stats_.decode_step_calls) << " ms\n"
              << "  acoustic_frame: total=" << benchmark_stats_.acoustic_frame_ms << " ms"
              << ", calls=" << benchmark_stats_.acoustic_frame_calls
              << ", avg=" << avg(benchmark_stats_.acoustic_frame_ms, benchmark_stats_.acoustic_frame_calls) << " ms\n"
              << "  codec_decode: total=" << benchmark_stats_.codec_decode_ms << " ms"
              << ", calls=" << benchmark_stats_.codec_decode_calls
              << ", avg=" << avg(benchmark_stats_.codec_decode_ms, benchmark_stats_.codec_decode_calls) << " ms\n";
}

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
    benchmark_enabled_ = env_enabled("VIENEU_BENCHMARK");
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
    int threads_to_use = env_int("VIENEU_ORT_THREADS", init.n_threads);
    if (threads_to_use <= 0) {
        unsigned int hardware_threads = std::thread::hardware_concurrency();
        threads_to_use = hardware_threads > 0 ? std::max(1, static_cast<int>(std::min(hardware_threads / 2, 4u))) : 4;
    }
    session_options_->SetIntraOpNumThreads(threads_to_use);
    const int inter_op_threads = env_int("VIENEU_ORT_INTER_OP_THREADS", 1);
    session_options_->SetInterOpNumThreads((std::max)(1, inter_op_threads));
    const std::string execution_mode = lowercase(getenv_string("VIENEU_ORT_EXECUTION_MODE"));
    if (execution_mode == "parallel") {
        session_options_->SetExecutionMode(ORT_PARALLEL);
    } else {
        session_options_->SetExecutionMode(ORT_SEQUENTIAL);
    }
    session_options_->SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    session_options_->EnableCpuMemArena();
    if (env_enabled("VIENEU_ORT_DISABLE_SPIN")) {
        session_options_->AddConfigEntry("session.intra_op.allow_spinning", "0");
        session_options_->AddConfigEntry("session.inter_op.allow_spinning", "0");
    }

    if (!append_requested_execution_provider(*session_options_, error)) {
        return false;
    }

    if (env_enabled("VIENEU_ORT_PROFILING")) {
        std::string profile_prefix = getenv_string("VIENEU_ORT_PROFILE_PREFIX");
        if (profile_prefix.empty()) {
            profile_prefix = "vieneu_profile";
        }
#ifdef _WIN32
        const std::wstring wide_profile_prefix(profile_prefix.begin(), profile_prefix.end());
        session_options_->EnableProfiling(wide_profile_prefix.c_str());
#else
        session_options_->EnableProfiling(profile_prefix.c_str());
#endif
    }

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
    reset_benchmark_stats();
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
        const auto prefill_start = benchmark_enabled_ ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
        auto pre = prefill_session_->Run(
            Ort::RunOptions{nullptr},
            prefill_io_.input_ptrs.data(),
            &prompt_tensor,
            1,
            prefill_io_.output_ptrs.data(),
            prefill_io_.output_ptrs.size());
        if (benchmark_enabled_) {
            const auto prefill_end = std::chrono::steady_clock::now();
            benchmark_stats_.prefill_ms += std::chrono::duration<double, std::milli>(prefill_end - prefill_start).count();
            benchmark_stats_.prefill_calls += 1;
        }
        const float* hidden_data = pre[0].GetTensorData<float>();
        std::vector<Ort::Value> past_k;
        std::vector<Ort::Value> past_v;
        past_k.reserve(static_cast<size_t>(config_.num_hidden_layers));
        past_v.reserve(static_cast<size_t>(config_.num_hidden_layers));
        for (int i = 0; i < config_.num_hidden_layers; ++i) {
            past_k.push_back(std::move(pre[1 + i]));
        }
        for (int i = 0; i < config_.num_hidden_layers; ++i) {
            past_v.push_back(std::move(pre[1 + config_.num_hidden_layers + i]));
        }

        synth_h_.resize(static_cast<size_t>(config_.hidden_size));
        const int64_t last_offset = (rows.rows - 1) * config_.hidden_size;
        std::copy(hidden_data + last_offset, hidden_data + last_offset + config_.hidden_size, synth_h_.begin());

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
        std::vector<int64_t> codes;
        codes.reserve(static_cast<size_t>(config_.n_vq));
        synth_se_.resize(static_cast<size_t>(config_.hidden_size));
        std::array<int64_t, 3> se_shape = {1, 1, config_.hidden_size};
        std::array<int64_t, 2> pos_shape = {1, 1};
        synth_decode_inputs_.reserve(static_cast<size_t>(expected_decode_inputs));
        for (int t = 0; t < max_frames; ++t) {
            bool eos = false;
            if (!acoustic_frame(synth_h_, params.temperature, params.top_k, params.top_p, params.repetition_penalty, history, codes, eos, error)) {
                return false;
            }
            frames.insert(frames.end(), codes.begin(), codes.end());
            if (eos) {
                break;
            }

            const float* sgs = text_emb_.data.data() + config_.speech_generation_start_token_id * text_emb_.cols;
            std::copy(sgs, sgs + config_.hidden_size, synth_se_.begin());
            for (int ch = 0; ch < config_.n_vq; ++ch) {
                const int64_t id = codes[static_cast<size_t>(ch)];
                if (id == config_.audio_pad_token_id || id < 0 || id >= audio_emb_.dim1) {
                    continue;
                }
                const float* src = audio_emb_.data.data() +
                    (static_cast<int64_t>(ch) * audio_emb_.dim1 + id) * audio_emb_.dim2;
                for (int h_idx = 0; h_idx < config_.hidden_size; ++h_idx) {
                    synth_se_[static_cast<size_t>(h_idx)] += src[h_idx];
                }
            }
            int64_t pos = rows.rows + t;

            synth_decode_inputs_.clear();
            synth_decode_inputs_.emplace_back(Ort::Value::CreateTensor<float>(mem, synth_se_.data(), synth_se_.size(), se_shape.data(), se_shape.size()));
            synth_decode_inputs_.emplace_back(Ort::Value::CreateTensor<int64_t>(mem, &pos, 1, pos_shape.data(), pos_shape.size()));
            for (auto& pk : past_k) synth_decode_inputs_.emplace_back(std::move(pk));
            for (auto& pv : past_v) synth_decode_inputs_.emplace_back(std::move(pv));
            const auto decode_start = benchmark_enabled_ ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
            auto dec = decode_session_->Run(
                Ort::RunOptions{nullptr},
                decode_io_.input_ptrs.data(),
                synth_decode_inputs_.data(),
                synth_decode_inputs_.size(),
                decode_io_.output_ptrs.data(),
                decode_io_.output_ptrs.size());
            if (benchmark_enabled_) {
                const auto decode_end = std::chrono::steady_clock::now();
                benchmark_stats_.decode_step_ms += std::chrono::duration<double, std::milli>(decode_end - decode_start).count();
                benchmark_stats_.decode_step_calls += 1;
            }
            const float* dec_hidden = dec[0].GetTensorData<float>();
            std::copy(dec_hidden, dec_hidden + config_.hidden_size, synth_h_.begin());
            for (int i = 0; i < config_.num_hidden_layers; ++i) {
                past_k[static_cast<size_t>(i)] = std::move(dec[1 + i]);
            }
            for (int i = 0; i < config_.num_hidden_layers; ++i) {
                past_v[static_cast<size_t>(i)] = std::move(dec[1 + config_.num_hidden_layers + i]);
            }
        }

        if (frames.empty()) {
            error = "VieNeu v3 synthesis produced no acoustic frames.";
            print_benchmark_stats();
            return false;
        }
        const bool ok = decode_codes(frames, static_cast<int64_t>(frames.size() / config_.n_vq), out_audio, error);
        print_benchmark_stats();
        return ok;
    } catch (const std::exception& e) {
        print_benchmark_stats();
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
