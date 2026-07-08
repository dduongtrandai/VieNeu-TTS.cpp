#include "vieneu_v3_native.h"
#include "vieneu/vieneu.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>

#include <nlohmann/json.hpp>

namespace {

std::string join_paths(const std::string& a, const std::string& b) {
    if (a.empty()) return b;
    const char last = a[a.size() - 1];
    if (last == '/' || last == '\\') return a + b;
#ifdef _WIN32
    return a + "\\" + b;
#else
    return a + "/" + b;
#endif
}

bool env_flag_enabled_local(const char* name) {
    const char* value = std::getenv(name);
    if (!value || !*value) return false;
    std::string normalized(value);
    std::transform(normalized.begin(), normalized.end(), normalized.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return normalized == "1" || normalized == "true" || normalized == "yes" || normalized == "on";
}

bool contains_v3_emotion_token(const std::string& text) {
    return text.find("<|emotion_1|>") != std::string::npos ||
           text.find("<|emotion_2|>") != std::string::npos ||
           text.find("<|emotion_3|>") != std::string::npos;
}

} // namespace

VieneuV3NativeEngine::VieneuV3NativeEngine() = default;
VieneuV3NativeEngine::~VieneuV3NativeEngine() = default;

bool VieneuV3NativeEngine::initialize(const VieneuV3NativeInit& init, std::string& error) {
    try {
        model_dir_ = init.model_dir;
        if (!assets_.load(init.model_dir, error)) return false;
        config_ = assets_.config();

        const std::string tok_path = init.tokenizer_path.empty() ? join_paths(init.model_dir, "tokenizer.json") : init.tokenizer_path;
        if (!tokenizer_.load(tok_path, error)) return false;

        prompt_builder_ = std::make_unique<V3NativePrompt>(config_, tokenizer_, assets_);
        acoustic_ = std::make_unique<V3NativeAcoustic>(config_, assets_, sampler_, init.n_threads);
        if (!acoustic_->initialize(error)) return false;

        V3BackboneParams backbone_params;
        backbone_params.model_path = join_paths(init.model_dir, "backbone.gguf");
        if (!v3_native_file_exists(backbone_params.model_path) && !init.onnx_dir.empty()) {
            backbone_params.model_path = join_paths(init.onnx_dir, "backbone.gguf");
        }
        backbone_params.n_threads = init.n_threads;
        backbone_params.n_threads_batch = init.n_threads;
        if (!backbone_.initialize(backbone_params)) {
            error = "Failed to initialize llama.cpp Qwen3 backbone model: " + backbone_params.model_path;
            return false;
        }

        V3CodecParams codec_params;
        codec_params.codec_dir = init.codec_dir;
        codec_params.n_threads = init.n_threads;
        codec_params.n_vq = config_.n_vq;
        if (!codec_.initialize(codec_params, error)) return false;

        if (config_.use_speaker_embedding) {
            const std::string spk_path = join_paths(init.model_dir, config_.speaker_encoder_filename.empty() ? "speaker_encoder.onnx" : config_.speaker_encoder_filename);
            if (!v3_native_file_exists(spk_path)) {
                error = "Missing v3 native speaker encoder: " + spk_path;
                return false;
            }
            if (!speaker_encoder_.initialize(spk_path, init.n_threads, error)) return false;
        }

        const std::string den_path = join_paths(init.model_dir, "denoiser.onnx");
        if (v3_native_file_exists(den_path)) {
            std::string den_error;
            has_denoiser_ = denoiser_.initialize(den_path, init.n_threads, den_error);
            if (!has_denoiser_) {
                std::cerr << "[V3NativeEngine] Warning: failed to initialize denoiser: " << den_error << "\n";
            }
        }

        voice_presets_.clear();
        default_voice_id_.clear();
        const auto root = nlohmann::json::parse(assets_.voices_json());
        if (root.contains("default_voice") && root.at("default_voice").is_string()) {
            default_voice_id_ = root.at("default_voice").get<std::string>();
        }
        if (root.contains("presets") && root.at("presets").is_object()) {
            const auto& presets = root.at("presets");
            for (auto it = presets.begin(); it != presets.end(); ++it) {
                const std::string id = it.key();
                const auto& item = it.value();
                VoicePreset preset;
                preset.found = true;
                preset.style = item.value("style", std::string());
                if (item.contains("speaker_emb") && item.at("speaker_emb").is_array()) {
                    for (const auto& v : item.at("speaker_emb")) preset.speaker_emb.push_back(v.get<float>());
                }
                if (config_.use_speaker_embedding && preset.speaker_emb.size() != static_cast<size_t>(config_.speaker_embedding_dim)) {
                    error = "VieNeu v3 native preset voice is missing a valid speaker_emb: " + id;
                    return false;
                }
                if (item.contains("codes") && item.at("codes").is_array() && !item.at("codes").is_null()) {
                    for (const auto& row : item.at("codes")) {
                        if (!row.is_array() || static_cast<int>(row.size()) != config_.n_vq) {
                            error = "VieNeu v3 native preset voice has invalid codes shape: " + id;
                            return false;
                        }
                        for (const auto& v : row) preset.codes.push_back(v.get<int64_t>());
                    }
                }
                voice_presets_[id] = std::move(preset);
            }
        }

        initialized_ = true;
        return true;
    } catch (const std::exception& e) {
        error = std::string("Failed to initialize VieNeu v3 native engine: ") + e.what();
        return false;
    }
}

int VieneuV3NativeEngine::resolve_style_token(const std::string& style) const {
    if (!style.empty()) {
        auto it = config_.style_labels.find(style);
        if (it != config_.style_labels.end()) return it->second;
    }
    return config_.default_style_token_id;
}

bool VieneuV3NativeEngine::resolve_voice_preset(const std::string& voice_id, VoicePreset& preset, std::string& error) const {
    preset = VoicePreset{};
    const std::string selected = voice_id.empty() ? default_voice_id_ : voice_id;
    if (selected.empty()) return true;
    auto it = voice_presets_.find(selected);
    if (it == voice_presets_.end()) {
        error = "VieNeu v3 native voice preset not found: " + selected;
        return false;
    }
    preset = it->second;
    return true;
}

bool VieneuV3NativeEngine::enroll_reference(const std::string& ref_audio_path,
                                            bool denoise_ref,
                                            bool use_ref_codes,
                                            std::vector<float>& speaker_emb,
                                            std::vector<int64_t>& ref_codes,
                                            std::string& error) {
    speaker_emb.clear();
    ref_codes.clear();
    V3NativeWaveform wav;
    if (!v3_read_wav_mono(ref_audio_path, wav, error)) return false;
    v3_trim_seconds(wav, 8.0);

    if (denoise_ref && has_denoiser_) {
        V3NativeWaveform clean;
        std::string warning;
        if (denoiser_.denoise(wav, clean, warning)) {
            wav = std::move(clean);
        } else if (!warning.empty()) {
            std::cerr << "[V3NativeEngine] Warning: " << warning << "\n";
        }
    }

    if (config_.use_speaker_embedding) {
        if (!speaker_encoder_.embed(wav.mono, wav.sample_rate, speaker_emb, error)) return false;
    }

    if (use_ref_codes) {
        std::vector<float> mono48 = v3_resample_linear(wav.mono, wav.sample_rate, sample_rate());
        const int64_t frames = static_cast<int64_t>(mono48.size());
        std::vector<float> stereo(static_cast<size_t>(2 * frames), 0.0f);
        std::copy(mono48.begin(), mono48.end(), stereo.begin());
        std::copy(mono48.begin(), mono48.end(), stereo.begin() + frames);
        if (!codec_.encode_stereo(stereo, frames, ref_codes, error)) return false;
    }
    return true;
}

bool VieneuV3NativeEngine::encode_reference(const std::string& ref_audio_path, std::vector<int64_t>& out_codes, std::string& error) {
    std::vector<float> speaker_emb;
    return enroll_reference(ref_audio_path, true, true, speaker_emb, out_codes, error);
}

bool VieneuV3NativeEngine::synthesize(const VieneuV3NativeParams& params, std::vector<float>& out_audio, std::string& error) {
    vieneu_report_progress(params.progress, "prepare", 0, 0, 0.0f, "Preparing v3 native synthesis.");
    if (!initialized_) {
        error = "Engine not initialized.";
        return false;
    }
    const std::vector<std::string> chunks = chunk_text_v3(params.text, params.max_chars);
    if (chunks.empty()) {
        error = "Text chunking produced no text chunks.";
        return false;
    }

    std::vector<float> speaker_emb;
    std::vector<int64_t> ref_codes;
    if (!params.ref_audio_path.empty()) {
        if (!enroll_reference(params.ref_audio_path, params.denoise_ref, params.use_ref_codes, speaker_emb, ref_codes, error)) return false;
    } else {
        VoicePreset voice_preset;
        if (!resolve_voice_preset(params.voice_id, voice_preset, error)) return false;
        speaker_emb = std::move(voice_preset.speaker_emb);
        if (params.use_ref_codes) ref_codes = std::move(voice_preset.codes);
    }

    if (config_.use_speaker_embedding && speaker_emb.size() != static_cast<size_t>(config_.speaker_embedding_dim)) {
        error = "VieNeu v3 native synthesis requires a valid speaker_emb.";
        return false;
    }
    if (!ref_codes.empty() && ref_codes.size() % static_cast<size_t>(config_.n_vq) != 0) {
        error = "VieNeu v3 native reference codes are not divisible by n_vq.";
        return false;
    }

    std::vector<float> speaker_anchor;
    if (!assets_.compute_speaker_anchor(speaker_emb, speaker_anchor, error)) return false;
    const int style_token_id = resolve_style_token(params.style);

    out_audio.clear();
    const int silence_samples = static_cast<int>(std::lround(static_cast<double>(sample_rate()) * 0.15));
    const bool debug_tags = env_flag_enabled_local("VIENEU_V3_NATIVE_DEBUG_TAGS");
    for (size_t i = 0; i < chunks.size(); ++i) {
        vieneu_report_progress(params.progress, "chunk", static_cast<int>(i), static_cast<int>(chunks.size()), static_cast<float>(i) / static_cast<float>(chunks.size()), "Starting v3 native text chunk.");
        const std::string phonemes = phonemize_for_v3(chunks[i]);
        if (debug_tags || contains_v3_emotion_token(phonemes)) {
            const std::vector<int64_t> phone_ids = tokenizer_.encode(phonemes);
            std::cerr << "[V3NativeTags] chunk " << (i + 1) << "/" << chunks.size() << " text=\"" << chunks[i] << "\"\n";
            std::cerr << "[V3NativeTags] phonemes=\"" << phonemes << "\"\n";
            std::cerr << "[V3NativeTags] phone_ids=";
            for (const int64_t id : phone_ids) std::cerr << id << ' ';
            std::cerr << "\n";
        }
        std::vector<float> chunk_audio;
        VieneuV3NativeParams chunk_params = params;
        chunk_params.progress_base = static_cast<float>(i) / static_cast<float>(chunks.size());
        chunk_params.progress_span = 1.0f / static_cast<float>(chunks.size());
        if (!synthesize_phonemes(
                phonemes,
                ref_codes.empty() ? nullptr : &ref_codes,
                speaker_anchor.empty() ? nullptr : &speaker_anchor,
                style_token_id,
                chunk_params,
                chunk_audio,
                error)) {
            if (chunks.size() > 1) error += " (chunk " + std::to_string(i + 1) + "/" + std::to_string(chunks.size()) + ")";
            return false;
        }
        if (!chunk_audio.empty()) {
            if (!out_audio.empty() && silence_samples > 0) out_audio.insert(out_audio.end(), static_cast<size_t>(silence_samples), 0.0f);
            out_audio.insert(out_audio.end(), chunk_audio.begin(), chunk_audio.end());
        }
        vieneu_report_progress(params.progress, "chunk", static_cast<int>(i + 1), static_cast<int>(chunks.size()), static_cast<float>(i + 1) / static_cast<float>(chunks.size()), "Finished v3 native text chunk.");
    }
    vieneu_report_progress(params.progress, "complete", 1, 1, 1.0f, "V3 native synthesis complete.");
    return true;
}

bool VieneuV3NativeEngine::synthesize_phonemes(
    const std::string& phonemes,
    const std::vector<int64_t>* ref_codes,
    const std::vector<float>* speaker_anchor,
    int style_token_id,
    const VieneuV3NativeParams& params,
    std::vector<float>& out_audio,
    std::string& error) {
    out_audio.clear();
    const V3PromptRows rows = prompt_builder_->build_rows(phonemes, ref_codes, style_token_id);
    prompt_builder_->embed_rows_into(rows, speaker_anchor, prompt_embeds_);

    std::lock_guard<std::mutex> lock(run_mutex_);
    try {
        const bool benchmark_enabled = env_flag_enabled_local("VIENEU_V3_NATIVE_BENCHMARK");
        std::vector<float> synth_h;
        auto scaled_progress = [&params](float local) { return params.progress_base + local * params.progress_span; };
        vieneu_report_progress(params.progress, "prefill", 0, 1, scaled_progress(0.10f), "Running v3 native prompt prefill.");
        auto t_prefill_start = benchmark_enabled ? std::chrono::high_resolution_clock::now() : std::chrono::high_resolution_clock::time_point{};
        if (!backbone_.prefill(prompt_embeds_, synth_h)) {
            error = "Semantic backbone prefill failed.";
            return false;
        }
        double prefill_ms = 0.0;
        if (benchmark_enabled) {
            auto t_prefill_end = std::chrono::high_resolution_clock::now();
            prefill_ms = std::chrono::duration<double, std::milli>(t_prefill_end - t_prefill_start).count();
        }
        vieneu_report_progress(params.progress, "prefill", 1, 1, scaled_progress(0.18f), "V3 native prompt prefill complete.");

        std::vector<V3RepetitionHistory> history;
        if (std::fabs(params.repetition_penalty - 1.0f) > 1e-6f) {
            history.resize(static_cast<size_t>(config_.n_vq));
            for (auto& item : history) item.initialize(static_cast<size_t>(config_.audio_vocab_size));
        }

        std::vector<int32_t> frames;
        const int max_frames = (std::max)(1, params.max_new_frames);
        frames.reserve(static_cast<size_t>(max_frames * config_.n_vq));
        std::vector<int64_t> codes;
        codes.reserve(static_cast<size_t>(config_.n_vq));
        std::vector<float> synth_se;
        synth_se.reserve(static_cast<size_t>(config_.hidden_size));
        const int H = config_.hidden_size;

        double acoustic_ms = 0.0;
        double backbone_decode_ms = 0.0;
        int actual_steps = 0;
        bool saw_eos = false;
        if (benchmark_enabled) acoustic_->reset_benchmark_stats();

        for (int t = 0; t < max_frames; ++t) {
            actual_steps++;
            auto t_ac_start = benchmark_enabled ? std::chrono::high_resolution_clock::now() : std::chrono::high_resolution_clock::time_point{};
            bool eos = false;
            if (!acoustic_->generate_frame(synth_h, params.temperature, params.top_k, params.top_p, params.repetition_penalty, history, codes, eos, error)) return false;
            if (benchmark_enabled) {
                auto t_ac_end = std::chrono::high_resolution_clock::now();
                acoustic_ms += std::chrono::duration<double, std::milli>(t_ac_end - t_ac_start).count();
            }
            for (int64_t code : codes) frames.push_back(static_cast<int32_t>(code));
            vieneu_report_progress(params.progress, "generate_frames", t + 1, max_frames, scaled_progress(0.18f + (static_cast<float>(t + 1) / static_cast<float>(max_frames)) * 0.68f), "Generating v3 native acoustic frames.");
            if (eos) {
                saw_eos = true;
                break;
            }

            prompt_builder_->embed_slot_into(codes, speaker_anchor, synth_se);
            const float* sgs = assets_.text_emb().data() + config_.speech_generation_start_token_id * H;
            for (int h = 0; h < H; ++h) synth_se[static_cast<size_t>(h)] += sgs[h];

            auto t_bb_start = benchmark_enabled ? std::chrono::high_resolution_clock::now() : std::chrono::high_resolution_clock::time_point{};
            if (!backbone_.decode_step(synth_se, synth_h)) {
                error = "Semantic backbone decode step failed.";
                return false;
            }
            if (benchmark_enabled) {
                auto t_bb_end = std::chrono::high_resolution_clock::now();
                backbone_decode_ms += std::chrono::duration<double, std::milli>(t_bb_end - t_bb_start).count();
            }
        }
        if (frames.empty()) {
            error = "VieNeu v3 native synthesis produced no acoustic frames.";
            return false;
        }
        if (!saw_eos) {
            std::cerr << "[V3NativeEngine] Warning: acoustic generation reached max_new_frames=" << max_frames << " before EOS.\n";
        }

        vieneu_report_progress(params.progress, "decode_audio", 0, 1, scaled_progress(0.90f), "Decoding v3 native frames to audio.");
        auto t_codec_start = benchmark_enabled ? std::chrono::high_resolution_clock::now() : std::chrono::high_resolution_clock::time_point{};
        bool ok = codec_.decode(frames, static_cast<int64_t>(frames.size() / config_.n_vq), out_audio, error);
        if (ok) vieneu_report_progress(params.progress, "decode_audio", 1, 1, scaled_progress(0.96f), "V3 native audio decode complete.");
        if (benchmark_enabled) {
            auto t_codec_end = std::chrono::high_resolution_clock::now();
            double codec_ms = std::chrono::duration<double, std::milli>(t_codec_end - t_codec_start).count();
            std::cout << "[V3NativeEngine] Timing Breakdown:\n"
                      << "  Prefill (backbone): " << prefill_ms << " ms\n"
                      << "  Acoustic generation (" << actual_steps << " frames): " << acoustic_ms << " ms (avg " << (acoustic_ms / actual_steps) << " ms/frame)\n"
                      << "  Backbone decode (" << (actual_steps - 1) << " steps): " << backbone_decode_ms << " ms (avg " << (actual_steps > 1 ? (backbone_decode_ms / (actual_steps - 1)) : 0.0) << " ms/step)\n"
                      << "  MOSS codec decode: " << codec_ms << " ms\n";
            acoustic_->print_benchmark_stats();
        }
        return ok;
    } catch (const std::exception& e) {
        error = std::string("VieNeu v3 native synthesis step failed: ") + e.what();
        return false;
    }
}

std::string VieneuV3NativeEngine::phonemize_for_v3(const std::string& text) const {
    return VieneuProfile::phonemize(text);
}
