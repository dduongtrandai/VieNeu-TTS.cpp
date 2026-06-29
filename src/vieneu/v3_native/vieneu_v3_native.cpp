#include "vieneu_v3_native.h"
#include "vieneu/vieneu.h"
#include <iostream>
#include <fstream>
#include <cstring>
#include <cmath>
#include <cstdlib>
#include <cctype>
#include <algorithm>
#include <stdexcept>
#include <chrono>
#include <nlohmann/json.hpp>

static bool file_exists_local(const std::string& path) {
    if (path.empty()) return false;
    std::ifstream fs(path, std::ios::binary);
    return fs.good();
}

static std::string join_paths(const std::string& a, const std::string& b) {
    if (a.empty()) return b;
    const char last = a[a.size() - 1];
    if (last == '/' || last == '\\') return a + b;
#ifdef _WIN32
    return a + "\\" + b;
#else
    return a + "/" + b;
#endif
}

static bool env_flag_enabled_local(const char* name) {
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

static int effective_chunk_chars_for_frame_budget(int requested_max_chars, int max_new_frames) {
    const int max_chars = (std::max)(16, requested_max_chars);
    if (max_new_frames <= 0) {
        return max_chars;
    }

    // max_new_frames is applied per chunk. If a long text chunk is allowed to be
    // much larger than the frame budget, generation can hit the frame cap before
    // EOS and the resulting audio sounds like it skipped the tail of the text.
    const int frame_limited_chars = (std::max)(64, max_new_frames);
    return (std::min)(max_chars, frame_limited_chars);
}

VieneuV3NativeEngine::VieneuV3NativeEngine() {}

VieneuV3NativeEngine::~VieneuV3NativeEngine() {}

bool VieneuV3NativeEngine::initialize(const VieneuV3NativeInit& init, std::string& error) {
    try {
        if (!assets_.load(init.model_dir, error)) {
            return false;
        }

        config_ = assets_.config();

        std::string tok_path = init.tokenizer_path.empty() ? join_paths(init.model_dir, "tokenizer.json") : init.tokenizer_path;
        if (!tokenizer_.load(tok_path, error)) {
            return false;
        }

        prompt_builder_ = std::make_unique<V3NativePrompt>(config_, tokenizer_, assets_);
        acoustic_ = std::make_unique<V3NativeAcoustic>(config_, assets_, sampler_, init.n_threads);

        if (!acoustic_->initialize(error)) {
            return false;
        }

        // Initialize llama backbone
        V3BackboneParams backbone_params;
        backbone_params.model_path = join_paths(init.model_dir, "backbone.gguf");
        if (!file_exists_local(backbone_params.model_path) && !init.onnx_dir.empty()) {
            backbone_params.model_path = join_paths(init.onnx_dir, "backbone.gguf");
        }
        backbone_params.n_threads = init.n_threads;
        backbone_params.n_threads_batch = init.n_threads;
        if (!backbone_.initialize(backbone_params)) {
            error = "Failed to initialize llama.cpp Qwen3 backbone model: " + backbone_params.model_path;
            return false;
        }

        // Initialize MOSS codec
        V3CodecParams codec_params;
        codec_params.codec_dir = init.codec_dir;
        codec_params.n_threads = init.n_threads;
        if (!codec_.initialize(codec_params, error)) {
            return false;
        }

        // Parse voices presets
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
                if (item.contains("reserved_id") && !item.at("reserved_id").is_null()) {
                    preset.has_reserved_id = true;
                    preset.reserved_id = item.at("reserved_id").get<int>();
                }
                if (item.contains("codes") && item.at("codes").is_array()) {
                    const auto& codes = item.at("codes");
                    for (const auto& row : codes) {
                        for (const auto& v : row) {
                            preset.codes.push_back(v.get<int64_t>());
                        }
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

bool VieneuV3NativeEngine::synthesize(const VieneuV3NativeParams& params, std::vector<float>& out_audio, std::string& error) {
    if (!initialized_) {
        error = "Engine not initialized.";
        return false;
    }

    // Chunk text. Keep chunk size in sympathy with the per-chunk frame budget so
    // explicit low --max-new-frames values do not silently truncate long chunks.
    const int effective_max_chars = effective_chunk_chars_for_frame_budget(params.max_chars, params.max_new_frames);
    const std::vector<std::string> chunks = chunk_text_v3(params.text, effective_max_chars);
    if (chunks.empty()) {
        error = "Text chunking produced no text chunks.";
        return false;
    }

    out_audio.clear();

    // Resolve voice preset
    VoicePreset voice_preset;
    if (!resolve_voice_preset(params.voice_id, voice_preset, error)) {
        return false;
    }

    // Resolve reference voice coding
    std::vector<int64_t> ref_codes;
    const std::vector<int64_t>* p_ref_codes = nullptr;
    if (!params.ref_audio_path.empty()) {
        if (!encode_reference(params.ref_audio_path, ref_codes, error)) {
            return false;
        }
        p_ref_codes = &ref_codes;
    } else if (voice_preset.found && !voice_preset.codes.empty()) {
        p_ref_codes = &voice_preset.codes;
    }

    // Resolve leading speaker token
    int leading_token = 16; // default fallback
    int reserved_id = 0;
    if (parse_voice_reserved_id(params.voice_id.empty() ? default_voice_id_ : params.voice_id, reserved_id)) {
        leading_token = reserved_id;
    }

    const int silence_samples = static_cast<int>(std::lround(static_cast<double>(sample_rate()) * 0.15));
    for (size_t i = 0; i < chunks.size(); ++i) {
        const std::string phonemes = phonemize_for_v3(chunks[i]);
        std::vector<float> chunk_audio;
        if (!synthesize_phonemes(phonemes, p_ref_codes, leading_token, params, chunk_audio, error)) {
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

    return true;
}

bool VieneuV3NativeEngine::synthesize_phonemes(
    const std::string& phonemes,
    const std::vector<int64_t>* ref_codes,
    int leading_token,
    const VieneuV3NativeParams& params,
    std::vector<float>& out_audio,
    std::string& error) {

    out_audio.clear();

    const V3PromptRows rows = prompt_builder_->build_rows(phonemes, ref_codes, leading_token);
    prompt_builder_->embed_rows_into(rows, prompt_embeds_);

    std::lock_guard<std::mutex> lock(run_mutex_);
    try {
        const bool benchmark_enabled = env_flag_enabled_local("VIENEU_V3_NATIVE_BENCHMARK");
        std::vector<float> synth_h;
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

        std::vector<V3RepetitionHistory> history;
        if (std::fabs(params.repetition_penalty - 1.0f) > 1e-6f) {
            history.resize(static_cast<size_t>(config_.n_vq));
            for (auto& item : history) {
                item.initialize(static_cast<size_t>(config_.audio_vocab_size));
            }
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
        if (benchmark_enabled) {
            acoustic_->reset_benchmark_stats();
        }

        for (int t = 0; t < max_frames; ++t) {
            actual_steps++;
            auto t_ac_start = benchmark_enabled ? std::chrono::high_resolution_clock::now() : std::chrono::high_resolution_clock::time_point{};
            bool eos = false;
            if (!acoustic_->generate_frame(synth_h, params.temperature, params.top_k, params.top_p, params.repetition_penalty, history, codes, eos, error)) {
                return false;
            }
            if (benchmark_enabled) {
                auto t_ac_end = std::chrono::high_resolution_clock::now();
                acoustic_ms += std::chrono::duration<double, std::milli>(t_ac_end - t_ac_start).count();
            }

            for (int64_t code : codes) {
                frames.push_back(static_cast<int32_t>(code));
            }
            if (eos) {
                saw_eos = true;
                break;
            }

            // Build next step input slot embedding: text generation start embedding + audio embeddings
            prompt_builder_->embed_slot_into(codes, synth_se);
            const float* sgs = assets_.text_emb().data() + config_.speech_generation_start_token_id * H;
            for (int h = 0; h < H; ++h) {
                synth_se[h] += sgs[h];
            }

            // Incremental decode step in backbone
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
            std::cerr << "[V3NativeEngine] Warning: acoustic generation reached max_new_frames="
                      << max_frames
                      << " before EOS. Increase --max-new-frames or lower --max-chars if audio sounds truncated.\n";
        }

        // MOSS decode
        auto t_codec_start = benchmark_enabled ? std::chrono::high_resolution_clock::now() : std::chrono::high_resolution_clock::time_point{};
        bool ok = codec_.decode(frames, static_cast<int64_t>(frames.size() / config_.n_vq), out_audio, error);
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

struct LocalWavData {
    int sample_rate = 0;
    int channels = 0;
    std::vector<float> samples;
};

static uint16_t wav_read_u16_le(const uint8_t* p) {
    return static_cast<uint16_t>(p[0]) | (static_cast<uint16_t>(p[1]) << 8);
}

static uint32_t wav_read_u32_le(const uint8_t* p) {
    return static_cast<uint32_t>(p[0]) |
           (static_cast<uint32_t>(p[1]) << 8) |
           (static_cast<uint32_t>(p[2]) << 16) |
           (static_cast<uint32_t>(p[3]) << 24);
}

static int16_t wav_read_i16_le(const uint8_t* p) {
    return static_cast<int16_t>(wav_read_u16_le(p));
}

static int32_t wav_read_i24_le(const uint8_t* p) {
    int32_t v = static_cast<int32_t>(p[0]) |
                (static_cast<int32_t>(p[1]) << 8) |
                (static_cast<int32_t>(p[2]) << 16);
    if (v & 0x00800000) {
        v |= static_cast<int32_t>(0xFF000000);
    }
    return v;
}

static int32_t wav_read_i32_le(const uint8_t* p) {
    return static_cast<int32_t>(wav_read_u32_le(p));
}

static bool local_read_wav_file(const std::string& path, LocalWavData& wav, std::string& error) {
    wav = LocalWavData{};
    try {
        std::ifstream fs(path, std::ios::binary);
        if (!fs.is_open()) {
            error = "Failed to open WAV: " + path;
            return false;
        }
        fs.seekg(0, std::ios::end);
        size_t size = fs.tellg();
        fs.seekg(0, std::ios::beg);
        std::vector<uint8_t> bytes(size);
        fs.read(reinterpret_cast<char*>(bytes.data()), size);

        const auto* data = bytes.data();
        if (size < 44 || std::memcmp(data, "RIFF", 4) != 0 || std::memcmp(data + 8, "WAVE", 4) != 0) {
            error = "Reference audio must be a RIFF/WAVE file: " + path;
            return false;
        }

        uint16_t audio_format = 0;
        uint16_t channels = 0;
        uint32_t sample_rate = 0;
        uint16_t bits_per_sample = 0;
        const uint8_t* pcm = nullptr;
        size_t pcm_size = 0;
        size_t off = 12;
        while (off + 8 <= size) {
            const char* id = reinterpret_cast<const char*>(data + off);
            const uint32_t chunk_size = wav_read_u32_le(data + off + 4);
            const size_t payload = off + 8;
            if (payload + chunk_size > size) {
                error = "Truncated WAV chunk in reference audio: " + path;
                return false;
            }
            if (std::memcmp(id, "fmt ", 4) == 0) {
                if (chunk_size < 16) {
                    error = "Invalid WAV fmt chunk in reference audio: " + path;
                    return false;
                }
                audio_format = wav_read_u16_le(data + payload);
                channels = wav_read_u16_le(data + payload + 2);
                sample_rate = wav_read_u32_le(data + payload + 4);
                bits_per_sample = wav_read_u16_le(data + payload + 14);
            } else if (std::memcmp(id, "data", 4) == 0) {
                pcm = data + payload;
                pcm_size = chunk_size;
            }
            off = payload + chunk_size + (chunk_size & 1u);
        }

        if (!pcm || pcm_size == 0 || channels == 0 || sample_rate == 0 || bits_per_sample == 0) {
            error = "Reference WAV is missing fmt/data chunks: " + path;
            return false;
        }
        if (audio_format != 1 && audio_format != 3) {
            error = "Reference WAV must be PCM or IEEE-float format: " + path;
            return false;
        }
        const size_t bytes_per_sample = bits_per_sample / 8;
        if (bytes_per_sample == 0 || pcm_size < bytes_per_sample * channels) {
            error = "Reference WAV has invalid sample size: " + path;
            return false;
        }

        const size_t frames = pcm_size / (bytes_per_sample * channels);
        wav.sample_rate = static_cast<int>(sample_rate);
        wav.channels = static_cast<int>(channels);
        wav.samples.resize(frames * channels);
        for (size_t i = 0; i < frames * channels; ++i) {
            const uint8_t* p = pcm + i * bytes_per_sample;
            float v = 0.0f;
            if (audio_format == 3 && bits_per_sample == 32) {
                std::memcpy(&v, p, sizeof(float));
            } else if (audio_format == 1 && bits_per_sample == 16) {
                v = static_cast<float>(wav_read_i16_le(p)) / 32768.0f;
            } else if (audio_format == 1 && bits_per_sample == 24) {
                v = static_cast<float>(wav_read_i24_le(p)) / 8388608.0f;
            } else if (audio_format == 1 && bits_per_sample == 32) {
                v = static_cast<float>(wav_read_i32_le(p)) / 2147483648.0f;
            } else if (audio_format == 1 && bits_per_sample == 8) {
                v = (static_cast<float>(*p) - 128.0f) / 128.0f;
            } else {
                error = "Unsupported reference WAV sample format: " + path;
                return false;
            }
            wav.samples[i] = std::clamp(v, -1.0f, 1.0f);
        }
        return true;
    } catch (const std::exception& e) {
        error = std::string("Failed to read reference WAV: ") + e.what();
        return false;
    }
}

bool VieneuV3NativeEngine::encode_reference(const std::string& ref_audio_path, std::vector<int64_t>& out_codes, std::string& error) {
    LocalWavData wav;
    if (!local_read_wav_file(ref_audio_path, wav, error)) {
        return false;
    }

    const int target_sr = 48000;
    const int64_t in_frames = static_cast<int64_t>(wav.samples.size() / wav.channels);
    const int64_t out_frames = wav.sample_rate == target_sr
        ? in_frames
        : static_cast<int64_t>(std::llround(static_cast<double>(in_frames) * target_sr / wav.sample_rate));
    if (out_frames <= 0) {
        error = "Reference WAV contains no samples.";
        return false;
    }

    // Resample to target mono/stereo waveform (reusing resampler logic)
    std::vector<float> resampled(static_cast<size_t>(out_frames), 0.0f);
    for (int64_t i = 0; i < out_frames; ++i) {
        const double src_pos = wav.sample_rate == target_sr
            ? static_cast<double>(i)
            : static_cast<double>(i) * wav.sample_rate / target_sr;
        const int64_t i0 = (std::min)(static_cast<int64_t>(std::floor(src_pos)), in_frames - 1);
        const int64_t i1 = (std::min)(i0 + 1, in_frames - 1);
        const float frac = static_cast<float>(src_pos - static_cast<double>(i0));
        
        // Sum/average channels to mono, then interpolate
        float sum_val_0 = 0.0f;
        float sum_val_1 = 0.0f;
        for (int c = 0; c < wav.channels; ++c) {
            sum_val_0 += wav.samples[static_cast<size_t>(i0 * wav.channels + c)];
            sum_val_1 += wav.samples[static_cast<size_t>(i1 * wav.channels + c)];
        }
        float val0 = sum_val_0 / static_cast<float>(wav.channels);
        float val1 = sum_val_1 / static_cast<float>(wav.channels);
        resampled[static_cast<size_t>(i)] = val0 + (val1 - val0) * frac;
    }

    return codec_.encode(resampled, out_codes, error);
}

std::string VieneuV3NativeEngine::phonemize_for_v3(const std::string& text) const {
    return VieneuProfile::phonemize(text);
}

bool VieneuV3NativeEngine::parse_voice_reserved_id(const std::string& voice_id, int& reserved_id) const {
    static const std::unordered_map<std::string, int> fallback = {
        {"Ngọc Lan", 13}, {"Ngọc Linh", 14}, {"Trúc Ly", 15}, {"Mỹ Duyên", 16},
        {"Xuân Vĩnh", 17}, {"Thái Sơn", 18}, {"Gia Bảo", 19}, {"Đức Trí", 20},
        {"Trọng Hữu", 21}, {"Bình An", 22}
    };
    if (!voice_id.empty()) {
        auto it = fallback.find(voice_id);
        if (it != fallback.end()) {
            reserved_id = it->second;
            return true;
        }
    }
    if (voice_id.empty()) {
        return false;
    }
    auto it = voice_presets_.find(voice_id);
    if (it != voice_presets_.end() && it->second.has_reserved_id) {
        reserved_id = it->second.reserved_id;
        return true;
    }
    return false;
}

bool VieneuV3NativeEngine::resolve_voice_preset(
    const std::string& voice_id,
    VoicePreset& preset,
    std::string& error) const {
    preset = VoicePreset{};
    if (voice_presets_.empty()) {
        return true;
    }

    const std::string selected = voice_id.empty() ? default_voice_id_ : voice_id;
    if (selected.empty()) {
        return true;
    }

    auto it = voice_presets_.find(selected);
    if (it == voice_presets_.end()) {
        error = "VieNeu v3 preset not found: " + selected;
        return false;
    }
    preset = it->second;
    return true;
}
