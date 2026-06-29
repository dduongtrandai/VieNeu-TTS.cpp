#include "../vieneu_v3_onnx.h"
#include "vieneu_v3_onnx_internal.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <numeric>
#include <vector>
#include <string>
#include <stdexcept>

// --- VieneuV3OnnxEngine Audio Member Functions ---

bool VieneuV3OnnxEngine::read_wav_file(const std::string& path, WavData& wav, std::string& error) const {
    wav = WavData{};
    try {
        const std::string bytes = read_file_bytes(path);
        const auto* data = reinterpret_cast<const uint8_t*>(bytes.data());
        const size_t size = bytes.size();
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
            const uint32_t chunk_size = read_u32_le(data + off + 4);
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
                audio_format = read_u16_le(data + payload);
                channels = read_u16_le(data + payload + 2);
                sample_rate = read_u32_le(data + payload + 4);
                bits_per_sample = read_u16_le(data + payload + 14);
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
                v = static_cast<float>(read_i16_le(p)) / 32768.0f;
            } else if (audio_format == 1 && bits_per_sample == 24) {
                v = static_cast<float>(read_i24_le(p)) / 8388608.0f;
            } else if (audio_format == 1 && bits_per_sample == 32) {
                v = static_cast<float>(read_i32_le(p)) / 2147483648.0f;
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

bool VieneuV3OnnxEngine::encode_reference_audio(
    const std::string& path,
    std::vector<int64_t>& out_codes,
    std::string& error) {
    out_codes.clear();
    WavData wav;
    if (!read_wav_file(path, wav, error)) {
        return false;
    }

    const int target_sr = sample_rate();
    const int64_t in_frames = static_cast<int64_t>(wav.samples.size() / wav.channels);
    const int64_t out_frames = wav.sample_rate == target_sr
        ? in_frames
        : static_cast<int64_t>(std::llround(static_cast<double>(in_frames) * target_sr / wav.sample_rate));
    if (out_frames <= 0) {
        error = "Reference WAV contains no samples: " + path;
        return false;
    }

    std::vector<float> stereo(static_cast<size_t>(2 * out_frames), 0.0f);
    for (int64_t i = 0; i < out_frames; ++i) {
        const double src_pos = wav.sample_rate == target_sr
            ? static_cast<double>(i)
            : static_cast<double>(i) * wav.sample_rate / target_sr;
        const int64_t i0 = (std::min)(static_cast<int64_t>(std::floor(src_pos)), in_frames - 1);
        const int64_t i1 = (std::min)(i0 + 1, in_frames - 1);
        const float frac = static_cast<float>(src_pos - static_cast<double>(i0));
        for (int c = 0; c < 2; ++c) {
            const int src_c = wav.channels == 1 ? 0 : (std::min)(c, wav.channels - 1);
            const float a = wav.samples[static_cast<size_t>(i0 * wav.channels + src_c)];
            const float b = wav.samples[static_cast<size_t>(i1 * wav.channels + src_c)];
            stereo[static_cast<size_t>(c * out_frames + i)] = a + (b - a) * frac;
        }
    }

    try {
        if (!codec_encode_session_) {
            if (!load_session(codec_encode_path_, codec_encode_session_, error)) {
                return false;
            }
            cache_session_io(*codec_encode_session_, codec_encode_io_);
        }
        Ort::MemoryInfo& mem = cpu_memory_info();
        if (codec_encode_io_.input_names.size() != 2 || codec_encode_io_.output_names.empty()) {
            error = "MOSS codec encode ONNX signature mismatch: expected 2 inputs and at least 1 output.";
            return false;
        }

        std::vector<int32_t> lengths = {static_cast<int32_t>(out_frames)};
        std::vector<int64_t> wav_shape = {1, 2, out_frames};
        std::vector<int64_t> len_shape = {1};
        std::vector<Ort::Value> inputs;
        inputs.emplace_back(Ort::Value::CreateTensor<float>(mem, stereo.data(), stereo.size(), wav_shape.data(), wav_shape.size()));
        inputs.emplace_back(Ort::Value::CreateTensor<int32_t>(mem, lengths.data(), lengths.size(), len_shape.data(), len_shape.size()));
        auto out = codec_encode_session_->Run(
            Ort::RunOptions{nullptr},
            codec_encode_io_.input_ptrs.data(),
            inputs.data(),
            inputs.size(),
            codec_encode_io_.output_ptrs.data(),
            codec_encode_io_.output_ptrs.size());
        const std::vector<int64_t> shape = tensor_shape(out[0]);
        if (shape.size() != 3) {
            error = "MOSS codec encode returned unexpected rank.";
            return false;
        }

        size_t count = 1;
        for (int64_t dim : shape) {
            count *= static_cast<size_t>(dim);
        }
        std::vector<int64_t> raw(count);
        const auto type = out[0].GetTensorTypeAndShapeInfo().GetElementType();
        if (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
            const int64_t* p = out[0].GetTensorData<int64_t>();
            raw.assign(p, p + count);
        } else if (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) {
            const int32_t* p = out[0].GetTensorData<int32_t>();
            for (size_t i = 0; i < count; ++i) {
                raw[i] = p[i];
            }
        } else {
            error = "MOSS codec encode returned non-integer codes.";
            return false;
        }

        if (shape[0] == 1 && shape[2] == config_.n_vq) {
            out_codes = std::move(raw);
        } else if (shape[0] == config_.n_vq && shape[1] == 1) {
            const int64_t frames = shape[2];
            out_codes.resize(static_cast<size_t>(frames * config_.n_vq));
            for (int ch = 0; ch < config_.n_vq; ++ch) {
                for (int64_t t = 0; t < frames; ++t) {
                    out_codes[static_cast<size_t>(t * config_.n_vq + ch)] =
                        raw[static_cast<size_t>(ch * frames + t)];
                }
            }
        } else if (shape[0] == 1 && shape[1] == config_.n_vq) {
            const int64_t frames = shape[2];
            out_codes.resize(static_cast<size_t>(frames * config_.n_vq));
            for (int ch = 0; ch < config_.n_vq; ++ch) {
                for (int64_t t = 0; t < frames; ++t) {
                    out_codes[static_cast<size_t>(t * config_.n_vq + ch)] =
                        raw[static_cast<size_t>(ch * frames + t)];
                }
            }
        } else {
            error = "MOSS codec encode returned unsupported code shape.";
            return false;
        }
        return !out_codes.empty();
    } catch (const std::exception& e) {
        error = std::string("MOSS codec encode failed: ") + e.what();
        return false;
    }
}

bool VieneuV3OnnxEngine::decode_codes(const std::vector<int64_t>& frames, int64_t frame_count, std::vector<float>& out_audio, std::string& error) {
    try {
        std::vector<int32_t> codes(frames.size());
        for (size_t i = 0; i < frames.size(); ++i) codes[i] = static_cast<int32_t>(frames[i]);
        std::vector<int32_t> lengths = {static_cast<int32_t>(frame_count)};
        std::vector<int64_t> codes_shape = {1, frame_count, config_.n_vq};
        std::vector<int64_t> len_shape = {1};
        Ort::MemoryInfo& mem = cpu_memory_info();
        if (codec_decode_io_.input_names.size() != 2 || codec_decode_io_.output_names.empty()) {
            error = "MOSS codec decode ONNX signature mismatch: expected 2 inputs and at least 1 output.";
            return false;
        }
        std::vector<Ort::Value> inputs;
        inputs.emplace_back(Ort::Value::CreateTensor<int32_t>(mem, codes.data(), codes.size(), codes_shape.data(), codes_shape.size()));
        inputs.emplace_back(Ort::Value::CreateTensor<int32_t>(mem, lengths.data(), lengths.size(), len_shape.data(), len_shape.size()));
        const auto decode_start = benchmark_enabled_ ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
        auto out = codec_decode_session_->Run(
            Ort::RunOptions{nullptr},
            codec_decode_io_.input_ptrs.data(),
            inputs.data(),
            inputs.size(),
            codec_decode_io_.output_ptrs.data(),
            codec_decode_io_.output_ptrs.size());
        if (benchmark_enabled_) {
            const auto decode_end = std::chrono::steady_clock::now();
            benchmark_stats_.codec_decode_ms += std::chrono::duration<double, std::milli>(decode_end - decode_start).count();
            benchmark_stats_.codec_decode_calls += 1;
        }
        TensorBlob audio = copy_float_tensor(out[0]);
        if (audio.shape.size() == 3 && audio.shape[0] == 1) {
            const int64_t channels = audio.shape[1];
            const int64_t samples = audio.shape[2];
            out_audio.assign(static_cast<size_t>(samples), 0.0f);
            for (int64_t c = 0; c < channels; ++c) {
                const float* src = audio.data.data() + c * samples;
                for (int64_t i = 0; i < samples; ++i) out_audio[static_cast<size_t>(i)] += src[i] / static_cast<float>(channels);
            }
        } else {
            out_audio = std::move(audio.data);
        }
        return true;
    } catch (const std::exception& e) {
        error = std::string("MOSS codec decode failed: ") + e.what();
        return false;
    }
}
