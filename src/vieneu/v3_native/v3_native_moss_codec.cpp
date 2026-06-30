#include "v3_native_moss_codec.h"
#include <fstream>
#include <iostream>
#include <cstring>
#include <stdexcept>
#include <cmath>
#include <algorithm>
#include <chrono>

namespace {

struct LocalWavData {
    int sample_rate = 0;
    int channels = 0;
    std::vector<float> samples;
};

// Endian helper functions for WAV reading
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

std::vector<std::string> get_session_input_names(Ort::Session& session) {
    std::vector<std::string> names;
    Ort::AllocatorWithDefaultOptions allocator;
    size_t count = session.GetInputCount();
    names.reserve(count);
    for (size_t i = 0; i < count; ++i) {
        auto name_alloc = session.GetInputNameAllocated(i, allocator);
        names.push_back(std::string(name_alloc.get()));
    }
    return names;
}

std::vector<std::string> get_session_output_names(Ort::Session& session) {
    std::vector<std::string> names;
    Ort::AllocatorWithDefaultOptions allocator;
    size_t count = session.GetOutputCount();
    names.reserve(count);
    for (size_t i = 0; i < count; ++i) {
        auto name_alloc = session.GetOutputNameAllocated(i, allocator);
        names.push_back(std::string(name_alloc.get()));
    }
    return names;
}

std::vector<const char*> convert_to_ptrs(const std::vector<std::string>& names) {
    std::vector<const char*> ptrs;
    ptrs.reserve(names.size());
    for (const auto& s : names) {
        ptrs.push_back(s.c_str());
    }
    return ptrs;
}

std::vector<int64_t> get_tensor_shape(const Ort::Value& value) {
    auto info = value.GetTensorTypeAndShapeInfo();
    return info.GetShape();
}

} // namespace

V3NativeMossCodec::V3NativeMossCodec() {}

V3NativeMossCodec::~V3NativeMossCodec() {
    free_resources();
}

bool V3NativeMossCodec::initialize(const V3CodecParams& params, std::string& error) {
    try {
        free_resources();

        env_ = std::make_shared<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "V3NativeMossCodec");
        session_options_ = std::make_unique<Ort::SessionOptions>();
        session_options_->SetIntraOpNumThreads(params.n_threads);
        session_options_->SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        n_vq_ = params.n_vq > 0 ? params.n_vq : 16;

        std::string decode_path = join_paths(params.codec_dir, "moss_audio_tokenizer_decode_full.onnx");
        std::string encode_path = join_paths(params.codec_dir, "moss_audio_tokenizer_encode.onnx");

#ifdef _WIN32
        std::wstring w_decode_path(decode_path.begin(), decode_path.end());
        decode_session_ = std::make_unique<Ort::Session>(*env_, w_decode_path.c_str(), *session_options_);
        std::wstring w_encode_path(encode_path.begin(), encode_path.end());
        encode_session_ = std::make_unique<Ort::Session>(*env_, w_encode_path.c_str(), *session_options_);
#else
        decode_session_ = std::make_unique<Ort::Session>(*env_, decode_path.c_str(), *session_options_);
        encode_session_ = std::make_unique<Ort::Session>(*env_, encode_path.c_str(), *session_options_);
#endif

        decode_in_names_ = get_session_input_names(*decode_session_);
        decode_out_names_ = get_session_output_names(*decode_session_);
        decode_in_ptrs_ = convert_to_ptrs(decode_in_names_);
        decode_out_ptrs_ = convert_to_ptrs(decode_out_names_);

        encode_in_names_ = get_session_input_names(*encode_session_);
        encode_out_names_ = get_session_output_names(*encode_session_);
        encode_in_ptrs_ = convert_to_ptrs(encode_in_names_);
        encode_out_ptrs_ = convert_to_ptrs(encode_out_names_);

        return true;
    } catch (const std::exception& e) {
        error = std::string("MOSS Codec initialization error: ") + e.what();
        return false;
    }
}

void V3NativeMossCodec::free_resources() {
    encode_session_.reset();
    decode_session_.reset();
    session_options_.reset();
    env_.reset();

    decode_in_names_.clear();
    decode_out_names_.clear();
    decode_in_ptrs_.clear();
    decode_out_ptrs_.clear();

    encode_in_names_.clear();
    encode_out_names_.clear();
    encode_in_ptrs_.clear();
    encode_out_ptrs_.clear();
}

bool V3NativeMossCodec::decode(const std::vector<int32_t>& codes, int64_t frame_count, std::vector<float>& out_audio, std::string& error) {
    if (!decode_session_) {
        error = "MOSS Codec decode session is not initialized.";
        return false;
    }
    try {
        Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        
        std::vector<int32_t> lengths = {static_cast<int32_t>(frame_count)};
        if (static_cast<int64_t>(codes.size()) != frame_count * n_vq_) {
            error = "MOSS codec decode code count does not match frame_count * n_vq.";
            return false;
        }

        std::vector<int64_t> codes_shape = {1, frame_count, n_vq_};
        std::vector<int64_t> len_shape = {1};

        if (decode_in_names_.size() != 2 || decode_out_names_.empty()) {
            error = "MOSS codec decode ONNX signature mismatch: expected 2 inputs and at least 1 output.";
            return false;
        }

        std::vector<Ort::Value> inputs;
        inputs.emplace_back(Ort::Value::CreateTensor<int32_t>(mem, const_cast<int32_t*>(codes.data()), codes.size(), codes_shape.data(), codes_shape.size()));
        inputs.emplace_back(Ort::Value::CreateTensor<int32_t>(mem, lengths.data(), lengths.size(), len_shape.data(), len_shape.size()));

        auto out = decode_session_->Run(
            Ort::RunOptions{nullptr},
            decode_in_ptrs_.data(),
            inputs.data(),
            inputs.size(),
            decode_out_ptrs_.data(),
            decode_out_ptrs_.size());

        const std::vector<int64_t> shape = get_tensor_shape(out[0]);
        const float* audio_data = out[0].GetTensorData<float>();

        if (shape.size() == 3 && shape[0] == 1) {
            const int64_t channels = shape[1];
            const int64_t samples = shape[2];
            out_audio.assign(static_cast<size_t>(samples), 0.0f);
            for (int64_t c = 0; c < channels; ++c) {
                const float* src = audio_data + c * samples;
                for (int64_t i = 0; i < samples; ++i) {
                    out_audio[static_cast<size_t>(i)] += src[i] / static_cast<float>(channels);
                }
            }
        } else {
            size_t count = 1;
            for (int64_t dim : shape) {
                count *= static_cast<size_t>(dim);
            }
            out_audio.assign(audio_data, audio_data + count);
        }
        return true;
    } catch (const std::exception& e) {
        error = std::string("MOSS codec decode failed: ") + e.what();
        return false;
    }
}

bool V3NativeMossCodec::encode(const std::vector<float>& waveform, std::vector<int64_t>& out_codes, std::string& error) {
    const int64_t frames = static_cast<int64_t>(waveform.size());
    if (frames <= 0) {
        error = "MOSS codec encode received an empty waveform.";
        return false;
    }
    std::vector<float> stereo(static_cast<size_t>(2 * frames), 0.0f);
    std::copy(waveform.begin(), waveform.end(), stereo.begin());
    std::copy(waveform.begin(), waveform.end(), stereo.begin() + frames);
    return encode_stereo(stereo, frames, out_codes, error);
}

bool V3NativeMossCodec::encode_stereo(const std::vector<float>& stereo_waveform, int64_t frame_count, std::vector<int64_t>& out_codes, std::string& error) {
    // encode is not required for synthesis preset voice. It is only required for voice cloning.
    // However we implement it using moss_audio_tokenizer_encode.onnx.
    out_codes.clear();
    if (!encode_session_) {
        error = "MOSS Codec encode session is not initialized.";
        return false;
    }
    if (frame_count <= 0 || static_cast<int64_t>(stereo_waveform.size()) != frame_count * 2) {
        error = "MOSS codec encode received invalid stereo waveform shape.";
        return false;
    }
    try {
        Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

        std::vector<int32_t> lengths = {static_cast<int32_t>(frame_count)};
        std::vector<int64_t> wav_shape = {1, 2, frame_count};
        std::vector<int64_t> len_shape = {1};

        if (encode_in_names_.size() != 2 || encode_out_names_.empty()) {
            error = "MOSS codec encode ONNX signature mismatch: expected 2 inputs and at least 1 output.";
            return false;
        }

        std::vector<Ort::Value> inputs;
        inputs.emplace_back(Ort::Value::CreateTensor<float>(
            mem,
            const_cast<float*>(stereo_waveform.data()),
            stereo_waveform.size(),
            wav_shape.data(),
            wav_shape.size()));
        inputs.emplace_back(Ort::Value::CreateTensor<int32_t>(mem, lengths.data(), lengths.size(), len_shape.data(), len_shape.size()));

        auto out = encode_session_->Run(
            Ort::RunOptions{nullptr},
            encode_in_ptrs_.data(),
            inputs.data(),
            inputs.size(),
            encode_out_ptrs_.data(),
            encode_out_ptrs_.size());

        const std::vector<int64_t> shape = get_tensor_shape(out[0]);
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

        if (shape[0] == 1 && shape[2] == n_vq_) {
            out_codes = std::move(raw);
        } else if (shape[0] == n_vq_ && shape[1] == 1) {
            const int64_t num_frames = shape[2];
            out_codes.resize(static_cast<size_t>(num_frames * n_vq_));
            for (int ch = 0; ch < n_vq_; ++ch) {
                for (int64_t t = 0; t < num_frames; ++t) {
                    out_codes[static_cast<size_t>(t * n_vq_ + ch)] =
                        raw[static_cast<size_t>(ch * num_frames + t)];
                }
            }
        } else if (shape[0] == 1 && shape[1] == n_vq_) {
            const int64_t num_frames = shape[2];
            out_codes.resize(static_cast<size_t>(num_frames * n_vq_));
            for (int ch = 0; ch < n_vq_; ++ch) {
                for (int64_t t = 0; t < num_frames; ++t) {
                    out_codes[static_cast<size_t>(t * n_vq_ + ch)] =
                        raw[static_cast<size_t>(ch * num_frames + t)];
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
