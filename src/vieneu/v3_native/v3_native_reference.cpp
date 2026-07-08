#include "v3_native_reference.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstring>
#include <fstream>
#include <stdexcept>

namespace {

constexpr int kSpeakerRate = 16000;
constexpr int kMelBins = 80;
constexpr int kFrameLength = 400; // 25 ms @ 16 kHz
constexpr int kFrameShift = 160;  // 10 ms @ 16 kHz
constexpr int kFftSize = 512;
constexpr double kPi = 3.14159265358979323846264338327950288;

uint16_t read_u16_le(const uint8_t* p) {
    return static_cast<uint16_t>(p[0]) | (static_cast<uint16_t>(p[1]) << 8);
}

uint32_t read_u32_le(const uint8_t* p) {
    return static_cast<uint32_t>(p[0]) |
           (static_cast<uint32_t>(p[1]) << 8) |
           (static_cast<uint32_t>(p[2]) << 16) |
           (static_cast<uint32_t>(p[3]) << 24);
}

int16_t read_i16_le(const uint8_t* p) { return static_cast<int16_t>(read_u16_le(p)); }
int32_t read_i32_le(const uint8_t* p) { return static_cast<int32_t>(read_u32_le(p)); }

int32_t read_i24_le(const uint8_t* p) {
    int32_t v = static_cast<int32_t>(p[0]) |
                (static_cast<int32_t>(p[1]) << 8) |
                (static_cast<int32_t>(p[2]) << 16);
    if (v & 0x00800000) v |= static_cast<int32_t>(0xFF000000);
    return v;
}

std::vector<std::string> session_input_names(Ort::Session& session) {
    std::vector<std::string> names;
    Ort::AllocatorWithDefaultOptions allocator;
    for (size_t i = 0; i < session.GetInputCount(); ++i) {
        auto name = session.GetInputNameAllocated(i, allocator);
        names.emplace_back(name.get());
    }
    return names;
}

std::vector<std::string> session_output_names(Ort::Session& session) {
    std::vector<std::string> names;
    Ort::AllocatorWithDefaultOptions allocator;
    for (size_t i = 0; i < session.GetOutputCount(); ++i) {
        auto name = session.GetOutputNameAllocated(i, allocator);
        names.emplace_back(name.get());
    }
    return names;
}

std::vector<const char*> ptrs(const std::vector<std::string>& names) {
    std::vector<const char*> out;
    out.reserve(names.size());
    for (const auto& name : names) out.push_back(name.c_str());
    return out;
}

double hz_to_mel(double hz) {
    return 1127.0 * std::log1p(hz / 700.0);
}

double mel_to_hz(double mel) {
    return 700.0 * (std::exp(mel / 1127.0) - 1.0);
}

} // namespace

bool v3_native_file_exists(const std::string& path) {
    if (path.empty()) return false;
    std::ifstream fs(path, std::ios::binary);
    return fs.good();
}

bool v3_read_wav_mono(const std::string& path, V3NativeWaveform& wav, std::string& error) {
    wav = V3NativeWaveform{};
    try {
        std::ifstream fs(path, std::ios::binary);
        if (!fs.is_open()) {
            error = "Failed to open WAV: " + path;
            return false;
        }
        fs.seekg(0, std::ios::end);
        const size_t size = static_cast<size_t>(fs.tellg());
        fs.seekg(0, std::ios::beg);
        std::vector<uint8_t> bytes(size);
        fs.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(size));
        const uint8_t* data = bytes.data();
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
                error = "Truncated WAV chunk: " + path;
                return false;
            }
            if (std::memcmp(id, "fmt ", 4) == 0) {
                if (chunk_size < 16) {
                    error = "Invalid WAV fmt chunk: " + path;
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
        if (!pcm || channels == 0 || sample_rate == 0 || bits_per_sample == 0) {
            error = "Reference WAV is missing fmt/data chunks: " + path;
            return false;
        }
        if (audio_format != 1 && audio_format != 3) {
            error = "Reference WAV must be PCM or IEEE-float: " + path;
            return false;
        }
        const size_t bytes_per_sample = bits_per_sample / 8;
        const size_t frames = pcm_size / (bytes_per_sample * channels);
        wav.sample_rate = static_cast<int>(sample_rate);
        wav.mono.assign(frames, 0.0f);
        for (size_t f = 0; f < frames; ++f) {
            double sum = 0.0;
            for (uint16_t ch = 0; ch < channels; ++ch) {
                const uint8_t* p = pcm + (f * channels + ch) * bytes_per_sample;
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
                    error = "Unsupported WAV sample format: " + path;
                    return false;
                }
                sum += std::clamp(v, -1.0f, 1.0f);
            }
            wav.mono[f] = static_cast<float>(sum / static_cast<double>(channels));
        }
        return !wav.mono.empty();
    } catch (const std::exception& e) {
        error = std::string("Failed to read WAV: ") + e.what();
        return false;
    }
}

std::vector<float> v3_resample_linear(const std::vector<float>& input, int input_rate, int output_rate) {
    if (input.empty() || input_rate <= 0 || output_rate <= 0) return {};
    if (input_rate == output_rate) return input;
    const int64_t out_frames = static_cast<int64_t>(std::llround(static_cast<double>(input.size()) * output_rate / input_rate));
    std::vector<float> out(static_cast<size_t>((std::max<int64_t>)(1, out_frames)));
    for (int64_t i = 0; i < static_cast<int64_t>(out.size()); ++i) {
        const double pos = static_cast<double>(i) * input_rate / output_rate;
        const int64_t i0 = (std::min)(static_cast<int64_t>(std::floor(pos)), static_cast<int64_t>(input.size()) - 1);
        const int64_t i1 = (std::min)(i0 + 1, static_cast<int64_t>(input.size()) - 1);
        const float frac = static_cast<float>(pos - static_cast<double>(i0));
        out[static_cast<size_t>(i)] = input[static_cast<size_t>(i0)] + (input[static_cast<size_t>(i1)] - input[static_cast<size_t>(i0)]) * frac;
    }
    return out;
}

void v3_trim_seconds(V3NativeWaveform& wav, double max_seconds) {
    if (wav.sample_rate <= 0 || max_seconds <= 0.0) return;
    const size_t cap = static_cast<size_t>(std::llround(max_seconds * wav.sample_rate));
    if (wav.mono.size() > cap) wav.mono.resize(cap);
}

bool V3NativeSpeakerEncoder::initialize(const std::string& onnx_path, int n_threads, std::string& error) {
    try {
        env_ = std::make_shared<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "V3NativeSpeakerEncoder");
        session_options_ = std::make_unique<Ort::SessionOptions>();
        session_options_->SetIntraOpNumThreads((std::max)(1, n_threads));
        session_options_->SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
#ifdef _WIN32
        std::wstring wide_path(onnx_path.begin(), onnx_path.end());
        session_ = std::make_unique<Ort::Session>(*env_, wide_path.c_str(), *session_options_);
#else
        session_ = std::make_unique<Ort::Session>(*env_, onnx_path.c_str(), *session_options_);
#endif
        input_names_ = session_input_names(*session_);
        output_names_ = session_output_names(*session_);
        input_ptrs_ = ptrs(input_names_);
        output_ptrs_ = ptrs(output_names_);
        return true;
    } catch (const std::exception& e) {
        error = std::string("Failed to initialize speaker encoder: ") + e.what();
        return false;
    }
}

std::vector<float> V3NativeSpeakerEncoder::extract_fbank(const std::vector<float>& mono, int sample_rate, int64_t& frames, std::string& error) const {
    std::vector<float> wav = v3_resample_linear(mono, sample_rate, kSpeakerRate);
    if (wav.size() < kFrameLength) {
        error = "Reference audio is too short for speaker embedding.";
        return {};
    }
    frames = 1 + (static_cast<int64_t>(wav.size()) - kFrameLength) / kFrameShift;
    if (frames <= 0) {
        error = "Reference audio produced no fbank frames.";
        return {};
    }

    std::vector<float> window(kFrameLength);
    for (int i = 0; i < kFrameLength; ++i) {
        const double hann = 0.5 - 0.5 * std::cos(2.0 * kPi * i / (kFrameLength - 1));
        window[static_cast<size_t>(i)] = static_cast<float>(std::pow(hann, 0.85));
    }

    const int bins = kFftSize / 2 + 1;
    std::vector<std::vector<float>> mel_filters(kMelBins, std::vector<float>(bins, 0.0f));
    const double low_mel = hz_to_mel(20.0);
    const double high_mel = hz_to_mel(kSpeakerRate / 2.0);
    std::vector<int> mel_bins(kMelBins + 2);
    for (int m = 0; m < kMelBins + 2; ++m) {
        const double mel = low_mel + (high_mel - low_mel) * m / (kMelBins + 1);
        const double hz = mel_to_hz(mel);
        mel_bins[static_cast<size_t>(m)] = static_cast<int>(std::floor((kFftSize + 1) * hz / kSpeakerRate));
        mel_bins[static_cast<size_t>(m)] = std::clamp(mel_bins[static_cast<size_t>(m)], 0, bins - 1);
    }
    for (int m = 1; m <= kMelBins; ++m) {
        const int left = mel_bins[static_cast<size_t>(m - 1)];
        const int center = mel_bins[static_cast<size_t>(m)];
        const int right = mel_bins[static_cast<size_t>(m + 1)];
        for (int k = left; k < center; ++k) {
            if (center > left) mel_filters[static_cast<size_t>(m - 1)][static_cast<size_t>(k)] = static_cast<float>(k - left) / static_cast<float>(center - left);
        }
        for (int k = center; k < right; ++k) {
            if (right > center) mel_filters[static_cast<size_t>(m - 1)][static_cast<size_t>(k)] = static_cast<float>(right - k) / static_cast<float>(right - center);
        }
    }

    std::vector<float> features(static_cast<size_t>(frames * kMelBins), 0.0f);
    std::vector<double> mean(kMelBins, 0.0);
    for (int64_t f = 0; f < frames; ++f) {
        std::vector<double> frame(kFftSize, 0.0);
        double dc = 0.0;
        const size_t start = static_cast<size_t>(f * kFrameShift);
        for (int i = 0; i < kFrameLength; ++i) dc += wav[start + static_cast<size_t>(i)];
        dc /= kFrameLength;
        double prev = 0.0;
        for (int i = 0; i < kFrameLength; ++i) {
            const double cur = wav[start + static_cast<size_t>(i)] - dc;
            const double emphasized = (i == 0) ? cur : (cur - 0.97 * prev);
            frame[static_cast<size_t>(i)] = emphasized * window[static_cast<size_t>(i)];
            prev = cur;
        }

        std::vector<double> power(bins, 0.0);
        for (int k = 0; k < bins; ++k) {
            std::complex<double> sum(0.0, 0.0);
            for (int n = 0; n < kFftSize; ++n) {
                const double angle = -2.0 * kPi * k * n / kFftSize;
                sum += frame[static_cast<size_t>(n)] * std::complex<double>(std::cos(angle), std::sin(angle));
            }
            power[static_cast<size_t>(k)] = std::norm(sum);
        }
        for (int m = 0; m < kMelBins; ++m) {
            double e = 0.0;
            for (int k = 0; k < bins; ++k) e += power[static_cast<size_t>(k)] * mel_filters[static_cast<size_t>(m)][static_cast<size_t>(k)];
            const float v = static_cast<float>(std::log((std::max)(e, 1.0e-10)));
            features[static_cast<size_t>(f * kMelBins + m)] = v;
            mean[static_cast<size_t>(m)] += v;
        }
    }
    for (double& v : mean) v /= static_cast<double>(frames);
    for (int64_t f = 0; f < frames; ++f) {
        for (int m = 0; m < kMelBins; ++m) {
            features[static_cast<size_t>(f * kMelBins + m)] -= static_cast<float>(mean[static_cast<size_t>(m)]);
        }
    }
    return features;
}

bool V3NativeSpeakerEncoder::embed(const std::vector<float>& mono, int sample_rate, std::vector<float>& out_embedding, std::string& error) {
    out_embedding.clear();
    if (!session_) {
        error = "Speaker encoder is not initialized.";
        return false;
    }
    int64_t frames = 0;
    std::vector<float> fbank = extract_fbank(mono, sample_rate, frames, error);
    if (fbank.empty()) return false;
    try {
        Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        std::vector<int64_t> shape = {1, frames, kMelBins};
        Ort::Value input = Ort::Value::CreateTensor<float>(mem, fbank.data(), fbank.size(), shape.data(), shape.size());
        auto out = session_->Run(Ort::RunOptions{nullptr}, input_ptrs_.data(), &input, 1, output_ptrs_.data(), output_ptrs_.size());
        const auto info = out[0].GetTensorTypeAndShapeInfo();
        size_t count = 1;
        for (int64_t d : info.GetShape()) count *= static_cast<size_t>(d);
        const float* p = out[0].GetTensorData<float>();
        out_embedding.assign(p, p + count);
        if (out_embedding.size() == 192) return true;
        if (out_embedding.size() > 192) {
            out_embedding.resize(192);
            return true;
        }
        error = "Speaker encoder returned unexpected embedding size.";
        return false;
    } catch (const std::exception& e) {
        error = std::string("Speaker encoder inference failed: ") + e.what();
        return false;
    }
}

bool V3NativeDenoiser::initialize(const std::string& onnx_path, int n_threads, std::string& error) {
    try {
        env_ = std::make_shared<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "V3NativeDenoiser");
        session_options_ = std::make_unique<Ort::SessionOptions>();
        session_options_->SetIntraOpNumThreads((std::max)(1, n_threads));
        session_options_->SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
#ifdef _WIN32
        std::wstring wide_path(onnx_path.begin(), onnx_path.end());
        session_ = std::make_unique<Ort::Session>(*env_, wide_path.c_str(), *session_options_);
#else
        session_ = std::make_unique<Ort::Session>(*env_, onnx_path.c_str(), *session_options_);
#endif
        return true;
    } catch (const std::exception& e) {
        error = std::string("Failed to initialize denoiser: ") + e.what();
        return false;
    }
}

bool V3NativeDenoiser::denoise(const V3NativeWaveform& input, V3NativeWaveform& output, std::string& warning) {
    output = input;
    if (!session_) return false;
    warning = "Native denoiser ONNX session is present, but the STFT/iSTFT denoiser path is not enabled yet; using raw reference audio.";
    return false;
}
