#ifndef V3_NATIVE_REFERENCE_H
#define V3_NATIVE_REFERENCE_H

#include <memory>
#include <string>
#include <vector>

#include "onnxruntime_cxx_api.h"

struct V3NativeWaveform {
    int sample_rate = 0;
    std::vector<float> mono;
};

bool v3_native_file_exists(const std::string& path);
bool v3_read_wav_mono(const std::string& path, V3NativeWaveform& wav, std::string& error);
std::vector<float> v3_resample_linear(const std::vector<float>& input, int input_rate, int output_rate);
void v3_trim_seconds(V3NativeWaveform& wav, double max_seconds);

class V3NativeSpeakerEncoder {
public:
    bool initialize(const std::string& onnx_path, int n_threads, std::string& error);
    bool initialized() const { return static_cast<bool>(session_); }
    bool embed(const std::vector<float>& mono, int sample_rate, std::vector<float>& out_embedding, std::string& error);

private:
    std::vector<float> extract_fbank(const std::vector<float>& mono, int sample_rate, int64_t& frames, std::string& error) const;

    std::shared_ptr<Ort::Env> env_;
    std::unique_ptr<Ort::SessionOptions> session_options_;
    std::unique_ptr<Ort::Session> session_;
    std::vector<std::string> input_names_;
    std::vector<std::string> output_names_;
    std::vector<const char*> input_ptrs_;
    std::vector<const char*> output_ptrs_;
};

class V3NativeDenoiser {
public:
    bool initialize(const std::string& onnx_path, int n_threads, std::string& error);
    bool initialized() const { return static_cast<bool>(session_); }
    bool denoise(const V3NativeWaveform& input, V3NativeWaveform& output, std::string& warning);

private:
    std::shared_ptr<Ort::Env> env_;
    std::unique_ptr<Ort::SessionOptions> session_options_;
    std::unique_ptr<Ort::Session> session_;
};

#endif // V3_NATIVE_REFERENCE_H
