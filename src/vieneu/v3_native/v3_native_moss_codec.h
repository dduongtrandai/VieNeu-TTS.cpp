#ifndef V3_NATIVE_MOSS_CODEC_H
#define V3_NATIVE_MOSS_CODEC_H

#include <vector>
#include <string>
#include <memory>
#include "onnxruntime_cxx_api.h"

struct V3CodecParams {
    std::string codec_dir;
    int n_threads = 4;
};

class V3NativeMossCodec {
public:
    V3NativeMossCodec();
    ~V3NativeMossCodec();

    bool initialize(const V3CodecParams& params, std::string& error);
    void free_resources();

    // Decode generated codes (shape: T * 16) to mono waveform
    bool decode(const std::vector<int32_t>& codes, int64_t frame_count, std::vector<float>& out_audio, std::string& error);

    // Encode mono waveform to reference codes (shape: T * 16)
    bool encode(const std::vector<float>& waveform, std::vector<int64_t>& out_codes, std::string& error);

private:
    std::shared_ptr<Ort::Env> env_;
    std::unique_ptr<Ort::SessionOptions> session_options_;
    std::unique_ptr<Ort::Session> decode_session_;
    std::unique_ptr<Ort::Session> encode_session_;

    std::vector<std::string> decode_in_names_;
    std::vector<std::string> decode_out_names_;
    std::vector<const char*> decode_in_ptrs_;
    std::vector<const char*> decode_out_ptrs_;

    std::vector<std::string> encode_in_names_;
    std::vector<std::string> encode_out_names_;
    std::vector<const char*> encode_in_ptrs_;
    std::vector<const char*> encode_out_ptrs_;
};

#endif // V3_NATIVE_MOSS_CODEC_H
