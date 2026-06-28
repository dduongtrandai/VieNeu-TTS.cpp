#ifndef NEUCODEC_ONNX_H
#define NEUCODEC_ONNX_H

#include <string>
#include <vector>
#include <memory>
#include "onnxruntime_cxx_api.h"

class NeuCodecOnnx {
public:
    NeuCodecOnnx();
    ~NeuCodecOnnx();

    bool initialize(const std::string& model_path, int n_threads = 4);
    void free_resources();

    // Decoder for VieNeu (takes tokens and voice embedding)
    bool decode_vieneu(const std::vector<int64_t>& speech_tokens, const std::vector<float>& voice_embedding, std::vector<float>& out_audio);

    // Decoder for NeuCodec (takes codes)
    bool decode_neucodec(const std::vector<int32_t>& codes, std::vector<float>& out_audio);

    // Encoder (takes waveform, outputs 128-dim embedding)
    bool encode_speaker(const std::vector<float>& waveform, std::vector<float>& out_embedding_128);

    // Helper to check session input names
    bool has_input(const std::string& name) const;

private:
    std::shared_ptr<Ort::Env> env;
    std::unique_ptr<Ort::Session> session;
    std::vector<std::string> input_names;
    std::vector<std::string> output_names;
};

// Helper function to read WAV file and resample to 24kHz mono
bool read_wav_file_24k_mono(const std::string& path, std::vector<float>& out_samples);

#endif // NEUCODEC_ONNX_H
