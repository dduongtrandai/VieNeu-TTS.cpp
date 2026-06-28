#ifndef VIENEU_H
#define VIENEU_H

#include <string>
#include <vector>
#include "backends/llama_backend.h"
#include "codecs/neucodec_onnx.h"

class VieneuProfile {
public:
    static std::string format_prompt(const std::string& phonemes);
    static std::vector<int64_t> extract_speech_ids(const std::string& generated_text);

    // Rule-based Vietnamese G2P
    static std::string phonemize(const std::string& text);

    static bool synthesize(
        LlamaBackend& llama,
        NeuCodecOnnx& decoder,
        const std::string& text,
        const std::vector<float>& voice_embedding,
        float temperature,
        int top_k,
        int max_tokens,
        bool skip_phonemize,
        std::vector<float>& out_audio
    );
};

#endif // VIENEU_H
