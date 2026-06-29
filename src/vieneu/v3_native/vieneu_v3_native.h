#ifndef VIENEU_V3_NATIVE_H
#define VIENEU_V3_NATIVE_H

#include <memory>
#include <string>
#include <vector>
#include <mutex>
#include "v3_native_config.h"
#include "v3_native_assets.h"
#include "v3_native_tokenizer.h"
#include "v3_native_prompt.h"
#include "v3_native_sampler.h"
#include "v3_native_backbone_llama.h"
#include "v3_native_acoustic_ggml.h"
#include "v3_native_moss_codec.h"
#include "../v3_common/v3_repetition_history.h"

struct VieneuV3NativeInit {
    std::string model_dir;
    std::string onnx_dir;
    std::string codec_dir;
    std::string config_path;
    std::string tokenizer_path;
    std::string voices_json_path;
    int n_threads = 4;
};

struct VieneuV3NativeParams {
    std::string text;
    std::string voice_id;
    std::string ref_audio_path;
    float temperature = 0.8f;
    int top_k = 25;
    float top_p = 0.95f;
    int max_new_frames = 300;
    float repetition_penalty = 1.2f;
    int max_chars = 384;
    bool apply_watermark = true;
};

class VieneuV3NativeEngine {
public:
    VieneuV3NativeEngine();
    ~VieneuV3NativeEngine();

    bool initialize(const VieneuV3NativeInit& init, std::string& error);
    bool synthesize(const VieneuV3NativeParams& params, std::vector<float>& out_audio, std::string& error);
    bool encode_reference(const std::string& ref_audio_path, std::vector<int64_t>& out_codes, std::string& error);

    const std::string& voices_json() const { return assets_.voices_json(); }
    int sample_rate() const { return 48000; }

private:
    struct VoicePreset {
        bool found = false;
        bool has_reserved_id = false;
        int reserved_id = 0;
        std::vector<int64_t> codes;
    };

    bool resolve_voice_preset(const std::string& voice_id, VoicePreset& preset, std::string& error) const;
    bool parse_voice_reserved_id(const std::string& voice_id, int& reserved_id) const;
    std::string phonemize_for_v3(const std::string& text) const;
    bool synthesize_phonemes(const std::string& phonemes,
                             const std::vector<int64_t>* ref_codes,
                             int leading_token,
                             const VieneuV3NativeParams& params,
                             std::vector<float>& out_audio,
                             std::string& error);

    V3NativeConfig config_;
    V3NativeAssets assets_;
    V3NativeTokenizer tokenizer_;
    std::unique_ptr<V3NativePrompt> prompt_builder_;
    V3NativeSampler sampler_;
    V3NativeBackbone backbone_;
    std::unique_ptr<V3NativeAcoustic> acoustic_;
    V3NativeMossCodec codec_;
    std::vector<float> prompt_embeds_;

    std::mutex run_mutex_;
    std::unordered_map<std::string, VoicePreset> voice_presets_;
    std::string default_voice_id_;
    bool initialized_ = false;
};

#endif // VIENEU_V3_NATIVE_H
