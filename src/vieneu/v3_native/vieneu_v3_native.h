#ifndef VIENEU_V3_NATIVE_H
#define VIENEU_V3_NATIVE_H

#include <memory>
#include <string>
#include <unordered_map>
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
#include "v3_native_reference.h"
#include "../v3_common/v3_repetition_history.h"
#include "../vieneu_progress.h"

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
    std::string style = "tu_nhien";
    float temperature = 0.8f;
    int top_k = 25;
    float top_p = 0.95f;
    int max_new_frames = 300;
    float repetition_penalty = 1.2f;
    int max_chars = 384;
    bool denoise_ref = true;
    bool use_ref_codes = true;
    bool apply_watermark = true;
    VieneuProgressFn progress;
    float progress_base = 0.0f;
    float progress_span = 1.0f;
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
        std::string style;
        std::vector<float> speaker_emb;
        std::vector<int64_t> codes;
    };

    bool resolve_voice_preset(const std::string& voice_id, VoicePreset& preset, std::string& error) const;
    int resolve_style_token(const std::string& style) const;
    std::string phonemize_for_v3(const std::string& text) const;
    bool enroll_reference(const std::string& ref_audio_path,
                          bool denoise_ref,
                          bool use_ref_codes,
                          std::vector<float>& speaker_emb,
                          std::vector<int64_t>& ref_codes,
                          std::string& error);
    bool synthesize_phonemes(const std::string& phonemes,
                             const std::vector<int64_t>* ref_codes,
                             const std::vector<float>* speaker_anchor,
                             int style_token_id,
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
    V3NativeSpeakerEncoder speaker_encoder_;
    V3NativeDenoiser denoiser_;
    bool has_denoiser_ = false;
    std::vector<float> prompt_embeds_;

    std::mutex run_mutex_;
    std::unordered_map<std::string, VoicePreset> voice_presets_;
    std::string default_voice_id_;
    std::string model_dir_;
    bool initialized_ = false;
};

#endif // VIENEU_V3_NATIVE_H
