#ifndef VIENEU_V3_ONNX_H
#define VIENEU_V3_ONNX_H

#include <memory>
#include <mutex>
#include <random>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "onnxruntime_cxx_api.h"

struct VieneuV3OnnxInit {
    std::string model_dir;
    std::string onnx_dir;
    std::string codec_dir;
    std::string config_path;
    std::string tokenizer_path;
    std::string voices_json_path;
    int n_threads = 2;
};

struct VieneuV3OnnxParams {
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

class VieneuV3OnnxEngine {
public:
    bool initialize(const VieneuV3OnnxInit& init, std::string& error);
    bool synthesize(const VieneuV3OnnxParams& params, std::vector<float>& out_audio, std::string& error);

    const std::string& voices_json() const { return voices_json_; }
    int sample_rate() const { return 48000; }

private:
    struct Tensor2D {
        int64_t rows = 0;
        int64_t cols = 0;
        std::vector<float> data;
    };

    struct Tensor3D {
        int64_t dim0 = 0;
        int64_t dim1 = 0;
        int64_t dim2 = 0;
        std::vector<float> data;
    };

    struct Config {
        int n_vq = 16;
        int hidden_size = 768;
        int num_hidden_layers = 12;
        int audio_pad_token_id = 1024;
        int text_prompt_start_token_id = 3;
        int text_prompt_end_token_id = 4;
        int speech_generation_start_token_id = 5;
        int speech_generation_end_token_id = 6;
        int audio_ref_slot_token_id = 7;
        int emotion_0_token_id = 8;
        int emotion_4_token_id = 12;
        int text_vocab_size = 419;
        int audio_vocab_size = 1024;
        int local_num_attention_heads = 8;
        int local_num_hidden_layers = 2;
        int local_intermediate_size = 2048;
        float rms_norm_eps = 1e-6f;
    };

    struct AcousticLayerWeights {
        std::vector<float> norm1;
        std::vector<float> qkv;
        std::vector<float> q_norm;
        std::vector<float> k_norm;
        std::vector<float> o_proj;
        std::vector<float> norm2;
        std::vector<float> ff_up;
        std::vector<float> ff_gate;
        std::vector<float> ff_down;
    };

    struct AcousticWeights {
        std::vector<float> slot_pos_emb;
        std::vector<float> final_norm;
        std::vector<AcousticLayerWeights> layers;
        bool loaded = false;
    };

    struct PromptRows {
        int64_t rows = 0;
        int64_t cols = 0;
        std::vector<int64_t> data;
    };

    struct WavData {
        int sample_rate = 0;
        int channels = 0;
        std::vector<float> samples; // interleaved
    };

    struct VoicePreset {
        bool found = false;
        bool has_reserved_id = false;
        int reserved_id = 0;
        std::vector<int64_t> codes;
    };

    struct SessionIo {
        std::vector<std::string> input_names;
        std::vector<std::string> output_names;
        std::vector<const char*> input_ptrs;
        std::vector<const char*> output_ptrs;
    };

    struct BenchmarkStats {
        double prefill_ms = 0.0;
        double decode_step_ms = 0.0;
        double acoustic_frame_ms = 0.0;
        double codec_decode_ms = 0.0;
        int64_t prefill_calls = 0;
        int64_t decode_step_calls = 0;
        int64_t acoustic_frame_calls = 0;
        int64_t codec_decode_calls = 0;
    };

    struct ByteBpeTokenizer {
        bool load(const std::string& path, std::string& error);
        std::vector<int64_t> encode(const std::string& text) const;

        std::unordered_map<std::string, int64_t> vocab;
        std::unordered_map<std::string, int> merge_ranks;
        int64_t unk_id = 43;
    };

    class AcousticExecutor {
    public:
        virtual ~AcousticExecutor() = default;
        virtual const char* backend_name() const = 0;
        virtual bool generate_frame(const std::vector<float>& h,
                                    float temperature,
                                    int top_k,
                                    float top_p,
                                    float repetition_penalty,
                                    std::vector<std::unordered_set<int>>& history,
                                    std::vector<int64_t>& codes,
                                    bool& eos,
                                    std::string& error) = 0;
    };

    class OnnxAcousticExecutor;
    class NativeAcousticExecutor;

    static std::string join_path(const std::string& dir, const std::string& name);
    static bool file_exists(const std::string& path);
    static bool read_text_file(const std::string& path, std::string& out);

    bool load_session(const std::string& path, std::unique_ptr<Ort::Session>& session, std::string& error);
    void cache_session_io(Ort::Session& session, SessionIo& io);
    bool validate_assets(const VieneuV3OnnxInit& init, std::string& error);
    bool load_voices(const std::string& voices_path, std::string& error);
    bool load_config(const std::string& path, std::string& error);
    bool load_heads_npz(const std::string& path, std::string& error);
    bool load_acoustic_weights(const std::string& path, std::string& error);
    bool parse_voice_reserved_id(const std::string& voice_id, int& reserved_id) const;
    bool resolve_voice_preset(const std::string& voice_id, VoicePreset& preset, std::string& error) const;
    bool read_wav_file(const std::string& path, WavData& wav, std::string& error) const;
    bool encode_reference_audio(const std::string& path, std::vector<int64_t>& out_codes, std::string& error);

    PromptRows build_rows(const std::string& phonemes, const std::vector<int64_t>* ref_codes, int leading_token) const;
    std::vector<float> embed_rows(const PromptRows& rows) const;
    bool synthesize_phonemes(const std::string& phonemes,
                             const std::vector<int64_t>* ref_codes,
                             int leading_token,
                             const VieneuV3OnnxParams& params,
                             std::vector<float>& out_audio,
                             std::string& error);
    bool initialize_acoustic_executor(std::string& error);
    bool initialize_native_acoustic_executor(std::string& error);
    bool acoustic_frame(const std::vector<float>& h,
                        float temperature,
                        int top_k,
                        float top_p,
                        float repetition_penalty,
                        std::vector<std::unordered_set<int>>& history,
                        std::vector<int64_t>& codes,
                        bool& eos,
                        std::string& error);
    bool acoustic_frame_onnx(const std::vector<float>& h,
                             float temperature,
                             int top_k,
                             float top_p,
                             float repetition_penalty,
                             std::vector<std::unordered_set<int>>& history,
                             std::vector<int64_t>& codes,
                             bool& eos,
                             std::string& error);
    int64_t sample_logits(std::vector<float>& logits,
                          float temperature,
                          int top_k,
                          float top_p,
                          float repetition_penalty,
                          const std::unordered_set<int>* previous);
    bool decode_codes(const std::vector<int32_t>& frames, int64_t frame_count, std::vector<float>& out_audio, std::string& error);
    std::string phonemize_for_v3(const std::string& text) const;
    void reset_benchmark_stats();
    void print_benchmark_stats() const;
    Ort::MemoryInfo& cpu_memory_info();

    std::shared_ptr<Ort::Env> env_;
    std::unique_ptr<Ort::SessionOptions> session_options_;
    std::unique_ptr<Ort::Session> prefill_session_;
    std::unique_ptr<Ort::Session> decode_session_;
    std::unique_ptr<Ort::Session> acoustic_session_;
    std::unique_ptr<Ort::Session> codec_decode_session_;
    std::unique_ptr<Ort::Session> codec_encode_session_;
    std::unique_ptr<AcousticExecutor> acoustic_executor_;
    std::unique_ptr<Ort::MemoryInfo> cpu_memory_info_;
    SessionIo prefill_io_;
    SessionIo decode_io_;
    SessionIo acoustic_io_;
    SessionIo codec_decode_io_;
    SessionIo codec_encode_io_;
    std::string codec_encode_path_;
    std::string voices_json_;
    std::string default_voice_id_;
    std::unordered_map<std::string, VoicePreset> voice_presets_;
    Config config_;
    Tensor2D text_emb_;
    Tensor2D text_emb_t_;
    Tensor3D audio_emb_;
    Tensor3D audio_emb_t_;
    AcousticWeights acoustic_weights_;
    ByteBpeTokenizer tokenizer_;
    std::string model_dir_;
    std::string onnx_dir_;
    std::string codec_dir_;
    std::mutex run_mutex_;
    std::mt19937 rng_;
    bool initialized_ = false;
    bool benchmark_enabled_ = false;
    BenchmarkStats benchmark_stats_;

    // Scratch buffers for sampling to avoid allocation overhead
    std::vector<float> sampling_tmp_;
    std::vector<std::pair<float, size_t>> sampling_pairs_;
    std::vector<float> sampling_probs_;

    std::vector<float> synth_h_;
    std::vector<float> synth_se_;
    std::vector<Ort::Value> synth_decode_inputs_;
    std::vector<float> acoustic_token_;
    std::vector<float> acoustic_empty_;
    std::vector<float> acoustic_slot0_;
    std::vector<float> acoustic_logits_;
    std::vector<float> acoustic_text_logits_;
    std::vector<Ort::Value> acoustic_inputs_;
    std::vector<Ort::Value> acoustic_step_inputs_;
};

#endif // VIENEU_V3_ONNX_H
