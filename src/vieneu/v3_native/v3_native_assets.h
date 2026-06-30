#ifndef V3_NATIVE_ASSETS_H
#define V3_NATIVE_ASSETS_H

#include <string>
#include <vector>
#include <unordered_map>
#include "v3_native_config.h"

struct V3NamedArray {
    std::vector<int64_t> shape;
    std::vector<float> data;
};

struct V3AcousticLayerWeights {
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

struct V3AcousticWeights {
    std::vector<float> slot_pos_emb;
    std::vector<float> final_norm;
    std::vector<V3AcousticLayerWeights> layers;
    bool loaded = false;
};

class V3NativeAssets {
public:
    V3NativeAssets() = default;

    bool load(const std::string& model_dir, std::string& error);

    const V3NativeConfig& config() const { return config_; }
    const std::vector<float>& text_emb() const { return text_emb_; }
    const std::vector<float>& audio_emb() const { return audio_emb_; }
    const std::vector<float>& text_emb_t() const { return text_emb_t_; }
    const std::vector<float>& audio_emb_t() const { return audio_emb_t_; }
    const V3AcousticWeights& acoustic_weights() const { return acoustic_weights_; }
    const std::string& voices_json() const { return voices_json_; }

    int text_vocab_size() const { return config_.text_vocab_size; }
    int hidden_size() const { return config_.hidden_size; }

private:
    bool load_config(const std::string& path, std::string& error);
    bool load_heads(const std::string& path, std::string& error);
    bool load_acoustic(const std::string& path, std::string& error);
    bool load_voices(const std::string& path, std::string& error);

    V3NativeConfig config_;
    std::vector<float> text_emb_;
    std::vector<float> audio_emb_; // Packed as [16 * 1024 * 768] (channels, vocab, hidden)
    std::vector<float> text_emb_t_; // Packed as [hidden, text_vocab]
    std::vector<float> audio_emb_t_; // Packed as [n_vq, hidden, audio_vocab]
    V3AcousticWeights acoustic_weights_;
    std::string voices_json_;
};

// Utilities for reading NPZ files
std::unordered_map<std::string, V3NamedArray> load_v3_npz(const std::string& path, std::string& error);

#endif // V3_NATIVE_ASSETS_H
