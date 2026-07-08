#ifndef V3_NATIVE_CONFIG_H
#define V3_NATIVE_CONFIG_H

#include <string>
#include <unordered_map>

struct V3NativeConfig {
    int n_vq = 16;
    int hidden_size = 768;
    int num_hidden_layers = 12;
    int num_attention_heads = 12;
    int num_key_value_heads = 4;
    int head_dim = 64;
    int intermediate_size = 3072;
    float rope_theta = 1000000.0f;
    float rms_norm_eps = 1e-6f;

    int audio_pad_token_id = 1024;
    int text_prompt_start_token_id = 3;
    int text_prompt_end_token_id = 4;
    int speech_generation_start_token_id = 5;
    int speech_generation_end_token_id = 6;
    int audio_ref_slot_token_id = 7;
    int emotion_0_token_id = 8;
    int emotion_1_token_id = 9;
    int emotion_2_token_id = 10;
    int emotion_3_token_id = 11;
    int emotion_4_token_id = 12;
    int default_style_token_id = 16;
    std::unordered_map<std::string, int> style_labels = {
        {"tu_nhien", 16},
        {"tin_tuc", 17},
        {"doc_truyen", 18},
    };
    int text_vocab_size = 389;
    int audio_vocab_size = 1024;

    int local_num_attention_heads = 8;
    int local_num_hidden_layers = 2;
    int local_intermediate_size = 2048;

    bool use_speaker_embedding = false;
    int speaker_embedding_dim = 192;
    std::string speaker_encoder_filename = "speaker_encoder.onnx";
};

#endif // V3_NATIVE_CONFIG_H
