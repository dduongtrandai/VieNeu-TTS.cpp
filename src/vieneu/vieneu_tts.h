#ifndef VIENEU_TTS_H
#define VIENEU_TTS_H

#include <stdbool.h>
#include <stdint.h>

#ifdef _WIN32
#   if defined(VIENEU_STATIC)
#       define VIENEU_API
#   elif defined(VIENEU_BUILD_DLL)
#       define VIENEU_API __declspec(dllexport)
#   else
#       define VIENEU_API __declspec(dllimport)
#   endif
#else
#   define VIENEU_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

struct vieneu_init_params {
    int          abi_version;
    const char * model_path;
    const char * encoder_path;
    const char * decoder_path;
    const char * voices_json_path;
    int          n_ctx;
    int          n_threads;
    int          n_gpu_layers;
    bool         flash_attn;
    bool         mlock;
};

struct vieneu_audio {
    float * samples;
    int     n_samples;
    int     sample_rate;
    int     channels;
};

struct vieneu_context;

struct vieneu_progress {
    int          abi_version;
    const char * stage;
    int          current;
    int          total;
    float        progress;
    const char * message;
};

typedef void (*vieneu_progress_callback)(const struct vieneu_progress * progress, void * user_data);

struct vieneu_tts_params {
    int           abi_version;
    const char *  text;
    const char *  voice_id;
    const float * voice_embedding;
    float         temperature;
    int           top_k;
    int           max_chars;
    int           max_tokens;
    bool          skip_normalize;
    bool          skip_phonemize;
    bool          apply_watermark;
};

struct vieneu_init_params_v2 {
    int          abi_version;
    const char * profile;
    const char * model_dir;
    const char * onnx_dir;
    const char * codec_dir;
    const char * config_path;
    const char * tokenizer_path;
    const char * voices_json_path;
    int          n_threads;
};

struct vieneu_tts_params_v2 {
    int          abi_version;
    const char * text;
    const char * voice_id;
    const char * ref_audio_path;
    float        temperature;
    int          top_k;
    float        top_p;
    int          max_new_frames;
    float        repetition_penalty;
    int          max_chars;
    bool         apply_watermark;
};

struct vieneu_tts_params_v3 {
    int          abi_version;
    const char * text;
    const char * voice_id;
    const char * ref_audio_path;
    const char * style;
    float        temperature;
    int          top_k;
    float        top_p;
    int          max_new_frames;
    float        repetition_penalty;
    int          max_chars;
    bool         denoise_ref;
    bool         use_ref_codes;
    bool         apply_watermark;
};

VIENEU_API const char * vieneu_version(void);
VIENEU_API const char * vieneu_last_error(void);

VIENEU_API void vieneu_audio_free(struct vieneu_audio * a);

VIENEU_API void vieneu_init_default_params(struct vieneu_init_params * p);
VIENEU_API struct vieneu_context * vieneu_init(const struct vieneu_init_params * params);
VIENEU_API void vieneu_free(struct vieneu_context * vieneu);
VIENEU_API void vieneu_set_progress_callback(struct vieneu_context * vieneu, vieneu_progress_callback callback, void * user_data);

VIENEU_API void vieneu_tts_default_params(struct vieneu_tts_params * p);
VIENEU_API int vieneu_synthesize(struct vieneu_context * vieneu, const struct vieneu_tts_params * params, struct vieneu_audio * out);

VIENEU_API void vieneu_init_v2_default_params(struct vieneu_init_params_v2 * p);
VIENEU_API struct vieneu_context * vieneu_init_v2(const struct vieneu_init_params_v2 * params);
VIENEU_API void vieneu_tts_v2_default_params(struct vieneu_tts_params_v2 * p);
VIENEU_API int vieneu_synthesize_v2(struct vieneu_context * vieneu, const struct vieneu_tts_params_v2 * params, struct vieneu_audio * out);
VIENEU_API void vieneu_tts_v3_default_params(struct vieneu_tts_params_v3 * p);
VIENEU_API int vieneu_synthesize_v3(struct vieneu_context * vieneu, const struct vieneu_tts_params_v3 * params, struct vieneu_audio * out);

VIENEU_API int vieneu_encode_reference(struct vieneu_context * vieneu, const char * ref_audio_path, float * out_embedding_128);
VIENEU_API int vieneu_list_preset_voices(struct vieneu_context * vieneu, char * out_json, int max_len);
VIENEU_API int vieneu_set_preset_voice(struct vieneu_context * vieneu, const char * voice_id);

#ifdef __cplusplus
}
#endif

#endif // VIENEU_TTS_H
