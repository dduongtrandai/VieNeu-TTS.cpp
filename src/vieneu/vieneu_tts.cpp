#include "vieneu_tts.h"
#include "backends/llama_backend.h"
#include "codecs/neucodec_onnx.h"
#include "vieneu.h"
#include "vieneu_v3_onnx.h"
#include "v3_native/vieneu_v3_native.h"
#include "v3_native/vieneu_v3_native.h"
#include "v3_native/vieneu_v3_native.h"
#include <string>
#include <vector>
#include <memory>
#include <sstream>
#include <regex>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <cctype>
#include <cstring>
#include <cstdlib>
#include <cmath>

// Thread-local error reporting
thread_local std::string g_last_error = "";

static void set_last_error(const std::string& err) {
    g_last_error = err;
}

static VieneuProgressFn make_progress_bridge(vieneu_progress_callback callback, void * user_data) {
    if (!callback) {
        return VieneuProgressFn();
    }
    return [callback, user_data](const VieneuProgressEvent& event) {
        vieneu_progress c_event;
        c_event.abi_version = 1;
        c_event.stage = event.stage ? event.stage : "";
        c_event.current = event.current;
        c_event.total = event.total;
        c_event.progress = event.progress;
        c_event.message = event.message.c_str();
        callback(&c_event, user_data);
    };
}

static std::string escape_regex(const std::string& text) {
    std::string escaped;
    escaped.reserve(text.size() * 2);
    for (char ch : text) {
        switch (ch) {
            case '\\': case '.': case '^': case '$': case '|': case '?':
            case '*': case '+': case '(': case ')': case '[': case ']':
            case '{': case '}':
                escaped.push_back('\\');
                break;
            default:
                break;
        }
        escaped.push_back(ch);
    }
    return escaped;
}

static bool parse_json_string_field(const std::string& json, const std::string& key, std::string& out_value) {
    const std::regex re("\"" + escape_regex(key) + "\"\\s*:\\s*\"([^\"]*)\"");
    std::smatch m;
    if (!std::regex_search(json, m, re)) {
        return false;
    }
    out_value = m[1].str();
    return true;
}

static bool parse_numeric_array_after_key(const std::string& json, size_t object_pos, std::vector<float>& out_values) {
    const size_t codes_pos = json.find("\"codes\"", object_pos);
    const size_t embedding_pos = json.find("\"embedding\"", object_pos);
    size_t field_pos = std::string::npos;
    if (codes_pos != std::string::npos && embedding_pos != std::string::npos) {
        field_pos = std::min(codes_pos, embedding_pos);
    } else {
        field_pos = codes_pos != std::string::npos ? codes_pos : embedding_pos;
    }
    if (field_pos == std::string::npos) {
        return false;
    }

    const size_t arr_start = json.find('[', field_pos);
    const size_t arr_end = json.find(']', arr_start);
    if (arr_start == std::string::npos || arr_end == std::string::npos) {
        return false;
    }

    out_values.clear();
    std::stringstream ss(json.substr(arr_start + 1, arr_end - arr_start - 1));
    std::string token;
    while (std::getline(ss, token, ',')) {
        try {
            out_values.push_back(std::stof(token));
        } catch (...) {
            return false;
        }
    }
    return !out_values.empty();
}

static bool parse_voice_embedding_from_json(const std::string& voices_json, const std::string& voice_id, std::vector<float>& out_embedding) {
    out_embedding.clear();

    const std::string quoted_id = "\"" + voice_id + "\"";
    const size_t object_pos = voices_json.find(quoted_id);
    if (object_pos == std::string::npos) {
        return false;
    }

    if (!parse_numeric_array_after_key(voices_json, object_pos, out_embedding)) {
        return false;
    }

    return out_embedding.size() == 128;
}

static bool set_first_available_voice(struct vieneu_context * vieneu);

enum class VieneuProfileType {
    VIENEU_V2_TURBO,
    VIENEU_V3_ONNX,
    VIENEU_V3_NATIVE
};

struct vieneu_context {
    VieneuProfileType profile = VieneuProfileType::VIENEU_V2_TURBO;
    std::string current_voice_id = "";
    std::vector<float> current_voice_embedding;

    std::unique_ptr<LlamaBackend> llama;
    std::unique_ptr<NeuCodecOnnx> codec_decoder;
    std::unique_ptr<NeuCodecOnnx> codec_encoder;
    std::unique_ptr<VieneuV3OnnxEngine> vieneu_v3;
    std::unique_ptr<VieneuV3NativeEngine> vieneu_v3_native;

    std::string voices_json = "";
    vieneu_progress_callback progress_callback = nullptr;
    void * progress_user_data = nullptr;
};

static void normalize_output_level(std::vector<float>& audio) {
    float peak = 0.0f;
    for (float sample : audio) {
        if (std::isfinite(sample)) {
            peak = std::max(peak, std::abs(sample));
        }
    }

    if (peak < 1.0e-5f || peak >= 0.80f) {
        return;
    }

    const float gain = std::min(0.80f / peak, 12.0f);
    for (float& sample : audio) {
        if (!std::isfinite(sample)) {
            sample = 0.0f;
        } else {
            sample = std::clamp(sample * gain, -1.0f, 1.0f);
        }
    }
}

VIENEU_API const char * vieneu_version(void) {
    return "0.1.0";
}

VIENEU_API const char * vieneu_last_error(void) {
    return g_last_error.c_str();
}

VIENEU_API void vieneu_audio_free(struct vieneu_audio * a) {
    if (a) {
        if (a->samples) {
            free(a->samples);
            a->samples = nullptr;
        }
        a->n_samples = 0;
    }
}

VIENEU_API void vieneu_init_default_params(struct vieneu_init_params * p) {
    if (p) {
        p->abi_version = 1;
        p->model_path = nullptr;
        p->encoder_path = nullptr;
        p->decoder_path = nullptr;
        p->voices_json_path = nullptr;
        p->n_ctx = 2048;
        p->n_threads = 4;
        p->n_gpu_layers = 0;
        p->flash_attn = false;
        p->mlock = true;
    }
}

VIENEU_API void vieneu_init_v2_default_params(struct vieneu_init_params_v2 * p) {
    if (p) {
        p->abi_version = 2;
        p->profile = nullptr;
        p->model_dir = nullptr;
        p->onnx_dir = nullptr;
        p->codec_dir = nullptr;
        p->config_path = nullptr;
        p->tokenizer_path = nullptr;
        p->voices_json_path = nullptr;
        p->n_threads = 2;
    }
}

VIENEU_API struct vieneu_context * vieneu_init(const struct vieneu_init_params * params) {
    if (!params || !params->model_path || !params->decoder_path) {
        set_last_error("Invalid initialization parameters: model_path and decoder_path are required.");
        return nullptr;
    }

    auto ctx = std::make_unique<vieneu_context>();
    ctx->profile = VieneuProfileType::VIENEU_V2_TURBO;

    // Initialize llama backend
    ctx->llama = std::make_unique<LlamaBackend>();
    LlamaBackendParams llama_params;
    llama_params.model_path = params->model_path;
    llama_params.n_ctx = params->n_ctx > 0 ? params->n_ctx : 2048;
    llama_params.n_threads = params->n_threads > 0 ? params->n_threads : 4;
    llama_params.n_gpu_layers = params->n_gpu_layers;
    llama_params.flash_attn = params->flash_attn;
    llama_params.mlock = params->mlock;

    if (!ctx->llama->initialize(llama_params)) {
        set_last_error("Failed to initialize llama.cpp model context.");
        return nullptr;
    }

    // Initialize ONNX decoder
    ctx->codec_decoder = std::make_unique<NeuCodecOnnx>();
    if (!ctx->codec_decoder->initialize(params->decoder_path, llama_params.n_threads)) {
        set_last_error("Failed to initialize ONNX decoder.");
        return nullptr;
    }

    // Initialize optional ONNX encoder
    if (params->encoder_path && strlen(params->encoder_path) > 0) {
        ctx->codec_encoder = std::make_unique<NeuCodecOnnx>();
        if (!ctx->codec_encoder->initialize(params->encoder_path, llama_params.n_threads)) {
            set_last_error("Failed to initialize ONNX encoder.");
            return nullptr;
        }
    }

    // Load preset voices if voices_json_path is provided
    if (params->voices_json_path && strlen(params->voices_json_path) > 0) {
        std::ifstream fs(params->voices_json_path);
        if (fs.is_open()) {
            std::stringstream buffer;
            buffer << fs.rdbuf();
            ctx->voices_json = buffer.str();

            std::string default_voice;
            if (parse_json_string_field(ctx->voices_json, "default_voice", default_voice)) {
                std::vector<float> emb;
                if (parse_voice_embedding_from_json(ctx->voices_json, default_voice, emb)) {
                    ctx->current_voice_id = default_voice;
                    ctx->current_voice_embedding = std::move(emb);
                }
            }
            if (ctx->current_voice_embedding.empty()) {
                set_first_available_voice(ctx.get());
            }
        }
    }

    return ctx.release();
}

VIENEU_API struct vieneu_context * vieneu_init_v2(const struct vieneu_init_params_v2 * params) {
    if (!params) {
        set_last_error("Invalid initialization parameters: params is required.");
        return nullptr;
    }

    const std::string profile = params->profile ? params->profile : "";
    if (profile != "vieneu-v3-onnx" && profile != "vieneu-v3-native") {
        set_last_error("Unsupported ABI v2 profile: " + profile);
        return nullptr;
    }

    auto ctx = std::make_unique<vieneu_context>();
    if (profile == "vieneu-v3-onnx") {
        ctx->profile = VieneuProfileType::VIENEU_V3_ONNX;
        ctx->vieneu_v3 = std::make_unique<VieneuV3OnnxEngine>();

        VieneuV3OnnxInit init;
        init.model_dir = params->model_dir ? params->model_dir : "";
        init.onnx_dir = params->onnx_dir ? params->onnx_dir : "";
        init.codec_dir = params->codec_dir ? params->codec_dir : "";
        init.config_path = params->config_path ? params->config_path : "";
        init.tokenizer_path = params->tokenizer_path ? params->tokenizer_path : "";
        init.voices_json_path = params->voices_json_path ? params->voices_json_path : "";
        init.n_threads = params->n_threads > 0 ? params->n_threads : 0;

        std::string error;
        if (!ctx->vieneu_v3->initialize(init, error)) {
            set_last_error(error);
            return nullptr;
        }
        ctx->voices_json = ctx->vieneu_v3->voices_json();
    } else {
        ctx->profile = VieneuProfileType::VIENEU_V3_NATIVE;
        ctx->vieneu_v3_native = std::make_unique<VieneuV3NativeEngine>();

        VieneuV3NativeInit init;
        init.model_dir = params->model_dir ? params->model_dir : "";
        init.onnx_dir = params->onnx_dir ? params->onnx_dir : "";
        init.codec_dir = params->codec_dir ? params->codec_dir : "";
        init.config_path = params->config_path ? params->config_path : "";
        init.tokenizer_path = params->tokenizer_path ? params->tokenizer_path : "";
        init.voices_json_path = params->voices_json_path ? params->voices_json_path : "";
        init.n_threads = params->n_threads > 0 ? params->n_threads : 4;

        std::string error;
        if (!ctx->vieneu_v3_native->initialize(init, error)) {
            set_last_error(error);
            return nullptr;
        }
        ctx->voices_json = ctx->vieneu_v3_native->voices_json();
    }
    return ctx.release();
}

VIENEU_API void vieneu_free(struct vieneu_context * vieneu) {
    if (vieneu) {
        delete vieneu;
    }
}

VIENEU_API void vieneu_set_progress_callback(struct vieneu_context * vieneu, vieneu_progress_callback callback, void * user_data) {
    if (vieneu) {
        vieneu->progress_callback = callback;
        vieneu->progress_user_data = user_data;
    }
}

VIENEU_API void vieneu_tts_default_params(struct vieneu_tts_params * p) {
    if (p) {
        p->abi_version = 1;
        p->text = nullptr;
        p->voice_id = nullptr;
        p->voice_embedding = nullptr;
        p->temperature = 0.4f;
        p->top_k = 50;
        p->max_chars = 256;
        p->max_tokens = 2048;
        p->skip_normalize = false;
        p->skip_phonemize = false;
        p->apply_watermark = true;
    }
}

VIENEU_API void vieneu_tts_v2_default_params(struct vieneu_tts_params_v2 * p) {
    if (p) {
        p->abi_version = 2;
        p->text = nullptr;
        p->voice_id = nullptr;
        p->ref_audio_path = nullptr;
        p->temperature = 0.8f;
        p->top_k = 25;
        p->top_p = 0.95f;
        p->max_new_frames = 300;
        p->repetition_penalty = 1.2f;
        p->max_chars = 384;
        p->apply_watermark = true;
    }
}

VIENEU_API void vieneu_tts_v3_default_params(struct vieneu_tts_params_v3 * p) {
    if (p) {
        p->abi_version = 3;
        p->text = nullptr;
        p->voice_id = nullptr;
        p->ref_audio_path = nullptr;
        p->style = "tu_nhien";
        p->temperature = 0.8f;
        p->top_k = 25;
        p->top_p = 0.95f;
        p->max_new_frames = 300;
        p->repetition_penalty = 1.2f;
        p->max_chars = 384;
        p->denoise_ref = true;
        p->use_ref_codes = true;
        p->apply_watermark = true;
    }
}

static int vieneu_synthesize_impl(struct vieneu_context * vieneu, const struct vieneu_tts_params * params, struct vieneu_audio * out) {
    if (!vieneu || !params || !params->text || !out) {
        set_last_error("Invalid synthesize arguments: context, params, text, and out are required.");
        return -1;
    }

    std::vector<float> voice_emb(128, 0.0f);

    if (params->voice_id && params->voice_id[0] != '\0') {
        if (vieneu_set_preset_voice(vieneu, params->voice_id) != 0) {
            return -1;
        }
    }

    if (params->voice_embedding) {
        std::copy(params->voice_embedding, params->voice_embedding + 128, voice_emb.begin());
    } else if (!vieneu->current_voice_embedding.empty()) {
        voice_emb = vieneu->current_voice_embedding;
    }

    std::vector<float> out_audio;
    bool success = false;
    const VieneuProgressFn progress = make_progress_bridge(vieneu->progress_callback, vieneu->progress_user_data);

    if (vieneu->profile == VieneuProfileType::VIENEU_V2_TURBO) {
        success = VieneuProfile::synthesize(
            *vieneu->llama,
            *vieneu->codec_decoder,
            params->text,
            voice_emb,
            params->temperature,
            params->top_k,
            params->max_tokens,
            params->skip_phonemize,
            out_audio,
            progress
        );
    }

    if (!success || out_audio.empty()) {
        set_last_error("Synthesis pipeline failed or generated empty audio.");
        return -1;
    }

    normalize_output_level(out_audio);

    // Allocate samples on the heap (using malloc to match standard C free)
    out->samples = (float*)malloc(out_audio.size() * sizeof(float));
    if (!out->samples) {
        set_last_error("Memory allocation failed for audio output buffer.");
        return -1;
    }

    std::copy(out_audio.begin(), out_audio.end(), out->samples);
    out->n_samples = (int)out_audio.size();
    out->sample_rate = 24000;
    out->channels = 1;

    return 0;
}

static int vieneu_synthesize_v2_impl(struct vieneu_context * vieneu, const struct vieneu_tts_params_v2 * params, struct vieneu_audio * out) {
    if (!vieneu || !params || !params->text || !out) {
        set_last_error("Invalid synthesize_v2 arguments: context, params, text, and out are required.");
        return -1;
    }
    if (vieneu->profile != VieneuProfileType::VIENEU_V3_ONNX && vieneu->profile != VieneuProfileType::VIENEU_V3_NATIVE) {
        set_last_error("vieneu_synthesize_v2 is only supported for the vieneu-v3-onnx or vieneu-v3-native profile.");
        return -1;
    }

    std::vector<float> out_audio;
    std::string error;
    int sr = 48000;

    if (vieneu->profile == VieneuProfileType::VIENEU_V3_ONNX) {
        if (!vieneu->vieneu_v3) {
            set_last_error("ONNX engine not initialized.");
            return -1;
        }
        VieneuV3OnnxParams v3_params;
        v3_params.text = params->text ? params->text : "";
        v3_params.voice_id = params->voice_id ? params->voice_id : "";
        v3_params.ref_audio_path = params->ref_audio_path ? params->ref_audio_path : "";
        v3_params.temperature = params->temperature;
        v3_params.top_k = params->top_k;
        v3_params.top_p = params->top_p;
        v3_params.max_new_frames = params->max_new_frames;
        v3_params.repetition_penalty = params->repetition_penalty;
        v3_params.max_chars = params->max_chars;
        v3_params.apply_watermark = params->apply_watermark;
        v3_params.progress = make_progress_bridge(vieneu->progress_callback, vieneu->progress_user_data);

        if (!vieneu->vieneu_v3->synthesize(v3_params, out_audio, error)) {
            set_last_error(error);
            return -1;
        }
        sr = vieneu->vieneu_v3->sample_rate();
    } else {
        if (!vieneu->vieneu_v3_native) {
            set_last_error("Native engine not initialized.");
            return -1;
        }
        VieneuV3NativeParams v3_params;
        v3_params.text = params->text ? params->text : "";
        v3_params.voice_id = params->voice_id ? params->voice_id : "";
        v3_params.ref_audio_path = params->ref_audio_path ? params->ref_audio_path : "";
        v3_params.temperature = params->temperature;
        v3_params.top_k = params->top_k;
        v3_params.top_p = params->top_p;
        v3_params.max_new_frames = params->max_new_frames;
        v3_params.repetition_penalty = params->repetition_penalty;
        v3_params.max_chars = params->max_chars;
        v3_params.apply_watermark = params->apply_watermark;
        v3_params.progress = make_progress_bridge(vieneu->progress_callback, vieneu->progress_user_data);

        if (!vieneu->vieneu_v3_native->synthesize(v3_params, out_audio, error)) {
            set_last_error(error);
            return -1;
        }
        sr = vieneu->vieneu_v3_native->sample_rate();
    }

    if (out_audio.empty()) {
        set_last_error("VieNeu v3 synthesis produced empty audio.");
        return -1;
    }

    normalize_output_level(out_audio);
    out->samples = (float*)malloc(out_audio.size() * sizeof(float));
    if (!out->samples) {
        set_last_error("Memory allocation failed for audio output buffer.");
        return -1;
    }
    std::copy(out_audio.begin(), out_audio.end(), out->samples);
    out->n_samples = (int)out_audio.size();
    out->sample_rate = sr;
    out->channels = 1;
    return 0;
}


static int vieneu_synthesize_v3_impl(struct vieneu_context * vieneu, const struct vieneu_tts_params_v3 * params, struct vieneu_audio * out) {
    if (!vieneu || !params || !params->text || !out) {
        set_last_error("Invalid synthesize_v3 arguments: context, params, text, and out are required.");
        return -1;
    }
    if (vieneu->profile != VieneuProfileType::VIENEU_V3_NATIVE) {
        set_last_error("vieneu_synthesize_v3 is only supported for the vieneu-v3-native profile.");
        return -1;
    }
    if (!vieneu->vieneu_v3_native) {
        set_last_error("Native engine not initialized.");
        return -1;
    }

    VieneuV3NativeParams v3_params;
    v3_params.text = params->text ? params->text : "";
    v3_params.voice_id = params->voice_id ? params->voice_id : "";
    v3_params.ref_audio_path = params->ref_audio_path ? params->ref_audio_path : "";
    v3_params.style = params->style ? params->style : "tu_nhien";
    v3_params.temperature = params->temperature;
    v3_params.top_k = params->top_k;
    v3_params.top_p = params->top_p;
    v3_params.max_new_frames = params->max_new_frames;
    v3_params.repetition_penalty = params->repetition_penalty;
    v3_params.max_chars = params->max_chars;
    v3_params.denoise_ref = params->denoise_ref;
    v3_params.use_ref_codes = params->use_ref_codes;
    v3_params.apply_watermark = params->apply_watermark;
    v3_params.progress = make_progress_bridge(vieneu->progress_callback, vieneu->progress_user_data);

    std::vector<float> out_audio;
    std::string error;
    if (!vieneu->vieneu_v3_native->synthesize(v3_params, out_audio, error)) {
        set_last_error(error);
        return -1;
    }
    if (out_audio.empty()) {
        set_last_error("VieNeu v3 native synthesis produced empty audio.");
        return -1;
    }
    normalize_output_level(out_audio);
    out->samples = (float*)malloc(out_audio.size() * sizeof(float));
    if (!out->samples) {
        set_last_error("Memory allocation failed for audio output buffer.");
        return -1;
    }
    std::copy(out_audio.begin(), out_audio.end(), out->samples);
    out->n_samples = (int)out_audio.size();
    out->sample_rate = vieneu->vieneu_v3_native->sample_rate();
    out->channels = 1;
    return 0;
}

static void clear_audio_output(struct vieneu_audio * out) {
    if (!out) {
        return;
    }
    if (out->samples) {
        free(out->samples);
    }
    out->samples = nullptr;
    out->n_samples = 0;
    out->sample_rate = 0;
    out->channels = 0;
}

static int vieneu_synthesize_cpp_guard(struct vieneu_context * vieneu, const struct vieneu_tts_params * params, struct vieneu_audio * out) {
    try {
        return vieneu_synthesize_impl(vieneu, params, out);
    } catch (const std::exception& e) {
        clear_audio_output(out);
        set_last_error(std::string("Unhandled C++ exception during synthesis: ") + e.what());
        return -1;
    } catch (...) {
        clear_audio_output(out);
        set_last_error("Unhandled unknown C++ exception during synthesis.");
        return -1;
    }
}

VIENEU_API int vieneu_synthesize(struct vieneu_context * vieneu, const struct vieneu_tts_params * params, struct vieneu_audio * out) {
    return vieneu_synthesize_cpp_guard(vieneu, params, out);
}

VIENEU_API int vieneu_synthesize_v2(struct vieneu_context * vieneu, const struct vieneu_tts_params_v2 * params, struct vieneu_audio * out) {
    try {
        return vieneu_synthesize_v2_impl(vieneu, params, out);
    } catch (const std::exception& e) {
        clear_audio_output(out);
        set_last_error(std::string("Unhandled C++ exception during ABI v2 synthesis: ") + e.what());
        return -1;
    } catch (...) {
        clear_audio_output(out);
        set_last_error("Unhandled unknown C++ exception during ABI v2 synthesis.");
        return -1;
    }
}

VIENEU_API int vieneu_synthesize_v3(struct vieneu_context * vieneu, const struct vieneu_tts_params_v3 * params, struct vieneu_audio * out) {
    try {
        return vieneu_synthesize_v3_impl(vieneu, params, out);
    } catch (const std::exception& e) {
        clear_audio_output(out);
        set_last_error(std::string("Unhandled C++ exception during ABI v3 synthesis: ") + e.what());
        return -1;
    } catch (...) {
        clear_audio_output(out);
        set_last_error("Unhandled unknown C++ exception during ABI v3 synthesis.");
        return -1;
    }
}

VIENEU_API int vieneu_encode_reference(struct vieneu_context * vieneu, const char * ref_audio_path, float * out_embedding_128) {
    if (!vieneu || !ref_audio_path || !out_embedding_128) {
        set_last_error("Invalid encode_reference arguments.");
        return -1;
    }

    if (vieneu->profile == VieneuProfileType::VIENEU_V3_ONNX) {
        set_last_error("vieneu_encode_reference returns a 128-dimensional v2 embedding and is not supported for VieNeu v3. Use vieneu_synthesize_v2 with ref_audio_path.");
        return -1;
    }

    if (!vieneu->codec_encoder) {
        set_last_error("Speaker encoder was not loaded at initialization.");
        return -1;
    }

    std::vector<float> waveform;
    if (!read_wav_file_24k_mono(ref_audio_path, waveform) || waveform.empty()) {
        set_last_error("Failed to read reference WAV file or it was empty.");
        return -1;
    }

    std::vector<float> emb;
    if (!vieneu->codec_encoder->encode_speaker(waveform, emb)) {
        set_last_error("Failed to extract speaker embedding using ONNX encoder.");
        return -1;
    }
    if (emb.size() != 128) {
        set_last_error("Speaker encoder returned invalid embedding size (expected 128).");
        return -1;
    }

    std::copy(emb.begin(), emb.end(), out_embedding_128);
    return 0;
}

VIENEU_API int vieneu_list_preset_voices(struct vieneu_context * vieneu, char * out_json, int max_len) {
    if (!vieneu || !out_json || max_len <= 0) {
        set_last_error("Invalid list_preset_voices arguments.");
        return -1;
    }

    std::string result = "[]";
    if (!vieneu->voices_json.empty()) {
        result = vieneu->voices_json;
    }

    if ((int)result.size() >= max_len) {
        set_last_error("Output buffer size is too small to fit the voices JSON string.");
        return -1;
    }

    strncpy(out_json, result.c_str(), max_len - 1);
    out_json[max_len - 1] = '\0';
    return 0;
}

VIENEU_API int vieneu_set_preset_voice(struct vieneu_context * vieneu, const char * voice_id) {
    if (!vieneu || !voice_id) {
        set_last_error("Invalid set_preset_voice arguments.");
        return -1;
    }

    if (vieneu->voices_json.empty()) {
        set_last_error("No voices.json loaded to look up the voice.");
        return -1;
    }

    std::vector<float> emb;
    if (parse_voice_embedding_from_json(vieneu->voices_json, voice_id, emb)) {
        vieneu->current_voice_id = voice_id;
        vieneu->current_voice_embedding = std::move(emb);
        return 0;
    }

    set_last_error("Failed to find voice embedding/codes[128] for voice ID: " + std::string(voice_id));
    return -1;
}

static bool set_first_available_voice(struct vieneu_context * vieneu) {
    if (!vieneu || vieneu->voices_json.empty()) {
        return false;
    }

    const std::regex preset_re("\"([^\"]+)\"\\s*:\\s*\\{[^\\}]*\"(?:embedding|codes)\"\\s*:");
    auto begin = std::sregex_iterator(vieneu->voices_json.begin(), vieneu->voices_json.end(), preset_re);
    auto end = std::sregex_iterator();
    for (auto it = begin; it != end; ++it) {
        std::string id = (*it)[1].str();
        if (id == "meta" || id == "presets") {
            continue;
        }
        std::vector<float> emb;
        if (parse_voice_embedding_from_json(vieneu->voices_json, id, emb)) {
            vieneu->current_voice_id = id;
            vieneu->current_voice_embedding = std::move(emb);
            return true;
        }
    }
    return false;
}
