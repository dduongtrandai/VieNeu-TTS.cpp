#include "vieneu/vieneu_tts.h"
#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <cstring>
#include <algorithm>
#include <chrono>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <shellapi.h>
#endif

static void print_usage(const std::string& argv0) {
    std::cout << "Usage: " << argv0 << " [options]\n\n"
              << "Options:\n"
              << "  --profile NAME         Runtime profile: vieneu-v2-turbo or vieneu-v3-onnx (default: vieneu-v2-turbo)\n"
              << "  -m, --model PATH       Path to GGUF model (required for v2)\n"
              << "  -d, --decoder PATH     Path to ONNX decoder (required for v2)\n"
              << "  -e, --encoder PATH     Path to ONNX encoder (optional for v2)\n"
              << "  --model-dir PATH       VieNeu v3 model directory containing config/tokenizer\n"
              << "  --onnx-dir PATH        VieNeu v3 ONNX directory\n"
              << "  --codec-dir PATH       MOSS ONNX codec directory\n"
              << "  --config PATH          VieNeu v3 config.json path (optional)\n"
              << "  --tokenizer PATH       VieNeu v3 tokenizer.json path (optional)\n"
              << "  --ref-audio PATH       VieNeu v3 reference WAV for voice cloning\n"
              << "  -t, --text TEXT        Text to synthesize (default: 'Xin chào các bạn.')\n"
              << "  -v, --voice ID         Preset voice ID (default: none)\n"
              << "  -o, --output PATH      Path to output WAV file (default: 'output.wav')\n"
              << "  --voices-json PATH     Path to voices.json file (optional)\n"
              << "  --temperature VALUE    Sampling temperature\n"
              << "  --top-k VALUE          Top-k sampling value\n"
              << "  --top-p VALUE          VieNeu v3 top-p sampling value\n"
              << "  --max-new-frames N     VieNeu v3 frame limit\n"
              << "  --threads N            ONNX/CPU thread count (0 = ONNX Runtime auto; omitted = runtime default)\n"
              << "  -h, --help             Show this help message\n";
}

#ifdef _WIN32
static std::string wide_to_utf8(const wchar_t* value) {
    if (!value) {
        return {};
    }
    const int required = WideCharToMultiByte(CP_UTF8, 0, value, -1, nullptr, 0, nullptr, nullptr);
    if (required <= 1) {
        return {};
    }
    std::string out(static_cast<size_t>(required - 1), '\0');
    WideCharToMultiByte(CP_UTF8, 0, value, -1, out.data(), required, nullptr, nullptr);
    return out;
}
#endif

static std::vector<std::string> get_utf8_args(int argc, char* argv[]) {
#ifdef _WIN32
    int wide_argc = 0;
    LPWSTR* wide_argv = CommandLineToArgvW(GetCommandLineW(), &wide_argc);
    if (wide_argv) {
        std::vector<std::string> args;
        args.reserve(static_cast<size_t>(wide_argc));
        for (int i = 0; i < wide_argc; ++i) {
            args.push_back(wide_to_utf8(wide_argv[i]));
        }
        LocalFree(wide_argv);
        return args;
    }
#endif

    std::vector<std::string> args;
    args.reserve(static_cast<size_t>(argc));
    for (int i = 0; i < argc; ++i) {
        args.emplace_back(argv[i] ? argv[i] : "");
    }
    return args;
}

static bool write_wav_file(const std::string& path, const float* samples, int n_samples, int sample_rate) {
    std::ofstream fs(path, std::ios::binary);
    if (!fs.is_open()) return false;

    struct WavHeader {
        char riff[4] = {'R', 'I', 'F', 'F'};
        int32_t overall_size;
        char wave[4] = {'W', 'A', 'V', 'E'};
        char fmt_chunk_marker[4] = {'f', 'm', 't', ' '};
        int32_t length_of_fmt = 16;
        int16_t format_type = 1;
        int16_t channels = 1;
        int32_t sample_rate;
        int32_t byterate;
        int16_t block_align;
        int16_t bits_per_sample = 16;
        char data_chunk_header[4] = {'d', 'a', 't', 'a'};
        int32_t data_size;
    } header;

    int32_t data_size = n_samples * sizeof(int16_t);
    header.overall_size = 36 + data_size;
    header.sample_rate = sample_rate;
    header.byterate = sample_rate * sizeof(int16_t);
    header.block_align = sizeof(int16_t);
    header.data_size = data_size;

    fs.write((const char*)&header, sizeof(header));

    for (int i = 0; i < n_samples; ++i) {
        float sample = samples[i];
        if (sample < -1.0f) sample = -1.0f;
        if (sample > 1.0f) sample = 1.0f;
        int16_t pcm = (int16_t)(sample * 32767.0f);
        fs.write((const char*)&pcm, sizeof(pcm));
    }

    return true;
}

int main(int argc, char* argv[]) {
    const std::vector<std::string> args = get_utf8_args(argc, argv);
    const std::string argv0 = args.empty() ? "vieneu-tts-cli" : args[0];
    std::string profile = "vieneu-v2-turbo";
    std::string model_path = "";
    std::string decoder_path = "";
    std::string encoder_path = "";
    std::string model_dir = "";
    std::string onnx_dir = "";
    std::string codec_dir = "";
    std::string config_path = "";
    std::string tokenizer_path = "";
    std::string ref_audio_path = "";
    std::string text = "Xin chào các bạn.";
    std::string voice_id = "";
    std::string output_path = "output.wav";
    std::string voices_json_path = "";
    float temperature = 0.0f;
    int top_k = 0;
    float top_p = 0.0f;
    int max_new_frames = 0;
    int n_threads = -1;

    for (size_t i = 1; i < args.size(); ++i) {
        std::string arg = args[i];
        if (arg == "--profile") {
            if (i + 1 < args.size()) profile = args[++i];
        } else if (arg == "-m" || arg == "--model") {
            if (i + 1 < args.size()) model_path = args[++i];
        } else if (arg == "-d" || arg == "--decoder") {
            if (i + 1 < args.size()) decoder_path = args[++i];
        } else if (arg == "-e" || arg == "--encoder") {
            if (i + 1 < args.size()) encoder_path = args[++i];
        } else if (arg == "--model-dir") {
            if (i + 1 < args.size()) model_dir = args[++i];
        } else if (arg == "--onnx-dir") {
            if (i + 1 < args.size()) onnx_dir = args[++i];
        } else if (arg == "--codec-dir") {
            if (i + 1 < args.size()) codec_dir = args[++i];
        } else if (arg == "--config") {
            if (i + 1 < args.size()) config_path = args[++i];
        } else if (arg == "--tokenizer") {
            if (i + 1 < args.size()) tokenizer_path = args[++i];
        } else if (arg == "--ref-audio") {
            if (i + 1 < args.size()) ref_audio_path = args[++i];
        } else if (arg == "-t" || arg == "--text") {
            if (i + 1 < args.size()) text = args[++i];
        } else if (arg == "-v" || arg == "--voice") {
            if (i + 1 < args.size()) voice_id = args[++i];
        } else if (arg == "-o" || arg == "--output") {
            if (i + 1 < args.size()) output_path = args[++i];
        } else if (arg == "--voices-json") {
            if (i + 1 < args.size()) voices_json_path = args[++i];
        } else if (arg == "--temperature") {
            if (i + 1 < args.size()) temperature = std::stof(args[++i]);
        } else if (arg == "--top-k") {
            if (i + 1 < args.size()) top_k = std::stoi(args[++i]);
        } else if (arg == "--top-p") {
            if (i + 1 < args.size()) top_p = std::stof(args[++i]);
        } else if (arg == "--max-new-frames") {
            if (i + 1 < args.size()) max_new_frames = std::stoi(args[++i]);
        } else if (arg == "--threads") {
            if (i + 1 < args.size()) n_threads = std::stoi(args[++i]);
        } else if (arg == "-h" || arg == "--help") {
            print_usage(argv0);
            return 0;
        } else {
            std::cerr << "Unknown option: " << arg << "\n";
            print_usage(argv0);
            return 1;
        }
    }

    if (profile == "vieneu-v3-onnx") {
        if (model_dir.empty() && onnx_dir.empty()) {
            std::cerr << "Error: --model-dir or --onnx-dir is required for vieneu-v3-onnx.\n";
            print_usage(argv0);
            return 1;
        }
        if (codec_dir.empty()) {
            std::cerr << "Error: --codec-dir is required for vieneu-v3-onnx.\n";
            print_usage(argv0);
            return 1;
        }

        std::cout << "[CLI] Initializing VieNeu v3 ONNX context...\n"
                  << "  Model dir: " << model_dir << "\n"
                  << "  ONNX dir: " << onnx_dir << "\n"
                  << "  Codec dir: " << codec_dir << "\n";

        vieneu_init_params_v2 params;
        vieneu_init_v2_default_params(&params);
        params.profile = "vieneu-v3-onnx";
        params.model_dir = model_dir.empty() ? nullptr : model_dir.c_str();
        params.onnx_dir = onnx_dir.empty() ? nullptr : onnx_dir.c_str();
        params.codec_dir = codec_dir.c_str();
        params.config_path = config_path.empty() ? nullptr : config_path.c_str();
        params.tokenizer_path = tokenizer_path.empty() ? nullptr : tokenizer_path.c_str();
        params.voices_json_path = voices_json_path.empty() ? nullptr : voices_json_path.c_str();
        if (n_threads >= 0) {
            params.n_threads = n_threads;
        }

        vieneu_context* ctx = vieneu_init_v2(&params);
        if (!ctx) {
            std::cerr << "[CLI] Initialization failed: " << vieneu_last_error() << "\n";
            return 1;
        }

        vieneu_tts_params_v2 tts_params;
        vieneu_tts_v2_default_params(&tts_params);
        tts_params.text = text.c_str();
        tts_params.voice_id = voice_id.empty() ? nullptr : voice_id.c_str();
        tts_params.ref_audio_path = ref_audio_path.empty() ? nullptr : ref_audio_path.c_str();
        if (temperature > 0.0f) tts_params.temperature = temperature;
        if (top_k > 0) tts_params.top_k = top_k;
        if (top_p > 0.0f) tts_params.top_p = top_p;
        if (max_new_frames > 0) tts_params.max_new_frames = max_new_frames;

        vieneu_audio audio;
        memset(&audio, 0, sizeof(audio));
        std::cout << "[CLI] Synthesizing with VieNeu v3: \"" << text << "\"\n";
        
        auto start_time = std::chrono::high_resolution_clock::now();
        int synth_status = vieneu_synthesize_v2(ctx, &tts_params, &audio);
        auto end_time = std::chrono::high_resolution_clock::now();
        
        if (synth_status != 0) {
            std::cerr << "[CLI] Synthesis failed: " << vieneu_last_error() << "\n";
            vieneu_free(ctx);
            return 1;
        }

        double elapsed_sec = std::chrono::duration<double>(end_time - start_time).count();
        double audio_dur_sec = (double)audio.n_samples / audio.sample_rate;
        double rtf = elapsed_sec / audio_dur_sec;

        std::cout << "[CLI] Synthesis successful. n_samples=" << audio.n_samples
                  << ", sample_rate=" << audio.sample_rate << "Hz\n";
        std::cout << "[CLI] Inference time (model execution): " << elapsed_sec << "s (" << (elapsed_sec * 1000.0) << " ms)\n";
        std::cout << "[CLI] Audio duration: " << audio_dur_sec << "s\n";
        std::cout << "[CLI] Real-Time Factor (RTF): " << rtf << " (" << (1.0 / rtf) << "x real-time speed)\n";
        if (!write_wav_file(output_path, audio.samples, audio.n_samples, audio.sample_rate)) {
            std::cerr << "[CLI] Failed to write WAV file.\n";
            vieneu_audio_free(&audio);
            vieneu_free(ctx);
            return 1;
        }

        std::cout << "[CLI] Done.\n";
        vieneu_audio_free(&audio);
        vieneu_free(ctx);
        return 0;
    }

    if (profile != "vieneu-v2-turbo") {
        std::cerr << "Error: unsupported profile: " << profile << "\n";
        print_usage(argv0);
        return 1;
    }

    if (model_path.empty()) {
        std::cerr << "Error: --model option is required.\n";
        print_usage(argv0);
        return 1;
    }
    if (decoder_path.empty()) {
        std::cerr << "Error: --decoder option is required.\n";
        print_usage(argv0);
        return 1;
    }

    std::cout << "[CLI] Initializing VieNeu TTS context...\n"
              << "  Model: " << model_path << "\n"
              << "  Decoder: " << decoder_path << "\n";
    if (!encoder_path.empty()) {
        std::cout << "  Encoder: " << encoder_path << "\n";
    }

    vieneu_init_params params;
    vieneu_init_default_params(&params);
    params.model_path = model_path.c_str();
    params.decoder_path = decoder_path.c_str();
    if (!encoder_path.empty()) {
        params.encoder_path = encoder_path.c_str();
    }
    if (!voices_json_path.empty()) {
        params.voices_json_path = voices_json_path.c_str();
    }
    if (n_threads > 0) {
        params.n_threads = n_threads;
    }

    vieneu_context* ctx = vieneu_init(&params);
    if (!ctx) {
        std::cerr << "[CLI] Initialization failed: " << vieneu_last_error() << "\n";
        return 1;
    }
    std::cout << "[CLI] Initialization successful.\n";

    if (!voice_id.empty() && !voices_json_path.empty()) {
        std::cout << "[CLI] Setting preset voice ID: " << voice_id << "\n";
        if (vieneu_set_preset_voice(ctx, voice_id.c_str()) != 0) {
            std::cerr << "[CLI] Failed to set voice: " << vieneu_last_error() << "\n";
            vieneu_free(ctx);
            return 1;
        }
    }

    std::cout << "[CLI] Synthesizing text: \"" << text << "\"\n";
    vieneu_tts_params tts_params;
    vieneu_tts_default_params(&tts_params);
    tts_params.text = text.c_str();
    if (temperature > 0.0f) tts_params.temperature = temperature;
    if (top_k > 0) tts_params.top_k = top_k;

    vieneu_audio audio;
    memset(&audio, 0, sizeof(audio));

    auto start_time = std::chrono::high_resolution_clock::now();
    int synth_status = vieneu_synthesize(ctx, &tts_params, &audio);
    auto end_time = std::chrono::high_resolution_clock::now();

    if (synth_status != 0) {
        std::cerr << "[CLI] Synthesis failed: " << vieneu_last_error() << "\n";
        vieneu_free(ctx);
        return 1;
    }

    double elapsed_sec = std::chrono::duration<double>(end_time - start_time).count();
    double audio_dur_sec = (double)audio.n_samples / audio.sample_rate;
    double rtf = elapsed_sec / audio_dur_sec;

    std::cout << "[CLI] Synthesis successful. n_samples=" << audio.n_samples 
              << ", sample_rate=" << audio.sample_rate << "Hz\n";
    std::cout << "[CLI] Inference time (model execution): " << elapsed_sec << "s (" << (elapsed_sec * 1000.0) << " ms)\n";
    std::cout << "[CLI] Audio duration: " << audio_dur_sec << "s\n";
    std::cout << "[CLI] Real-Time Factor (RTF): " << rtf << " (" << (1.0 / rtf) << "x real-time speed)\n";

    std::cout << "[CLI] Writing output file: " << output_path << "\n";
    if (!write_wav_file(output_path, audio.samples, audio.n_samples, audio.sample_rate)) {
        std::cerr << "[CLI] Failed to write WAV file.\n";
        vieneu_audio_free(&audio);
        vieneu_free(ctx);
        return 1;
    }

    std::cout << "[CLI] Done.\n";
    vieneu_audio_free(&audio);
    vieneu_free(ctx);
    return 0;
}
