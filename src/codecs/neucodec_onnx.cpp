#include "neucodec_onnx.h"
#include <fstream>
#include <iostream>
#include <algorithm>
#include <cmath>

NeuCodecOnnx::NeuCodecOnnx() {}

NeuCodecOnnx::~NeuCodecOnnx() {
    free_resources();
}

bool NeuCodecOnnx::initialize(const std::string& model_path, int n_threads) {
    try {
        free_resources();
        
        env = std::make_shared<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "NeuCodecOnnx");
        
        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(n_threads);
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

#ifdef _WIN32
        std::wstring w_model_path(model_path.begin(), model_path.end());
        session = std::make_unique<Ort::Session>(*env, w_model_path.c_str(), session_options);
#else
        session = std::make_unique<Ort::Session>(*env, model_path.c_str(), session_options);
#endif

        Ort::AllocatorWithDefaultOptions allocator;
        
        size_t num_inputs = session->GetInputCount();
        for (size_t i = 0; i < num_inputs; ++i) {
            auto input_name_alloc = session->GetInputNameAllocated(i, allocator);
            input_names.push_back(std::string(input_name_alloc.get()));
        }

        size_t num_outputs = session->GetOutputCount();
        for (size_t i = 0; i < num_outputs; ++i) {
            auto output_name_alloc = session->GetOutputNameAllocated(i, allocator);
            output_names.push_back(std::string(output_name_alloc.get()));
        }

        return true;
    } catch (const std::exception& e) {
        std::cerr << "ONNX initialization error: " << e.what() << std::endl;
        return false;
    }
}

void NeuCodecOnnx::free_resources() {
    session.reset();
    env.reset();
    input_names.clear();
    output_names.clear();
}

bool NeuCodecOnnx::has_input(const std::string& name) const {
    return std::find(input_names.begin(), input_names.end(), name) != input_names.end();
}

bool NeuCodecOnnx::decode_vieneu(
    const std::vector<int64_t>& speech_tokens,
    const std::vector<float>& voice_embedding,
    std::vector<float>& out_audio)
{
    if (!session) return false;

    try {
        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

        // Input 1: content_ids
        std::vector<int64_t> dims_tokens = { 1, (int64_t)speech_tokens.size() };
        Ort::Value tensor_tokens = Ort::Value::CreateTensor<int64_t>(
            memory_info,
            const_cast<int64_t*>(speech_tokens.data()),
            speech_tokens.size(),
            dims_tokens.data(),
            dims_tokens.size()
        );

        // Input 2: voice_embedding
        std::vector<int64_t> dims_emb = { 1, 128 };
        Ort::Value tensor_emb = Ort::Value::CreateTensor<float>(
            memory_info,
            const_cast<float*>(voice_embedding.data()),
            voice_embedding.size(),
            dims_emb.data(),
            dims_emb.size()
        );

        // Inputs and Outputs
        std::vector<const char*> in_names = { "content_ids", "voice_embedding" };
        std::vector<Ort::Value> in_tensors;
        in_tensors.push_back(std::move(tensor_tokens));
        in_tensors.push_back(std::move(tensor_emb));

        std::vector<const char*> out_names_ptr;
        for (const auto& name : output_names) {
            out_names_ptr.push_back(name.c_str());
        }

        auto output_values = session->Run(
            Ort::RunOptions{nullptr},
            in_names.data(),
            in_tensors.data(),
            in_tensors.size(),
            out_names_ptr.data(),
            out_names_ptr.size()
        );

        if (output_values.empty()) return false;

        float* audio_data = output_values[0].GetTensorMutableData<float>();
        auto type_info = output_values[0].GetTensorTypeAndShapeInfo();
        size_t total_elements = type_info.GetElementCount();

        out_audio.assign(audio_data, audio_data + total_elements);
        return true;
    } catch (const std::exception& e) {
        std::cerr << "ONNX decode_vieneu error: " << e.what() << std::endl;
        return false;
    }
}

bool NeuCodecOnnx::decode_neucodec(
    const std::vector<int32_t>& codes,
    std::vector<float>& out_audio)
{
    if (!session) return false;

    try {
        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

        // Input: codes [1, 1, F]
        std::vector<int64_t> dims_codes = { 1, 1, (int64_t)codes.size() };
        Ort::Value tensor_codes = Ort::Value::CreateTensor<int32_t>(
            memory_info,
            const_cast<int32_t*>(codes.data()),
            codes.size(),
            dims_codes.data(),
            dims_codes.size()
        );

        std::vector<const char*> in_names = { "codes" };
        std::vector<Ort::Value> in_tensors;
        in_tensors.push_back(std::move(tensor_codes));

        std::vector<const char*> out_names_ptr;
        for (const auto& name : output_names) {
            out_names_ptr.push_back(name.c_str());
        }

        auto output_values = session->Run(
            Ort::RunOptions{nullptr},
            in_names.data(),
            in_tensors.data(),
            in_tensors.size(),
            out_names_ptr.data(),
            out_names_ptr.size()
        );

        if (output_values.empty()) return false;

        float* audio_data = output_values[0].GetTensorMutableData<float>();
        auto type_info = output_values[0].GetTensorTypeAndShapeInfo();
        size_t total_elements = type_info.GetElementCount();

        out_audio.assign(audio_data, audio_data + total_elements);
        return true;
    } catch (const std::exception& e) {
        std::cerr << "ONNX decode_neucodec error: " << e.what() << std::endl;
        return false;
    }
}

bool NeuCodecOnnx::encode_speaker(
    const std::vector<float>& waveform,
    std::vector<float>& out_embedding_128)
{
    if (!session) return false;

    try {
        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

        // Input: waveform [1, T]
        std::vector<int64_t> dims_wave = { 1, (int64_t)waveform.size() };
        Ort::Value tensor_wave = Ort::Value::CreateTensor<float>(
            memory_info,
            const_cast<float*>(waveform.data()),
            waveform.size(),
            dims_wave.data(),
            dims_wave.size()
        );

        std::vector<const char*> in_names = { "waveform" };
        std::vector<Ort::Value> in_tensors;
        in_tensors.push_back(std::move(tensor_wave));

        std::vector<const char*> out_names_ptr;
        for (const auto& name : output_names) {
            out_names_ptr.push_back(name.c_str());
        }

        auto output_values = session->Run(
            Ort::RunOptions{nullptr},
            in_names.data(),
            in_tensors.data(),
            in_tensors.size(),
            out_names_ptr.data(),
            out_names_ptr.size()
        );

        if (output_values.empty()) return false;

        float* emb_data = output_values[0].GetTensorMutableData<float>();
        auto type_info = output_values[0].GetTensorTypeAndShapeInfo();
        size_t total_elements = type_info.GetElementCount();

        if (total_elements != 128) {
            std::cerr << "ONNX encode_speaker output size mismatch: expected 128, got " << total_elements << std::endl;
            return false;
        }

        out_embedding_128.assign(emb_data, emb_data + 128);
        return true;
    } catch (const std::exception& e) {
        std::cerr << "ONNX encode_speaker error: " << e.what() << std::endl;
        return false;
    }
}

bool read_wav_file_24k_mono(const std::string& path, std::vector<float>& out_samples) {
    std::ifstream fs(path, std::ios::binary);
    if (!fs.is_open()) return false;

    struct WavHeader {
        char riff[4];
        int32_t overall_size;
        char wave[4];
        char fmt_chunk_marker[4];
        int32_t length_of_fmt;
        int16_t format_type;
        int16_t channels;
        int32_t sample_rate;
        int32_t byterate;
        int16_t block_align;
        int16_t bits_per_sample;
    } header;

    fs.read((char*)&header, sizeof(header));
    if (fs.gcount() != sizeof(header)) return false;

    if (std::string(header.riff, 4) != "RIFF" || std::string(header.wave, 4) != "WAVE") {
        return false;
    }

    char chunk_id[4];
    int32_t chunk_size = 0;
    while (true) {
        fs.read(chunk_id, 4);
        if (fs.gcount() != 4) return false;
        fs.read((char*)&chunk_size, 4);
        if (fs.gcount() != 4) return false;

        if (std::string(chunk_id, 4) == "data") {
            break;
        }

        fs.seekg(chunk_size, std::ios::cur);
        if (fs.fail()) return false;
    }

    std::vector<char> raw_data(chunk_size);
    fs.read(raw_data.data(), chunk_size);
    if (fs.gcount() != chunk_size) return false;

    std::vector<float> input_samples;
    if (header.bits_per_sample == 16) {
        size_t n_samples = chunk_size / sizeof(int16_t);
        const int16_t* p = (const int16_t*)raw_data.data();
        input_samples.resize(n_samples);
        for (size_t i = 0; i < n_samples; ++i) {
            input_samples[i] = p[i] / 32768.0f;
        }
    } else if (header.bits_per_sample == 32) {
        size_t n_samples = chunk_size / sizeof(float);
        const float* p = (const float*)raw_data.data();
        input_samples.assign(p, p + n_samples);
    } else {
        return false;
    }

    std::vector<float> mono_samples;
    if (header.channels == 2) {
        size_t n_samples = input_samples.size() / 2;
        mono_samples.resize(n_samples);
        for (size_t i = 0; i < n_samples; ++i) {
            mono_samples[i] = (input_samples[2 * i] + input_samples[2 * i + 1]) * 0.5f;
        }
    } else if (header.channels == 1) {
        mono_samples = std::move(input_samples);
    } else {
        return false;
    }

    if (header.sample_rate == 24000) {
        out_samples = std::move(mono_samples);
    } else {
        double ratio = 24000.0 / (double)header.sample_rate;
        size_t out_size = (size_t)(mono_samples.size() * ratio);
        out_samples.resize(out_size);
        for (size_t i = 0; i < out_size; ++i) {
            double src_idx = (double)i / ratio;
            size_t idx_low = (size_t)std::floor(src_idx);
            size_t idx_high = std::min(idx_low + 1, mono_samples.size() - 1);
            double weight = src_idx - (double)idx_low;
            out_samples[i] = (float)((1.0 - weight) * mono_samples[idx_low] + weight * mono_samples[idx_high]);
        }
    }

    return true;
}
