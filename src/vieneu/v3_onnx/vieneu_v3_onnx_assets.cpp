#include "../vieneu_v3_onnx.h"
#include "vieneu_v3_onnx_internal.h"

#include <algorithm>
#include <cctype>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>
#include <nlohmann/json.hpp>

// --- Helper transposes ---

std::vector<float> transpose_2d(const std::vector<float>& src, int64_t rows, int64_t cols) {
    std::vector<float> dst(static_cast<size_t>(rows * cols));
    for (int64_t r = 0; r < rows; ++r) {
        const float* row = src.data() + r * cols;
        for (int64_t c = 0; c < cols; ++c) {
            dst[static_cast<size_t>(c * rows + r)] = row[c];
        }
    }
    return dst;
}

std::vector<float> transpose_audio_emb(const std::vector<float>& src, int64_t channels, int64_t vocab, int64_t hidden) {
    std::vector<float> dst(static_cast<size_t>(channels * hidden * vocab));
    for (int64_t ch = 0; ch < channels; ++ch) {
        for (int64_t v = 0; v < vocab; ++v) {
            const float* emb = src.data() + (ch * vocab + v) * hidden;
            for (int64_t h = 0; h < hidden; ++h) {
                dst[static_cast<size_t>((ch * hidden + h) * vocab + v)] = emb[h];
            }
        }
    }
    return dst;
}

// --- NPZ / NPY Loader Helpers ---

std::vector<std::string> parse_shape_items(const std::string& shape_text) {
    std::vector<std::string> out;
    std::string cur;
    for (char c : shape_text) {
        if (c == ',') {
            if (!cur.empty()) {
                out.push_back(cur);
                cur.clear();
            }
        } else if (!std::isspace(static_cast<unsigned char>(c)) && c != '(' && c != ')') {
            cur.push_back(c);
        }
    }
    if (!cur.empty()) {
        out.push_back(cur);
    }
    return out;
}

NamedArray parse_npy(const uint8_t* data, size_t size, const std::string& name) {
    if (size < 16 || std::memcmp(data, "\x93NUMPY", 6) != 0) {
        throw std::runtime_error("invalid npy header for " + name);
    }
    const uint8_t major = data[6];
    size_t header_len = 0;
    size_t header_offset = 0;
    if (major == 1) {
        header_len = read_u16_le(data + 8);
        header_offset = 10;
    } else if (major == 2 || major == 3) {
        header_len = read_u32_le(data + 8);
        header_offset = 12;
    } else {
        throw std::runtime_error("unsupported npy version for " + name);
    }
    if (header_offset + header_len > size) {
        throw std::runtime_error("truncated npy header for " + name);
    }
    const std::string header(reinterpret_cast<const char*>(data + header_offset), header_len);
    if (header.find("'descr': '<f2'") == std::string::npos && header.find("\"descr\": \"<f2\"") == std::string::npos) {
        throw std::runtime_error("unsupported npy dtype for " + name + " (expected float16)");
    }
    if (header.find("True") != std::string::npos) {
        throw std::runtime_error("fortran-order npy arrays are not supported for " + name);
    }
    const size_t shape_pos = header.find("'shape':");
    const size_t paren_start = header.find('(', shape_pos);
    const size_t paren_end = header.find(')', paren_start);
    if (shape_pos == std::string::npos || paren_start == std::string::npos || paren_end == std::string::npos) {
        throw std::runtime_error("missing npy shape for " + name);
    }

    NamedArray arr;
    const auto items = parse_shape_items(header.substr(paren_start, paren_end - paren_start + 1));
    int64_t count = 1;
    for (const std::string& item : items) {
        const int64_t dim = std::stoll(item);
        arr.shape.push_back(dim);
        count *= dim;
    }

    const size_t payload_offset = header_offset + header_len;
    const size_t payload_bytes = static_cast<size_t>(count) * sizeof(uint16_t);
    if (payload_offset + payload_bytes > size) {
        throw std::runtime_error("truncated npy payload for " + name);
    }
    arr.data.resize(static_cast<size_t>(count));
    const uint8_t* p = data + payload_offset;
    for (int64_t i = 0; i < count; ++i) {
        arr.data[static_cast<size_t>(i)] = half_to_float(read_u16_le(p + i * 2));
    }
    return arr;
}

std::unordered_map<std::string, NamedArray> load_npz_stored(const std::string& path) {
    const std::string bytes = read_file_bytes(path);
    const auto* data = reinterpret_cast<const uint8_t*>(bytes.data());
    const size_t size = bytes.size();
    size_t off = 0;
    std::unordered_map<std::string, NamedArray> arrays;

    while (off + 30 <= size) {
        const uint32_t sig = read_u32_le(data + off);
        if (sig != 0x04034b50u) {
            break;
        }
        const uint16_t method = read_u16_le(data + off + 8);
        const uint32_t compressed_size32 = read_u32_le(data + off + 18);
        const uint32_t uncompressed_size32 = read_u32_le(data + off + 22);
        const uint16_t name_len = read_u16_le(data + off + 26);
        const uint16_t extra_len = read_u16_le(data + off + 28);
        const size_t name_off = off + 30;
        const size_t payload_off = name_off + name_len + extra_len;
        uint64_t compressed_size64 = compressed_size32;
        uint64_t uncompressed_size64 = uncompressed_size32;
        if (compressed_size32 == 0xFFFFFFFFu || uncompressed_size32 == 0xFFFFFFFFu) {
            bool found_zip64 = false;
            size_t extra_off = name_off + name_len;
            const size_t extra_end = extra_off + extra_len;
            while (extra_off + 4 <= extra_end) {
                const uint16_t field_id = read_u16_le(data + extra_off);
                const uint16_t field_size = read_u16_le(data + extra_off + 2);
                const size_t field_payload = extra_off + 4;
                if (field_payload + field_size > extra_end) {
                    throw std::runtime_error("truncated zip extra field in " + path);
                }
                if (field_id == 0x0001u) {
                    found_zip64 = true;
                    size_t zip64_off = field_payload;
                    if (uncompressed_size32 == 0xFFFFFFFFu) {
                        if (zip64_off + 8 > field_payload + field_size) {
                            throw std::runtime_error("truncated zip64 uncompressed size in " + path);
                        }
                        uncompressed_size64 = read_u64_le(data + zip64_off);
                        zip64_off += 8;
                    }
                    if (compressed_size32 == 0xFFFFFFFFu) {
                        if (zip64_off + 8 > field_payload + field_size) {
                            throw std::runtime_error("truncated zip64 compressed size in " + path);
                        }
                        compressed_size64 = read_u64_le(data + zip64_off);
                    }
                    break;
                }
                extra_off = field_payload + field_size;
            }
            if (!found_zip64) {
                throw std::runtime_error("missing zip64 size extra field in " + path);
            }
        }
        if (compressed_size64 > static_cast<uint64_t>(std::numeric_limits<size_t>::max()) ||
            uncompressed_size64 > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
            throw std::runtime_error("npz entry is too large in " + path);
        }
        const size_t compressed_size = static_cast<size_t>(compressed_size64);
        const size_t uncompressed_size = static_cast<size_t>(uncompressed_size64);
        if (payload_off > size || payload_off + compressed_size > size) {
            throw std::runtime_error("truncated npz entry in " + path);
        }
        std::string name(reinterpret_cast<const char*>(data + name_off), name_len);
        if (method != 0) {
            throw std::runtime_error("compressed npz entries are not supported: " + name);
        }
        if (compressed_size != uncompressed_size) {
            throw std::runtime_error("invalid stored npz size for " + name);
        }
        arrays[name] = parse_npy(data + payload_off, uncompressed_size, name);
        off = payload_off + compressed_size;
    }
    return arrays;
}

// --- VieneuV3OnnxEngine Member Functions ---

bool VieneuV3OnnxEngine::load_session(const std::string& path, std::unique_ptr<Ort::Session>& session, std::string& error) {
    try {
#ifdef _WIN32
        std::wstring w_path(path.begin(), path.end());
        session = std::make_unique<Ort::Session>(*env_, w_path.c_str(), *session_options_);
#else
        session = std::make_unique<Ort::Session>(*env_, path.c_str(), *session_options_);
#endif
        return true;
    } catch (const std::exception& e) {
        error = "Failed to load ONNX session " + path + ": " + e.what();
        return false;
    }
}

void VieneuV3OnnxEngine::cache_session_io(Ort::Session& session, SessionIo& io) {
    io.input_names = session_input_names(session);
    io.output_names = session_output_names(session);
    io.input_ptrs = name_ptrs(io.input_names);
    io.output_ptrs = name_ptrs(io.output_names);
}

bool VieneuV3OnnxEngine::validate_assets(const VieneuV3OnnxInit& init, std::string& error) {
    onnx_dir_ = init.onnx_dir.empty() ? init.model_dir : init.onnx_dir;
    model_dir_ = init.model_dir.empty() ? onnx_dir_ : init.model_dir;
    codec_dir_ = init.codec_dir;
    const std::string config_path = init.config_path.empty() ? join_path(model_dir_, "config.json") : init.config_path;
    const std::string tokenizer_path = init.tokenizer_path.empty() ? join_path(model_dir_, "tokenizer.json") : init.tokenizer_path;

    const std::vector<std::string> required = {
        join_path(onnx_dir_, "vieneu_prefill.onnx"),
        join_path(onnx_dir_, "vieneu_decode_step.onnx"),
        join_path(onnx_dir_, "vieneu_acoustic_cached.onnx"),
        join_path(onnx_dir_, "vieneu_v3_heads.npz"),
        config_path,
        tokenizer_path,
        join_path(codec_dir_, "moss_audio_tokenizer_decode_full.onnx"),
        join_path(codec_dir_, "moss_audio_tokenizer_encode.onnx"),
    };

    for (const std::string& path : required) {
        if (!file_exists(path)) {
            error = "Missing required VieNeu v3 ONNX asset: " + path;
            return false;
        }
    }
    codec_encode_path_ = join_path(codec_dir_, "moss_audio_tokenizer_encode.onnx");
    return true;
}

bool VieneuV3OnnxEngine::load_config(const std::string& path, std::string& error) {
    try {
        const auto c = nlohmann::json::parse(read_file_bytes(path));
        config_.n_vq = c.value("n_vq", config_.n_vq);
        config_.hidden_size = c.value("hidden_size", config_.hidden_size);
        config_.num_hidden_layers = c.value("num_hidden_layers", config_.num_hidden_layers);
        config_.audio_pad_token_id = c.value("audio_pad_token_id", config_.audio_pad_token_id);
        config_.text_prompt_start_token_id = c.value("text_prompt_start_token_id", config_.text_prompt_start_token_id);
        config_.text_prompt_end_token_id = c.value("text_prompt_end_token_id", config_.text_prompt_end_token_id);
        config_.speech_generation_start_token_id = c.value("speech_generation_start_token_id", config_.speech_generation_start_token_id);
        config_.speech_generation_end_token_id = c.value("speech_generation_end_token_id", config_.speech_generation_end_token_id);
        config_.audio_ref_slot_token_id = c.value("audio_ref_slot_token_id", config_.audio_ref_slot_token_id);
        config_.emotion_0_token_id = c.value("emotion_0_token_id", config_.emotion_0_token_id);
        config_.emotion_4_token_id = c.value("emotion_4_token_id", config_.emotion_4_token_id);
        config_.text_vocab_size = c.value("text_vocab_size", config_.text_vocab_size);
        config_.audio_vocab_size = c.value("audio_vocab_size", config_.audio_vocab_size);
        config_.local_num_attention_heads = c.value("local_num_attention_heads", config_.local_num_attention_heads);
        return true;
    } catch (const std::exception& e) {
        error = std::string("Failed to load VieNeu v3 config: ") + e.what();
        return false;
    }
}

bool VieneuV3OnnxEngine::load_heads_npz(const std::string& path, std::string& error) {
    try {
        auto arrays = load_npz_stored(path);
        auto text_it = arrays.find("text_emb.npy");
        auto audio_it = arrays.find("audio_emb.npy");
        if (text_it == arrays.end() || audio_it == arrays.end()) {
            error = "vieneu_v3_heads.npz is missing text_emb.npy or audio_emb.npy";
            return false;
        }
        const auto& text = text_it->second;
        const auto& audio = audio_it->second;
        if (text.shape.size() != 2 || audio.shape.size() != 3) {
            error = "Unexpected embedding rank in vieneu_v3_heads.npz";
            return false;
        }
        text_emb_.rows = text.shape[0];
        text_emb_.cols = text.shape[1];
        text_emb_.data = text.data;
        text_emb_t_.rows = text_emb_.cols;
        text_emb_t_.cols = text_emb_.rows;
        text_emb_t_.data = transpose_2d(text_emb_.data, text_emb_.rows, text_emb_.cols);
        audio_emb_.dim0 = audio.shape[0];
        audio_emb_.dim1 = audio.shape[1];
        audio_emb_.dim2 = audio.shape[2];
        audio_emb_.data = audio.data;
        audio_emb_t_.dim0 = audio_emb_.dim0;
        audio_emb_t_.dim1 = audio_emb_.dim2;
        audio_emb_t_.dim2 = audio_emb_.dim1;
        audio_emb_t_.data = transpose_audio_emb(audio_emb_.data, audio_emb_.dim0, audio_emb_.dim1, audio_emb_.dim2);
        if (text_emb_.cols != config_.hidden_size || audio_emb_.dim2 != config_.hidden_size) {
            error = "Embedding hidden size does not match config.json";
            return false;
        }
        return true;
    } catch (const std::exception& e) {
        error = std::string("Failed to load vieneu_v3_heads.npz: ") + e.what();
        return false;
    }
}
