#include "v3_native_assets.h"
#include <fstream>
#include <iostream>
#include <cstring>
#include <stdexcept>
#include <limits>
#include <cmath>
#include <sstream>
#include <algorithm>
#include <nlohmann/json.hpp>

// Endian helper functions
static uint16_t local_read_u16_le(const uint8_t* p) {
    return static_cast<uint16_t>(p[0]) | (static_cast<uint16_t>(p[1]) << 8);
}

static uint32_t local_read_u32_le(const uint8_t* p) {
    return static_cast<uint32_t>(p[0]) |
           (static_cast<uint32_t>(p[1]) << 8) |
           (static_cast<uint32_t>(p[2]) << 16) |
           (static_cast<uint32_t>(p[3]) << 24);
}

static uint64_t local_read_u64_le(const uint8_t* p) {
    return static_cast<uint64_t>(local_read_u32_le(p)) |
           (static_cast<uint64_t>(local_read_u32_le(p + 4)) << 32);
}

static float local_half_to_float(uint16_t h) {
    const uint16_t h_exp = h & 0x7C00u;
    const uint16_t h_sig = h & 0x03FFu;
    const uint32_t f_sgn = static_cast<uint32_t>(h & 0x8000u) << 16;
    uint32_t f;
    if (h_exp == 0) {
        if (h_sig == 0) {
            f = f_sgn;
        } else {
            uint16_t sig = h_sig;
            int exp = -1;
            do {
                exp++;
                sig <<= 1;
            } while ((sig & 0x0400u) == 0);
            sig &= 0x03FFu;
            const uint32_t f_exp = static_cast<uint32_t>(127 - 15 - exp) << 23;
            const uint32_t f_sig = static_cast<uint32_t>(sig) << 13;
            f = f_sgn | f_exp | f_sig;
        }
    } else if (h_exp == 0x7C00u) {
        f = f_sgn | 0x7F800000u | (static_cast<uint32_t>(h_sig) << 13);
    } else {
        const uint32_t f_exp = static_cast<uint32_t>((h_exp >> 10) + (127 - 15)) << 23;
        const uint32_t f_sig = static_cast<uint32_t>(h_sig) << 13;
        f = f_sgn | f_exp | f_sig;
    }
    float out;
    std::memcpy(&out, &f, sizeof(out));
    return out;
}

static std::string read_bytes(const std::string& path) {
    std::ifstream fs(path, std::ios::binary);
    if (!fs.is_open()) {
        throw std::runtime_error("failed to open file: " + path);
    }
    fs.seekg(0, std::ios::end);
    const size_t sz = fs.tellg();
    fs.seekg(0, std::ios::beg);
    std::string out(sz, '\0');
    fs.read(&out[0], sz);
    return out;
}

static std::vector<std::string> parse_shape(const std::string& shape_text) {
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

static V3NamedArray parse_npy_data(const uint8_t* data, size_t size, const std::string& name) {
    if (size < 16 || std::memcmp(data, "\x93NUMPY", 6) != 0) {
        throw std::runtime_error("invalid npy header for " + name);
    }
    const uint8_t major = data[6];
    size_t header_len = 0;
    size_t header_offset = 0;
    if (major == 1) {
        header_len = local_read_u16_le(data + 8);
        header_offset = 10;
    } else if (major == 2 || major == 3) {
        header_len = local_read_u32_le(data + 8);
        header_offset = 12;
    } else {
        throw std::runtime_error("unsupported npy version for " + name);
    }
    if (header_offset + header_len > size) {
        throw std::runtime_error("truncated npy header for " + name);
    }
    const std::string header(reinterpret_cast<const char*>(data + header_offset), header_len);
    const bool is_f16 = header.find("'descr': '<f2'") != std::string::npos || header.find("\"descr\": \"<f2\"") != std::string::npos;
    const bool is_f32 = header.find("'descr': '<f4'") != std::string::npos || header.find("\"descr\": \"<f4\"") != std::string::npos;
    if (!is_f16 && !is_f32) {
        throw std::runtime_error("unsupported npy dtype for " + name + " (expected float16 or float32)");
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

    V3NamedArray arr;
    const auto items = parse_shape(header.substr(paren_start, paren_end - paren_start + 1));
    int64_t count = 1;
    for (const std::string& item : items) {
        const int64_t dim = std::stoll(item);
        arr.shape.push_back(dim);
        count *= dim;
    }

    const size_t payload_offset = header_offset + header_len;
    const size_t element_bytes = is_f16 ? sizeof(uint16_t) : sizeof(float);
    const size_t payload_bytes = static_cast<size_t>(count) * element_bytes;
    if (payload_offset + payload_bytes > size) {
        throw std::runtime_error("truncated npy payload for " + name);
    }
    arr.data.resize(static_cast<size_t>(count));
    const uint8_t* p = data + payload_offset;
    if (is_f16) {
        for (int64_t i = 0; i < count; ++i) {
            arr.data[static_cast<size_t>(i)] = local_half_to_float(local_read_u16_le(p + i * 2));
        }
    } else {
        for (int64_t i = 0; i < count; ++i) {
            float v = 0.0f;
            std::memcpy(&v, p + static_cast<size_t>(i) * sizeof(float), sizeof(float));
            arr.data[static_cast<size_t>(i)] = v;
        }
    }
    return arr;
}

std::unordered_map<std::string, V3NamedArray> load_v3_npz(const std::string& path, std::string& error) {
    std::unordered_map<std::string, V3NamedArray> arrays;
    try {
        const std::string bytes = read_bytes(path);
        const auto* data = reinterpret_cast<const uint8_t*>(bytes.data());
        const size_t size = bytes.size();
        size_t off = 0;

        while (off + 30 <= size) {
            const uint32_t sig = local_read_u32_le(data + off);
            if (sig != 0x04034b50u) {
                break;
            }
            const uint16_t method = local_read_u16_le(data + off + 8);
            const uint32_t compressed_size32 = local_read_u32_le(data + off + 18);
            const uint32_t uncompressed_size32 = local_read_u32_le(data + off + 22);
            const uint16_t name_len = local_read_u16_le(data + off + 26);
            const uint16_t extra_len = local_read_u16_le(data + off + 28);
            const size_t name_off = off + 30;
            const size_t payload_off = name_off + name_len + extra_len;
            uint64_t compressed_size64 = compressed_size32;
            uint64_t uncompressed_size64 = uncompressed_size32;
            if (compressed_size32 == 0xFFFFFFFFu || uncompressed_size32 == 0xFFFFFFFFu) {
                bool found_zip64 = false;
                size_t extra_off = name_off + name_len;
                const size_t extra_end = extra_off + extra_len;
                while (extra_off + 4 <= extra_end) {
                    const uint16_t field_id = local_read_u16_le(data + extra_off);
                    const uint16_t field_size = local_read_u16_le(data + extra_off + 2);
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
                            uncompressed_size64 = local_read_u64_le(data + zip64_off);
                            zip64_off += 8;
                        }
                        if (compressed_size32 == 0xFFFFFFFFu) {
                            if (zip64_off + 8 > field_payload + field_size) {
                                throw std::runtime_error("truncated zip64 compressed size in " + path);
                            }
                            compressed_size64 = local_read_u64_le(data + zip64_off);
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
            arrays[name] = parse_npy_data(data + payload_off, uncompressed_size, name);
            off = payload_off + compressed_size;
        }
    } catch (const std::exception& e) {
        error = e.what();
    }
    return arrays;
}

static std::string join_paths(const std::string& a, const std::string& b) {
    if (a.empty()) return b;
    const char last = a[a.size() - 1];
    if (last == '/' || last == '\\') return a + b;
#ifdef _WIN32
    return a + "\\" + b;
#else
    return a + "/" + b;
#endif
}

static std::vector<float> transpose_2d_local(const std::vector<float>& src, int64_t rows, int64_t cols) {
    std::vector<float> dst(static_cast<size_t>(rows * cols));
    for (int64_t r = 0; r < rows; ++r) {
        const float* row = src.data() + r * cols;
        for (int64_t c = 0; c < cols; ++c) {
            dst[static_cast<size_t>(c * rows + r)] = row[c];
        }
    }
    return dst;
}

static std::vector<float> transpose_audio_emb_local(const std::vector<float>& src, int64_t channels, int64_t vocab, int64_t hidden) {
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

bool V3NativeAssets::load(const std::string& model_dir, std::string& error) {
    if (!load_config(join_paths(model_dir, "config.json"), error)) return false;
    if (!load_heads(join_paths(model_dir, "vieneu_v3_heads.npz"), error)) return false;
    
    // Check if acoustic folder exists inside model_dir
    std::string acoustic_path = join_paths(join_paths(model_dir, "acoustic"), "vieneu_acoustic_weights.npz");
    if (!load_acoustic(acoustic_path, error)) return false;
    
    if (!load_voices(join_paths(model_dir, "voices_v3_turbo.json"), error)) return false;

    return true;
}

bool V3NativeAssets::load_config(const std::string& path, std::string& error) {
    try {
        const std::string bytes = read_bytes(path);
        const auto c = nlohmann::json::parse(bytes);
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
        config_.local_num_hidden_layers = c.value("local_num_hidden_layers", config_.local_num_hidden_layers);
        config_.local_intermediate_size = c.value("local_intermediate_size", config_.local_intermediate_size);
        config_.rms_norm_eps = c.value("rms_norm_eps", config_.rms_norm_eps);
        return true;
    } catch (const std::exception& e) {
        error = std::string("Failed to load native config: ") + e.what();
        return false;
    }
}

bool V3NativeAssets::load_heads(const std::string& path, std::string& error) {
    try {
        auto arrays = load_v3_npz(path, error);
        if (!error.empty()) {
            return false;
        }
        auto text_it = arrays.find("text_emb.npy");
        auto audio_it = arrays.find("audio_emb.npy");
        if (text_it == arrays.end()) text_it = arrays.find("text_emb");
        if (audio_it == arrays.end()) audio_it = arrays.find("audio_emb");

        if (text_it == arrays.end() || audio_it == arrays.end()) {
            error = "vieneu_v3_heads.npz is missing text_emb or audio_emb";
            return false;
        }

        const auto& text = text_it->second;
        const auto& audio = audio_it->second;
        if (text.shape.size() != 2 || audio.shape.size() != 3) {
            error = "Unexpected embedding rank in vieneu_v3_heads.npz";
            return false;
        }
        if (text.shape[0] != config_.text_vocab_size ||
            text.shape[1] != config_.hidden_size ||
            audio.shape[0] != config_.n_vq ||
            audio.shape[1] != config_.audio_vocab_size ||
            audio.shape[2] != config_.hidden_size) {
            error = "Embedding shapes in vieneu_v3_heads.npz do not match config.json";
            return false;
        }

        text_emb_ = text.data;
        audio_emb_ = audio.data;
        text_emb_t_ = transpose_2d_local(text_emb_, config_.text_vocab_size, config_.hidden_size);
        audio_emb_t_ = transpose_audio_emb_local(audio_emb_, config_.n_vq, config_.audio_vocab_size, config_.hidden_size);
        return true;
    } catch (const std::exception& e) {
        error = std::string("Failed to load vieneu_v3_heads.npz: ") + e.what();
        return false;
    }
}

bool V3NativeAssets::load_acoustic(const std::string& path, std::string& error) {
    try {
        auto arrays = load_v3_npz(path, error);
        if (!error.empty()) {
            return false;
        }

        auto take = [&](const std::string& name, std::vector<float>& dst) -> bool {
            auto it = arrays.find(name + ".npy");
            if (it == arrays.end()) it = arrays.find(name);
            if (it == arrays.end()) {
                error = "Acoustic weights are missing tensor: " + name;
                return false;
            }
            dst = it->second.data;
            return true;
        };

        if (!take("slot_pos_emb", acoustic_weights_.slot_pos_emb)) return false;
        if (!take("norm", acoustic_weights_.final_norm)) return false;

        const int num_layers = config_.local_num_hidden_layers;
        acoustic_weights_.layers.resize(num_layers);
        for (int i = 0; i < num_layers; ++i) {
            std::string prefix = "layers." + std::to_string(i) + ".";
            V3AcousticLayerWeights& l = acoustic_weights_.layers[i];
            if (!take(prefix + "norm1", l.norm1)) return false;
            if (!take(prefix + "attn.qkv", l.qkv)) return false;
            if (!take(prefix + "attn.q_norm", l.q_norm)) return false;
            if (!take(prefix + "attn.k_norm", l.k_norm)) return false;
            if (!take(prefix + "attn.o_proj", l.o_proj)) return false;
            if (!take(prefix + "norm2", l.norm2)) return false;
            if (!take(prefix + "ff_up", l.ff_up)) return false;
            if (!take(prefix + "ff_gate", l.ff_gate)) return false;
            if (!take(prefix + "ff_down", l.ff_down)) return false;
        }

        acoustic_weights_.loaded = true;
        return true;
    } catch (const std::exception& e) {
        error = std::string("Failed to load vieneu_acoustic_weights.npz: ") + e.what();
        return false;
    }
}

bool V3NativeAssets::load_voices(const std::string& path, std::string& error) {
    try {
        voices_json_ = read_bytes(path);
        return true;
    } catch (const std::exception& e) {
        error = std::string("Failed to load voices: ") + e.what();
        return false;
    }
}
