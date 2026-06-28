#include "vieneu_v3_onnx.h"

#include "vieneu.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <utility>

#include <nlohmann/json.hpp>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <sys/stat.h>
#endif

namespace {

struct NamedArray {
    std::vector<int64_t> shape;
    std::vector<float> data;
};

struct TensorBlob {
    std::vector<int64_t> shape;
    std::vector<float> data;
};

uint16_t read_u16_le(const uint8_t* p) {
    return static_cast<uint16_t>(p[0]) | (static_cast<uint16_t>(p[1]) << 8);
}

uint32_t read_u32_le(const uint8_t* p) {
    return static_cast<uint32_t>(p[0]) |
           (static_cast<uint32_t>(p[1]) << 8) |
           (static_cast<uint32_t>(p[2]) << 16) |
           (static_cast<uint32_t>(p[3]) << 24);
}

uint64_t read_u64_le(const uint8_t* p) {
    return static_cast<uint64_t>(read_u32_le(p)) |
           (static_cast<uint64_t>(read_u32_le(p + 4)) << 32);
}

int16_t read_i16_le(const uint8_t* p) {
    return static_cast<int16_t>(read_u16_le(p));
}

int32_t read_i24_le(const uint8_t* p) {
    int32_t v = static_cast<int32_t>(p[0]) |
                (static_cast<int32_t>(p[1]) << 8) |
                (static_cast<int32_t>(p[2]) << 16);
    if (v & 0x00800000) {
        v |= static_cast<int32_t>(0xFF000000);
    }
    return v;
}

int32_t read_i32_le(const uint8_t* p) {
    return static_cast<int32_t>(read_u32_le(p));
}

float half_to_float(uint16_t h) {
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

std::string read_file_bytes(const std::string& path) {
    std::ifstream fs(path, std::ios::binary);
    if (!fs.is_open()) {
        throw std::runtime_error("failed to open file: " + path);
    }
    std::ostringstream ss;
    ss << fs.rdbuf();
    return ss.str();
}

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

std::string utf8_from_codepoint(uint32_t cp) {
    std::string out;
    if (cp <= 0x7F) {
        out.push_back(static_cast<char>(cp));
    } else if (cp <= 0x7FF) {
        out.push_back(static_cast<char>(0xC0 | (cp >> 6)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else if (cp <= 0xFFFF) {
        out.push_back(static_cast<char>(0xE0 | (cp >> 12)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else {
        out.push_back(static_cast<char>(0xF0 | (cp >> 18)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    }
    return out;
}

std::array<std::string, 256> build_byte_encoder() {
    std::vector<int> bs;
    for (int i = static_cast<int>('!'); i <= static_cast<int>('~'); ++i) bs.push_back(i);
    for (int i = 0xA1; i <= 0xAC; ++i) bs.push_back(i);
    for (int i = 0xAE; i <= 0xFF; ++i) bs.push_back(i);
    std::vector<int> cs = bs;
    int n = 0;
    for (int b = 0; b < 256; ++b) {
        if (std::find(bs.begin(), bs.end(), b) == bs.end()) {
            bs.push_back(b);
            cs.push_back(256 + n);
            ++n;
        }
    }
    std::array<std::string, 256> enc;
    for (size_t i = 0; i < bs.size(); ++i) {
        enc[static_cast<size_t>(bs[i])] = utf8_from_codepoint(static_cast<uint32_t>(cs[i]));
    }
    return enc;
}

std::vector<std::string> byte_level_symbols(const std::string& text) {
    static const auto enc = build_byte_encoder();
    std::vector<std::string> out;
    out.reserve(text.size());
    for (unsigned char c : text) {
        out.push_back(enc[static_cast<size_t>(c)]);
    }
    return out;
}

std::string join_pair_key(const std::string& a, const std::string& b) {
    return a + "\n" + b;
}

std::vector<int64_t> tensor_shape(const Ort::Value& value) {
    return value.GetTensorTypeAndShapeInfo().GetShape();
}

TensorBlob copy_float_tensor(const Ort::Value& value) {
    TensorBlob blob;
    blob.shape = tensor_shape(value);
    size_t count = 1;
    for (int64_t dim : blob.shape) {
        count *= static_cast<size_t>(dim);
    }
    const float* src = value.GetTensorData<float>();
    blob.data.assign(src, src + count);
    return blob;
}

std::vector<float> softmax(const std::vector<float>& logits) {
    float max_v = -std::numeric_limits<float>::infinity();
    for (float v : logits) {
        if (v > max_v) max_v = v;
    }
    double sum = 0.0;
    std::vector<float> probs(logits.size());
    for (size_t i = 0; i < logits.size(); ++i) {
        const double e = std::exp(static_cast<double>(logits[i] - max_v));
        probs[i] = static_cast<float>(e);
        sum += e;
    }
    if (sum <= 0.0 || !std::isfinite(sum)) {
        const float uniform = logits.empty() ? 0.0f : 1.0f / static_cast<float>(logits.size());
        std::fill(probs.begin(), probs.end(), uniform);
        return probs;
    }
    for (float& p : probs) {
        p = static_cast<float>(static_cast<double>(p) / sum);
    }
    return probs;
}

std::vector<std::string> session_input_names(Ort::Session& session) {
    Ort::AllocatorWithDefaultOptions allocator;
    std::vector<std::string> names;
    const size_t count = session.GetInputCount();
    names.reserve(count);
    for (size_t i = 0; i < count; ++i) {
        auto name = session.GetInputNameAllocated(i, allocator);
        names.emplace_back(name.get());
    }
    return names;
}

std::vector<std::string> session_output_names(Ort::Session& session) {
    Ort::AllocatorWithDefaultOptions allocator;
    std::vector<std::string> names;
    const size_t count = session.GetOutputCount();
    names.reserve(count);
    for (size_t i = 0; i < count; ++i) {
        auto name = session.GetOutputNameAllocated(i, allocator);
        names.emplace_back(name.get());
    }
    return names;
}

std::vector<const char*> name_ptrs(const std::vector<std::string>& names) {
    std::vector<const char*> ptrs;
    ptrs.reserve(names.size());
    for (const auto& name : names) {
        ptrs.push_back(name.c_str());
    }
    return ptrs;
}

} // namespace

bool VieneuV3OnnxEngine::ByteBpeTokenizer::load(const std::string& path, std::string& error) {
    try {
        const auto doc = nlohmann::json::parse(read_file_bytes(path));
        const auto& model = doc.at("model");
        vocab.clear();
        merge_ranks.clear();
        for (auto it = model.at("vocab").begin(); it != model.at("vocab").end(); ++it) {
            vocab[it.key()] = it.value().get<int64_t>();
        }
        if (model.contains("unk_token")) {
            const std::string unk = model.at("unk_token").get<std::string>();
            auto it = vocab.find(unk);
            if (it != vocab.end()) {
                unk_id = it->second;
            }
        }
        int rank = 0;
        for (const auto& merge : model.at("merges")) {
            if (merge.is_array() && merge.size() == 2) {
                merge_ranks[join_pair_key(merge[0].get<std::string>(), merge[1].get<std::string>())] = rank++;
            } else if (merge.is_string()) {
                std::istringstream ss(merge.get<std::string>());
                std::string a, b;
                if (ss >> a >> b) {
                    merge_ranks[join_pair_key(a, b)] = rank++;
                }
            }
        }
        return true;
    } catch (const std::exception& e) {
        error = std::string("Failed to load tokenizer: ") + e.what();
        return false;
    }
}

std::vector<int64_t> VieneuV3OnnxEngine::ByteBpeTokenizer::encode(const std::string& text) const {
    std::vector<std::string> word = byte_level_symbols(text);
    if (word.empty()) {
        return {};
    }
    while (word.size() > 1) {
        int best_rank = (std::numeric_limits<int>::max)();
        size_t best_idx = static_cast<size_t>(-1);
        for (size_t i = 0; i + 1 < word.size(); ++i) {
            auto it = merge_ranks.find(join_pair_key(word[i], word[i + 1]));
            if (it != merge_ranks.end() && it->second < best_rank) {
                best_rank = it->second;
                best_idx = i;
            }
        }
        if (best_idx == static_cast<size_t>(-1)) {
            break;
        }
        word[best_idx] += word[best_idx + 1];
        word.erase(word.begin() + static_cast<std::ptrdiff_t>(best_idx + 1));
    }

    std::vector<int64_t> ids;
    ids.reserve(word.size());
    for (const std::string& token : word) {
        auto it = vocab.find(token);
        ids.push_back(it == vocab.end() ? unk_id : it->second);
    }
    return ids;
}

std::string VieneuV3OnnxEngine::join_path(const std::string& dir, const std::string& name) {
    if (dir.empty()) {
        return name;
    }
    const char last = dir[dir.size() - 1];
    if (last == '/' || last == '\\') {
        return dir + name;
    }
#ifdef _WIN32
    return dir + "\\" + name;
#else
    return dir + "/" + name;
#endif
}

bool VieneuV3OnnxEngine::file_exists(const std::string& path) {
    if (path.empty()) {
        return false;
    }
#ifdef _WIN32
    const DWORD attrs = GetFileAttributesA(path.c_str());
    return attrs != INVALID_FILE_ATTRIBUTES && !(attrs & FILE_ATTRIBUTE_DIRECTORY);
#else
    struct stat st;
    return stat(path.c_str(), &st) == 0 && S_ISREG(st.st_mode);
#endif
}

bool VieneuV3OnnxEngine::read_text_file(const std::string& path, std::string& out) {
    try {
        out = read_file_bytes(path);
        return true;
    } catch (...) {
        return false;
    }
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
        audio_emb_.dim0 = audio.shape[0];
        audio_emb_.dim1 = audio.shape[1];
        audio_emb_.dim2 = audio.shape[2];
        audio_emb_.data = audio.data;
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

bool VieneuV3OnnxEngine::load_voices(const std::string& voices_path, std::string& error) {
    voices_json_.clear();
    if (voices_path.empty()) {
        return true;
    }
    if (!read_text_file(voices_path, voices_json_)) {
        error = "Failed to read VieNeu v3 voices JSON: " + voices_path;
        return false;
    }
    return true;
}

bool VieneuV3OnnxEngine::parse_voice_reserved_id(const std::string& voice_id, int& reserved_id) const {
    static const std::unordered_map<std::string, int> fallback = {
        {"Ngọc Lan", 13}, {"Ngọc Linh", 14}, {"Trúc Ly", 15}, {"Mỹ Duyên", 16},
        {"Xuân Vĩnh", 17}, {"Thái Sơn", 18}, {"Gia Bảo", 19}, {"Đức Trí", 20},
        {"Trọng Hữu", 21}, {"Bình An", 22}
    };
    if (!voice_id.empty()) {
        auto it = fallback.find(voice_id);
        if (it != fallback.end()) {
            reserved_id = it->second;
            return true;
        }
    }
    if (voices_json_.empty() || voice_id.empty()) {
        return false;
    }
    try {
        const auto root = nlohmann::json::parse(voices_json_);
        const auto& presets = root.at("presets");
        if (presets.contains(voice_id) && presets.at(voice_id).contains("reserved_id")) {
            reserved_id = presets.at(voice_id).at("reserved_id").get<int>();
            return true;
        }
    } catch (...) {
    }
    return false;
}

bool VieneuV3OnnxEngine::resolve_voice_preset(
    const std::string& voice_id,
    VoicePreset& preset,
    std::string& error) const {
    preset = VoicePreset{};
    if (voices_json_.empty()) {
        return true;
    }

    try {
        const auto root = nlohmann::json::parse(voices_json_);
        if (!root.contains("presets") || !root.at("presets").is_object()) {
            return true;
        }

        std::string selected = voice_id;
        if (selected.empty() && root.contains("default_voice") && root.at("default_voice").is_string()) {
            selected = root.at("default_voice").get<std::string>();
        }
        if (selected.empty()) {
            return true;
        }

        const auto& presets = root.at("presets");
        if (!presets.contains(selected)) {
            error = "VieNeu v3 voice preset not found: " + selected;
            return false;
        }
        const auto& item = presets.at(selected);
        preset.found = true;
        if (item.contains("reserved_id") && !item.at("reserved_id").is_null()) {
            preset.has_reserved_id = true;
            preset.reserved_id = item.at("reserved_id").get<int>();
        }
        if (item.contains("codes") && item.at("codes").is_array()) {
            const auto& codes = item.at("codes");
            for (const auto& row : codes) {
                if (!row.is_array() || static_cast<int>(row.size()) != config_.n_vq) {
                    error = "VieNeu v3 preset voice has invalid codes shape: " + selected;
                    return false;
                }
                for (const auto& v : row) {
                    preset.codes.push_back(v.get<int64_t>());
                }
            }
        }
        return true;
    } catch (const std::exception& e) {
        error = std::string("Failed to parse VieNeu v3 voices JSON: ") + e.what();
        return false;
    }
}

bool VieneuV3OnnxEngine::read_wav_file(const std::string& path, WavData& wav, std::string& error) const {
    wav = WavData{};
    try {
        const std::string bytes = read_file_bytes(path);
        const auto* data = reinterpret_cast<const uint8_t*>(bytes.data());
        const size_t size = bytes.size();
        if (size < 44 || std::memcmp(data, "RIFF", 4) != 0 || std::memcmp(data + 8, "WAVE", 4) != 0) {
            error = "Reference audio must be a RIFF/WAVE file: " + path;
            return false;
        }

        uint16_t audio_format = 0;
        uint16_t channels = 0;
        uint32_t sample_rate = 0;
        uint16_t bits_per_sample = 0;
        const uint8_t* pcm = nullptr;
        size_t pcm_size = 0;
        size_t off = 12;
        while (off + 8 <= size) {
            const char* id = reinterpret_cast<const char*>(data + off);
            const uint32_t chunk_size = read_u32_le(data + off + 4);
            const size_t payload = off + 8;
            if (payload + chunk_size > size) {
                error = "Truncated WAV chunk in reference audio: " + path;
                return false;
            }
            if (std::memcmp(id, "fmt ", 4) == 0) {
                if (chunk_size < 16) {
                    error = "Invalid WAV fmt chunk in reference audio: " + path;
                    return false;
                }
                audio_format = read_u16_le(data + payload);
                channels = read_u16_le(data + payload + 2);
                sample_rate = read_u32_le(data + payload + 4);
                bits_per_sample = read_u16_le(data + payload + 14);
            } else if (std::memcmp(id, "data", 4) == 0) {
                pcm = data + payload;
                pcm_size = chunk_size;
            }
            off = payload + chunk_size + (chunk_size & 1u);
        }

        if (!pcm || pcm_size == 0 || channels == 0 || sample_rate == 0 || bits_per_sample == 0) {
            error = "Reference WAV is missing fmt/data chunks: " + path;
            return false;
        }
        if (audio_format != 1 && audio_format != 3) {
            error = "Reference WAV must be PCM or IEEE-float format: " + path;
            return false;
        }
        const size_t bytes_per_sample = bits_per_sample / 8;
        if (bytes_per_sample == 0 || pcm_size < bytes_per_sample * channels) {
            error = "Reference WAV has invalid sample size: " + path;
            return false;
        }

        const size_t frames = pcm_size / (bytes_per_sample * channels);
        wav.sample_rate = static_cast<int>(sample_rate);
        wav.channels = static_cast<int>(channels);
        wav.samples.resize(frames * channels);
        for (size_t i = 0; i < frames * channels; ++i) {
            const uint8_t* p = pcm + i * bytes_per_sample;
            float v = 0.0f;
            if (audio_format == 3 && bits_per_sample == 32) {
                std::memcpy(&v, p, sizeof(float));
            } else if (audio_format == 1 && bits_per_sample == 16) {
                v = static_cast<float>(read_i16_le(p)) / 32768.0f;
            } else if (audio_format == 1 && bits_per_sample == 24) {
                v = static_cast<float>(read_i24_le(p)) / 8388608.0f;
            } else if (audio_format == 1 && bits_per_sample == 32) {
                v = static_cast<float>(read_i32_le(p)) / 2147483648.0f;
            } else if (audio_format == 1 && bits_per_sample == 8) {
                v = (static_cast<float>(*p) - 128.0f) / 128.0f;
            } else {
                error = "Unsupported reference WAV sample format: " + path;
                return false;
            }
            wav.samples[i] = std::clamp(v, -1.0f, 1.0f);
        }
        return true;
    } catch (const std::exception& e) {
        error = std::string("Failed to read reference WAV: ") + e.what();
        return false;
    }
}

bool VieneuV3OnnxEngine::encode_reference_audio(
    const std::string& path,
    std::vector<int64_t>& out_codes,
    std::string& error) {
    out_codes.clear();
    WavData wav;
    if (!read_wav_file(path, wav, error)) {
        return false;
    }

    const int target_sr = sample_rate();
    const int64_t in_frames = static_cast<int64_t>(wav.samples.size() / wav.channels);
    const int64_t out_frames = wav.sample_rate == target_sr
        ? in_frames
        : static_cast<int64_t>(std::llround(static_cast<double>(in_frames) * target_sr / wav.sample_rate));
    if (out_frames <= 0) {
        error = "Reference WAV contains no samples: " + path;
        return false;
    }

    std::vector<float> stereo(static_cast<size_t>(2 * out_frames), 0.0f);
    for (int64_t i = 0; i < out_frames; ++i) {
        const double src_pos = wav.sample_rate == target_sr
            ? static_cast<double>(i)
            : static_cast<double>(i) * wav.sample_rate / target_sr;
        const int64_t i0 = (std::min)(static_cast<int64_t>(std::floor(src_pos)), in_frames - 1);
        const int64_t i1 = (std::min)(i0 + 1, in_frames - 1);
        const float frac = static_cast<float>(src_pos - static_cast<double>(i0));
        for (int c = 0; c < 2; ++c) {
            const int src_c = wav.channels == 1 ? 0 : (std::min)(c, wav.channels - 1);
            const float a = wav.samples[static_cast<size_t>(i0 * wav.channels + src_c)];
            const float b = wav.samples[static_cast<size_t>(i1 * wav.channels + src_c)];
            stereo[static_cast<size_t>(c * out_frames + i)] = a + (b - a) * frac;
        }
    }

    try {
        if (!codec_encode_session_) {
            if (!load_session(codec_encode_path_, codec_encode_session_, error)) {
                return false;
            }
            cache_session_io(*codec_encode_session_, codec_encode_io_);
        }
        auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        if (codec_encode_io_.input_names.size() != 2 || codec_encode_io_.output_names.empty()) {
            error = "MOSS codec encode ONNX signature mismatch: expected 2 inputs and at least 1 output.";
            return false;
        }

        std::vector<int32_t> lengths = {static_cast<int32_t>(out_frames)};
        std::vector<int64_t> wav_shape = {1, 2, out_frames};
        std::vector<int64_t> len_shape = {1};
        std::vector<Ort::Value> inputs;
        inputs.emplace_back(Ort::Value::CreateTensor<float>(mem, stereo.data(), stereo.size(), wav_shape.data(), wav_shape.size()));
        inputs.emplace_back(Ort::Value::CreateTensor<int32_t>(mem, lengths.data(), lengths.size(), len_shape.data(), len_shape.size()));
        auto out = codec_encode_session_->Run(
            Ort::RunOptions{nullptr},
            codec_encode_io_.input_ptrs.data(),
            inputs.data(),
            inputs.size(),
            codec_encode_io_.output_ptrs.data(),
            codec_encode_io_.output_ptrs.size());
        const std::vector<int64_t> shape = tensor_shape(out[0]);
        if (shape.size() != 3) {
            error = "MOSS codec encode returned unexpected rank.";
            return false;
        }

        size_t count = 1;
        for (int64_t dim : shape) {
            count *= static_cast<size_t>(dim);
        }
        std::vector<int64_t> raw(count);
        const auto type = out[0].GetTensorTypeAndShapeInfo().GetElementType();
        if (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
            const int64_t* p = out[0].GetTensorData<int64_t>();
            raw.assign(p, p + count);
        } else if (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) {
            const int32_t* p = out[0].GetTensorData<int32_t>();
            for (size_t i = 0; i < count; ++i) {
                raw[i] = p[i];
            }
        } else {
            error = "MOSS codec encode returned non-integer codes.";
            return false;
        }

        if (shape[0] == 1 && shape[2] == config_.n_vq) {
            out_codes = std::move(raw);
        } else if (shape[0] == config_.n_vq && shape[1] == 1) {
            const int64_t frames = shape[2];
            out_codes.resize(static_cast<size_t>(frames * config_.n_vq));
            for (int ch = 0; ch < config_.n_vq; ++ch) {
                for (int64_t t = 0; t < frames; ++t) {
                    out_codes[static_cast<size_t>(t * config_.n_vq + ch)] =
                        raw[static_cast<size_t>(ch * frames + t)];
                }
            }
        } else if (shape[0] == 1 && shape[1] == config_.n_vq) {
            const int64_t frames = shape[2];
            out_codes.resize(static_cast<size_t>(frames * config_.n_vq));
            for (int ch = 0; ch < config_.n_vq; ++ch) {
                for (int64_t t = 0; t < frames; ++t) {
                    out_codes[static_cast<size_t>(t * config_.n_vq + ch)] =
                        raw[static_cast<size_t>(ch * frames + t)];
                }
            }
        } else {
            error = "MOSS codec encode returned unsupported code shape.";
            return false;
        }
        return !out_codes.empty();
    } catch (const std::exception& e) {
        error = std::string("MOSS codec encode failed: ") + e.what();
        return false;
    }
}

bool VieneuV3OnnxEngine::initialize(const VieneuV3OnnxInit& init, std::string& error) {
    initialized_ = false;
    env_.reset();
    prefill_session_.reset();
    decode_session_.reset();
    acoustic_session_.reset();
    codec_decode_session_.reset();
    codec_encode_session_.reset();
    session_options_.reset();
    prefill_io_ = SessionIo{};
    decode_io_ = SessionIo{};
    acoustic_io_ = SessionIo{};
    codec_decode_io_ = SessionIo{};
    codec_encode_io_ = SessionIo{};
    codec_encode_path_.clear();
    rng_.seed(std::random_device{}());

    if (init.model_dir.empty() && init.onnx_dir.empty()) {
        error = "VieNeu v3 requires model_dir or onnx_dir.";
        return false;
    }
    if (init.codec_dir.empty()) {
        error = "VieNeu v3 requires codec_dir with MOSS ONNX codec files.";
        return false;
    }
    if (!validate_assets(init, error)) {
        return false;
    }

    const std::string config_path = init.config_path.empty() ? join_path(model_dir_, "config.json") : init.config_path;
    const std::string tokenizer_path = init.tokenizer_path.empty() ? join_path(model_dir_, "tokenizer.json") : init.tokenizer_path;
    if (!load_config(config_path, error) ||
        !load_heads_npz(join_path(onnx_dir_, "vieneu_v3_heads.npz"), error) ||
        !tokenizer_.load(tokenizer_path, error)) {
        return false;
    }

    env_ = std::make_shared<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "VieneuV3Onnx");
    session_options_ = std::make_unique<Ort::SessionOptions>();
    if (init.n_threads > 0) {
        session_options_->SetIntraOpNumThreads(init.n_threads);
    }
    session_options_->SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    if (!load_session(join_path(onnx_dir_, "vieneu_prefill.onnx"), prefill_session_, error) ||
        !load_session(join_path(onnx_dir_, "vieneu_decode_step.onnx"), decode_session_, error) ||
        !load_session(join_path(onnx_dir_, "vieneu_acoustic_cached.onnx"), acoustic_session_, error) ||
        !load_session(join_path(codec_dir_, "moss_audio_tokenizer_decode_full.onnx"), codec_decode_session_, error)) {
        return false;
    }
    cache_session_io(*prefill_session_, prefill_io_);
    cache_session_io(*decode_session_, decode_io_);
    cache_session_io(*acoustic_session_, acoustic_io_);
    cache_session_io(*codec_decode_session_, codec_decode_io_);

    if (!load_voices(init.voices_json_path, error)) {
        return false;
    }

    initialized_ = true;
    return true;
}

VieneuV3OnnxEngine::PromptRows VieneuV3OnnxEngine::build_rows(
    const std::string& phonemes,
    const std::vector<int64_t>* ref_codes,
    int leading_token) const {
    const std::vector<int64_t> phone_ids = tokenizer_.encode(phonemes);
    const int64_t cols = config_.n_vq + 1;
    const int64_t text_rows = static_cast<int64_t>(phone_ids.size()) + 3;
    const int64_t ref_rows = ref_codes ? static_cast<int64_t>(ref_codes->size() / config_.n_vq) : 0;
    PromptRows rows;
    rows.rows = text_rows + ref_rows;
    rows.cols = cols;
    rows.data.assign(static_cast<size_t>(rows.rows * rows.cols), config_.audio_pad_token_id);
    rows.data[0] = leading_token;
    rows.data[cols] = config_.text_prompt_start_token_id;
    for (size_t i = 0; i < phone_ids.size(); ++i) {
        rows.data[static_cast<size_t>((static_cast<int64_t>(i) + 2) * cols)] = phone_ids[i];
    }
    rows.data[static_cast<size_t>((text_rows - 1) * cols)] = config_.text_prompt_end_token_id;
    if (ref_codes) {
        for (int64_t r = 0; r < ref_rows; ++r) {
            const int64_t dst_row = text_rows + r;
            rows.data[static_cast<size_t>(dst_row * cols)] = config_.audio_ref_slot_token_id;
            for (int ch = 0; ch < config_.n_vq; ++ch) {
                rows.data[static_cast<size_t>(dst_row * cols + ch + 1)] =
                    (*ref_codes)[static_cast<size_t>(r * config_.n_vq + ch)];
            }
        }
    }
    return rows;
}

std::vector<float> VieneuV3OnnxEngine::embed_rows(const PromptRows& rows) const {
    std::vector<float> embeds(static_cast<size_t>(rows.rows * config_.hidden_size), 0.0f);
    for (int64_t r = 0; r < rows.rows; ++r) {
        float* dst = embeds.data() + r * config_.hidden_size;
        const int64_t text_id = rows.data[static_cast<size_t>(r * rows.cols)];
        if (text_id >= 0 && text_id < text_emb_.rows) {
            const float* src = text_emb_.data.data() + text_id * text_emb_.cols;
            std::copy(src, src + config_.hidden_size, dst);
        }
        for (int ch = 0; ch < config_.n_vq; ++ch) {
            const int64_t id = rows.data[static_cast<size_t>(r * rows.cols + ch + 1)];
            if (id == config_.audio_pad_token_id || id < 0 || id >= audio_emb_.dim1) {
                continue;
            }
            const float* src = audio_emb_.data.data() +
                (static_cast<int64_t>(ch) * audio_emb_.dim1 + id) * audio_emb_.dim2;
            for (int h = 0; h < config_.hidden_size; ++h) {
                dst[h] += src[h];
            }
        }
    }
    return embeds;
}

int64_t VieneuV3OnnxEngine::sample_logits(
    std::vector<float> logits,
    float temperature,
    int top_k,
    float top_p,
    float repetition_penalty,
    const std::unordered_set<int>* previous) {
    if (previous && std::fabs(repetition_penalty - 1.0f) > 1e-6f) {
        for (int idx : *previous) {
            if (idx >= 0 && static_cast<size_t>(idx) < logits.size()) {
                logits[static_cast<size_t>(idx)] = logits[static_cast<size_t>(idx)] < 0.0f
                    ? logits[static_cast<size_t>(idx)] * repetition_penalty
                    : logits[static_cast<size_t>(idx)] / repetition_penalty;
            }
        }
    }
    if (!(temperature > 0.0f)) {
        return static_cast<int64_t>(std::distance(logits.begin(), std::max_element(logits.begin(), logits.end())));
    }
    for (float& v : logits) {
        v /= temperature;
    }
    if (top_k > 0 && static_cast<size_t>(top_k) < logits.size()) {
        std::vector<float> tmp = logits;
        std::nth_element(tmp.begin(), tmp.end() - top_k, tmp.end());
        const float kth = *(tmp.end() - top_k);
        for (float& v : logits) {
            if (v < kth) v = -std::numeric_limits<float>::infinity();
        }
    }
    if (top_p > 0.0f && top_p < 1.0f) {
        std::vector<size_t> order(logits.size());
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(), [&](size_t a, size_t b) { return logits[a] > logits[b]; });
        std::vector<float> sorted(order.size());
        for (size_t i = 0; i < order.size(); ++i) sorted[i] = logits[order[i]];
        const std::vector<float> probs = softmax(sorted);
        float cumulative_before = 0.0f;
        for (size_t i = 0; i < order.size(); ++i) {
            if (cumulative_before > top_p) {
                logits[order[i]] = -std::numeric_limits<float>::infinity();
            }
            cumulative_before += probs[i];
        }
    }
    const std::vector<float> probs = softmax(logits);
    std::discrete_distribution<int64_t> dist(probs.begin(), probs.end());
    return dist(rng_);
}

bool VieneuV3OnnxEngine::acoustic_frame(
    const std::vector<float>& h,
    float temperature,
    int top_k,
    float top_p,
    float repetition_penalty,
    std::vector<std::unordered_set<int>>& history,
    std::vector<int64_t>& codes,
    bool& eos,
    std::string& error) {
    try {
        const int H = config_.hidden_size;
        const int nH = config_.local_num_attention_heads;
        const int hd = H / nH;
        std::vector<float> token(static_cast<size_t>(2 * H));
        std::copy(h.begin(), h.begin() + H, token.begin());
        const float* sgs = text_emb_.data.data() + config_.speech_generation_start_token_id * text_emb_.cols;
        std::copy(sgs, sgs + H, token.begin() + H);
        std::vector<int64_t> pos = {0, 1};
        std::vector<float> empty;
        std::vector<int64_t> empty_shape = {1, nH, 0, hd};
        std::vector<int64_t> token_shape = {1, 2, H};
        std::vector<int64_t> pos_shape = {1, 2};

        auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        if (acoustic_io_.input_names.size() != 6 || acoustic_io_.output_names.size() != 5) {
            error = "VieNeu v3 acoustic ONNX signature mismatch: expected 6 inputs and 5 outputs.";
            return false;
        }
        std::vector<Ort::Value> inputs;
        inputs.emplace_back(Ort::Value::CreateTensor<float>(mem, token.data(), token.size(), token_shape.data(), token_shape.size()));
        inputs.emplace_back(Ort::Value::CreateTensor<int64_t>(mem, pos.data(), pos.size(), pos_shape.data(), pos_shape.size()));
        inputs.emplace_back(Ort::Value::CreateTensor<float>(mem, empty.data(), 0, empty_shape.data(), empty_shape.size()));
        inputs.emplace_back(Ort::Value::CreateTensor<float>(mem, empty.data(), 0, empty_shape.data(), empty_shape.size()));
        inputs.emplace_back(Ort::Value::CreateTensor<float>(mem, empty.data(), 0, empty_shape.data(), empty_shape.size()));
        inputs.emplace_back(Ort::Value::CreateTensor<float>(mem, empty.data(), 0, empty_shape.data(), empty_shape.size()));
        auto out = acoustic_session_->Run(
            Ort::RunOptions{nullptr},
            acoustic_io_.input_ptrs.data(),
            inputs.data(),
            inputs.size(),
            acoustic_io_.output_ptrs.data(),
            acoustic_io_.output_ptrs.size());
        TensorBlob hidden = copy_float_tensor(out[0]);
        TensorBlob pk0 = copy_float_tensor(out[1]);
        TensorBlob pk1 = copy_float_tensor(out[2]);
        TensorBlob pv0 = copy_float_tensor(out[3]);
        TensorBlob pv1 = copy_float_tensor(out[4]);
        std::vector<float> slot0(hidden.data.begin(), hidden.data.begin() + H);

        auto sample_channel = [&](int ch, const float* vec) {
            std::vector<float> logits(static_cast<size_t>(audio_emb_.dim1), 0.0f);
            for (int64_t v = 0; v < audio_emb_.dim1; ++v) {
                const float* emb = audio_emb_.data.data() + (static_cast<int64_t>(ch) * audio_emb_.dim1 + v) * audio_emb_.dim2;
                float sum = 0.0f;
                for (int i = 0; i < H; ++i) sum += vec[i] * emb[i];
                logits[static_cast<size_t>(v)] = sum;
            }
            std::unordered_set<int>* prev = history.empty() ? nullptr : &history[static_cast<size_t>(ch)];
            int64_t code = sample_logits(logits, temperature, top_k, top_p, repetition_penalty, prev);
            if (prev) prev->insert(static_cast<int>(code));
            return code;
        };

        codes.clear();
        codes.reserve(static_cast<size_t>(config_.n_vq));
        codes.push_back(sample_channel(0, hidden.data.data() + H));
        for (int ch = 1; ch < config_.n_vq; ++ch) {
            const float* emb = audio_emb_.data.data() +
                (static_cast<int64_t>(ch - 1) * audio_emb_.dim1 + codes.back()) * audio_emb_.dim2;
            std::vector<float> step_token(emb, emb + H);
            std::vector<int64_t> step_pos = {ch + 1};
            std::vector<int64_t> step_token_shape = {1, 1, H};
            std::vector<int64_t> step_pos_shape = {1, 1};
            std::vector<Ort::Value> step_inputs;
            step_inputs.emplace_back(Ort::Value::CreateTensor<float>(mem, step_token.data(), step_token.size(), step_token_shape.data(), step_token_shape.size()));
            step_inputs.emplace_back(Ort::Value::CreateTensor<int64_t>(mem, step_pos.data(), step_pos.size(), step_pos_shape.data(), step_pos_shape.size()));
            step_inputs.emplace_back(Ort::Value::CreateTensor<float>(mem, pk0.data.data(), pk0.data.size(), pk0.shape.data(), pk0.shape.size()));
            step_inputs.emplace_back(Ort::Value::CreateTensor<float>(mem, pk1.data.data(), pk1.data.size(), pk1.shape.data(), pk1.shape.size()));
            step_inputs.emplace_back(Ort::Value::CreateTensor<float>(mem, pv0.data.data(), pv0.data.size(), pv0.shape.data(), pv0.shape.size()));
            step_inputs.emplace_back(Ort::Value::CreateTensor<float>(mem, pv1.data.data(), pv1.data.size(), pv1.shape.data(), pv1.shape.size()));
            auto step_out = acoustic_session_->Run(
                Ort::RunOptions{nullptr},
                acoustic_io_.input_ptrs.data(),
                step_inputs.data(),
                step_inputs.size(),
                acoustic_io_.output_ptrs.data(),
                acoustic_io_.output_ptrs.size());
            hidden = copy_float_tensor(step_out[0]);
            pk0 = copy_float_tensor(step_out[1]);
            pk1 = copy_float_tensor(step_out[2]);
            pv0 = copy_float_tensor(step_out[3]);
            pv1 = copy_float_tensor(step_out[4]);
            codes.push_back(sample_channel(ch, hidden.data.data()));
        }

        std::vector<float> text_logits(static_cast<size_t>(text_emb_.rows), 0.0f);
        for (int64_t v = 0; v < text_emb_.rows; ++v) {
            const float* emb = text_emb_.data.data() + v * text_emb_.cols;
            float sum = 0.0f;
            for (int i = 0; i < H; ++i) sum += slot0[static_cast<size_t>(i)] * emb[i];
            text_logits[static_cast<size_t>(v)] = sum;
        }
        eos = static_cast<int>(std::distance(text_logits.begin(), std::max_element(text_logits.begin(), text_logits.end()))) ==
              config_.speech_generation_end_token_id;
        return true;
    } catch (const std::exception& e) {
        error = std::string("VieNeu v3 acoustic frame failed: ") + e.what();
        return false;
    }
}

bool VieneuV3OnnxEngine::decode_codes(const std::vector<int64_t>& frames, int64_t frame_count, std::vector<float>& out_audio, std::string& error) {
    try {
        std::vector<int32_t> codes(frames.size());
        for (size_t i = 0; i < frames.size(); ++i) codes[i] = static_cast<int32_t>(frames[i]);
        std::vector<int32_t> lengths = {static_cast<int32_t>(frame_count)};
        std::vector<int64_t> codes_shape = {1, frame_count, config_.n_vq};
        std::vector<int64_t> len_shape = {1};
        auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        if (codec_decode_io_.input_names.size() != 2 || codec_decode_io_.output_names.empty()) {
            error = "MOSS codec decode ONNX signature mismatch: expected 2 inputs and at least 1 output.";
            return false;
        }
        std::vector<Ort::Value> inputs;
        inputs.emplace_back(Ort::Value::CreateTensor<int32_t>(mem, codes.data(), codes.size(), codes_shape.data(), codes_shape.size()));
        inputs.emplace_back(Ort::Value::CreateTensor<int32_t>(mem, lengths.data(), lengths.size(), len_shape.data(), len_shape.size()));
        auto out = codec_decode_session_->Run(
            Ort::RunOptions{nullptr},
            codec_decode_io_.input_ptrs.data(),
            inputs.data(),
            inputs.size(),
            codec_decode_io_.output_ptrs.data(),
            codec_decode_io_.output_ptrs.size());
        TensorBlob audio = copy_float_tensor(out[0]);
        if (audio.shape.size() == 3 && audio.shape[0] == 1) {
            const int64_t channels = audio.shape[1];
            const int64_t samples = audio.shape[2];
            out_audio.assign(static_cast<size_t>(samples), 0.0f);
            for (int64_t c = 0; c < channels; ++c) {
                const float* src = audio.data.data() + c * samples;
                for (int64_t i = 0; i < samples; ++i) out_audio[static_cast<size_t>(i)] += src[i] / static_cast<float>(channels);
            }
        } else {
            out_audio = std::move(audio.data);
        }
        return true;
    } catch (const std::exception& e) {
        error = std::string("MOSS codec decode failed: ") + e.what();
        return false;
    }
}

std::string VieneuV3OnnxEngine::phonemize_for_v3(const std::string& text) const {
    return VieneuProfile::phonemize(text);
}

bool VieneuV3OnnxEngine::synthesize_phonemes(
    const std::string& phonemes,
    const std::vector<int64_t>* ref_codes,
    int leading_token,
    const VieneuV3OnnxParams& params,
    std::vector<float>& out_audio,
    std::string& error) {
    out_audio.clear();
    const PromptRows rows = build_rows(phonemes, ref_codes, leading_token);
    std::vector<float> prompt_embeds = embed_rows(rows);
    std::vector<int64_t> prompt_shape = {1, rows.rows, config_.hidden_size};
    auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    std::lock_guard<std::mutex> lock(run_mutex_);
    try {
        const size_t expected_lm_outputs = static_cast<size_t>(1 + config_.num_hidden_layers * 2);
        if (prefill_io_.input_names.size() != 1 || prefill_io_.output_names.size() != expected_lm_outputs) {
            error = "VieNeu v3 prefill ONNX signature mismatch.";
            return false;
        }
        Ort::Value prompt_tensor = Ort::Value::CreateTensor<float>(mem, prompt_embeds.data(), prompt_embeds.size(), prompt_shape.data(), prompt_shape.size());
        auto pre = prefill_session_->Run(
            Ort::RunOptions{nullptr},
            prefill_io_.input_ptrs.data(),
            &prompt_tensor,
            1,
            prefill_io_.output_ptrs.data(),
            prefill_io_.output_ptrs.size());
        TensorBlob hidden_blob = copy_float_tensor(pre[0]);
        std::vector<TensorBlob> past_k;
        std::vector<TensorBlob> past_v;
        past_k.reserve(static_cast<size_t>(config_.num_hidden_layers));
        past_v.reserve(static_cast<size_t>(config_.num_hidden_layers));
        for (int i = 0; i < config_.num_hidden_layers; ++i) past_k.push_back(copy_float_tensor(pre[1 + i]));
        for (int i = 0; i < config_.num_hidden_layers; ++i) past_v.push_back(copy_float_tensor(pre[1 + config_.num_hidden_layers + i]));

        std::vector<float> h(static_cast<size_t>(config_.hidden_size));
        const int64_t last_offset = (rows.rows - 1) * config_.hidden_size;
        std::copy(hidden_blob.data.begin() + last_offset, hidden_blob.data.begin() + last_offset + config_.hidden_size, h.begin());

        const size_t expected_decode_inputs = static_cast<size_t>(2 + config_.num_hidden_layers * 2);
        if (decode_io_.input_names.size() != expected_decode_inputs || decode_io_.output_names.size() != expected_lm_outputs) {
            error = "VieNeu v3 decode-step ONNX signature mismatch.";
            return false;
        }

        std::vector<std::unordered_set<int>> history;
        if (std::fabs(params.repetition_penalty - 1.0f) > 1e-6f) {
            history.resize(static_cast<size_t>(config_.n_vq));
        }
        std::vector<int64_t> frames;
        const int max_frames = (std::max)(1, params.max_new_frames);
        frames.reserve(static_cast<size_t>(max_frames * config_.n_vq));
        for (int t = 0; t < max_frames; ++t) {
            std::vector<int64_t> codes;
            bool eos = false;
            if (!acoustic_frame(h, params.temperature, params.top_k, params.top_p, params.repetition_penalty, history, codes, eos, error)) {
                return false;
            }
            frames.insert(frames.end(), codes.begin(), codes.end());
            if (eos) {
                break;
            }

            PromptRows slot;
            slot.rows = 1;
            slot.cols = config_.n_vq + 1;
            slot.data.assign(static_cast<size_t>(slot.cols), config_.audio_pad_token_id);
            slot.data[0] = config_.speech_generation_start_token_id;
            for (int ch = 0; ch < config_.n_vq; ++ch) slot.data[static_cast<size_t>(ch + 1)] = codes[static_cast<size_t>(ch)];
            std::vector<float> se = embed_rows(slot);
            std::vector<int64_t> se_shape = {1, 1, config_.hidden_size};
            std::vector<int64_t> pos = {rows.rows + t};
            std::vector<int64_t> pos_shape = {1, 1};

            std::vector<Ort::Value> inputs;
            inputs.reserve(static_cast<size_t>(expected_decode_inputs));
            inputs.emplace_back(Ort::Value::CreateTensor<float>(mem, se.data(), se.size(), se_shape.data(), se_shape.size()));
            inputs.emplace_back(Ort::Value::CreateTensor<int64_t>(mem, pos.data(), pos.size(), pos_shape.data(), pos_shape.size()));
            for (auto& pk : past_k) inputs.emplace_back(Ort::Value::CreateTensor<float>(mem, pk.data.data(), pk.data.size(), pk.shape.data(), pk.shape.size()));
            for (auto& pv : past_v) inputs.emplace_back(Ort::Value::CreateTensor<float>(mem, pv.data.data(), pv.data.size(), pv.shape.data(), pv.shape.size()));
            auto dec = decode_session_->Run(
                Ort::RunOptions{nullptr},
                decode_io_.input_ptrs.data(),
                inputs.data(),
                inputs.size(),
                decode_io_.output_ptrs.data(),
                decode_io_.output_ptrs.size());
            TensorBlob dec_hidden = copy_float_tensor(dec[0]);
            std::copy(dec_hidden.data.begin(), dec_hidden.data.begin() + config_.hidden_size, h.begin());
            for (int i = 0; i < config_.num_hidden_layers; ++i) past_k[static_cast<size_t>(i)] = copy_float_tensor(dec[1 + i]);
            for (int i = 0; i < config_.num_hidden_layers; ++i) past_v[static_cast<size_t>(i)] = copy_float_tensor(dec[1 + config_.num_hidden_layers + i]);
        }

        if (frames.empty()) {
            error = "VieNeu v3 synthesis produced no acoustic frames.";
            return false;
        }
        return decode_codes(frames, static_cast<int64_t>(frames.size() / config_.n_vq), out_audio, error);
    } catch (const std::exception& e) {
        error = std::string("VieNeu v3 synthesis failed: ") + e.what();
        return false;
    }
}

bool VieneuV3OnnxEngine::synthesize(const VieneuV3OnnxParams& params, std::vector<float>& out_audio, std::string& error) {
    out_audio.clear();
    if (!initialized_) {
        error = "VieNeu v3 ONNX engine is not initialized.";
        return false;
    }
    if (params.text.empty()) {
        error = "VieNeu v3 synthesis requires non-empty text.";
        return false;
    }

    int leading_token = config_.emotion_0_token_id;
    std::vector<int64_t> ref_codes;
    if (!params.ref_audio_path.empty()) {
        if (!encode_reference_audio(params.ref_audio_path, ref_codes, error)) {
            return false;
        }
    } else {
        VoicePreset preset;
        if (!resolve_voice_preset(params.voice_id, preset, error)) {
            return false;
        }
        if (preset.has_reserved_id) {
            leading_token = preset.reserved_id;
        }
        if (!preset.codes.empty()) {
            ref_codes = std::move(preset.codes);
        }
    }

    if (!ref_codes.empty() && ref_codes.size() % static_cast<size_t>(config_.n_vq) != 0) {
        error = "VieNeu v3 reference codes are not divisible by n_vq.";
        return false;
    }

    const std::string phonemes = phonemize_for_v3(params.text);
    return synthesize_phonemes(
        phonemes,
        ref_codes.empty() ? nullptr : &ref_codes,
        leading_token,
        params,
        out_audio,
        error);
}
