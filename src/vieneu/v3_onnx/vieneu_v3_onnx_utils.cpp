#include "../vieneu_v3_onnx.h"
#include "vieneu_v3_onnx_internal.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>

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

// --- Endian Readers ---

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

// --- Float Conversion ---

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

// --- File Helpers ---

std::string read_file_bytes(const std::string& path) {
    std::ifstream fs(path, std::ios::binary);
    if (!fs.is_open()) {
        throw std::runtime_error("failed to open file: " + path);
    }
    std::ostringstream ss;
    ss << fs.rdbuf();
    return ss.str();
}

// --- ONNX Utilities ---

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

void copy_float_tensor_into(const Ort::Value& value, TensorBlob& blob) {
    blob.shape = tensor_shape(value);
    size_t count = 1;
    for (int64_t dim : blob.shape) {
        count *= static_cast<size_t>(dim);
    }
    blob.data.resize(count);
    const float* src = value.GetTensorData<float>();
    std::copy(src, src + count, blob.data.begin());
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

// --- Text Splitting Utilities ---

bool is_space_char(char c) {
    return c == ' ' || c == '\t' || c == '\r' || c == '\n';
}

std::string trim_copy(const std::string& value) {
    size_t begin = 0;
    while (begin < value.size() && is_space_char(value[begin])) {
        ++begin;
    }
    size_t end = value.size();
    while (end > begin && is_space_char(value[end - 1])) {
        --end;
    }
    return value.substr(begin, end - begin);
}

size_t utf8_codepoint_count(const std::string& value) {
    size_t count = 0;
    for (unsigned char c : value) {
        if ((c & 0xC0u) != 0x80u) {
            ++count;
        }
    }
    return count;
}

bool can_append_chunk(const std::string& current, const std::string& item, size_t max_chars) {
    if (current.empty()) {
        return utf8_codepoint_count(item) <= max_chars;
    }
    return utf8_codepoint_count(current) + 1 + utf8_codepoint_count(item) <= max_chars;
}

void append_joined(std::string& current, const std::string& item) {
    if (current.empty()) {
        current = item;
    } else {
        current += ' ';
        current += item;
    }
}

std::vector<std::string> split_words(const std::string& text) {
    std::vector<std::string> words;
    std::string cur;
    for (char c : text) {
        if (is_space_char(c)) {
            if (!cur.empty()) {
                words.push_back(cur);
                cur.clear();
            }
        } else {
            cur.push_back(c);
        }
    }
    if (!cur.empty()) {
        words.push_back(cur);
    }
    return words;
}

std::vector<std::string> split_long_text_part(const std::string& text, size_t max_chars) {
    std::vector<std::string> chunks;
    std::string current;
    for (const std::string& word : split_words(text)) {
        if (can_append_chunk(current, word, max_chars)) {
            append_joined(current, word);
        } else {
            if (!current.empty()) {
                chunks.push_back(current);
            }
            current = word;
        }
    }
    if (!current.empty()) {
        chunks.push_back(current);
    }
    return chunks;
}

std::vector<std::string> split_text_for_v3_chunks(const std::string& text, int max_chars_value) {
    const size_t max_chars = static_cast<size_t>((std::max)(16, max_chars_value));
    std::vector<std::string> chunks;
    std::string current;
    std::string sentence;

    auto flush_current = [&]() {
        std::string item = trim_copy(current);
        if (!item.empty()) {
            chunks.push_back(item);
        }
        current.clear();
    };

    auto add_part = [&](const std::string& raw) {
        const std::string part = trim_copy(raw);
        if (part.empty()) {
            return;
        }
        if (utf8_codepoint_count(part) > max_chars) {
            flush_current();
            std::string sub;
            for (char c : part) {
                sub.push_back(c);
                if (c == ',' || c == ';' || c == ':') {
                    const std::string minor = trim_copy(sub);
                    if (!minor.empty()) {
                        if (utf8_codepoint_count(minor) > max_chars) {
                            const auto word_chunks = split_long_text_part(minor, max_chars);
                            chunks.insert(chunks.end(), word_chunks.begin(), word_chunks.end());
                        } else {
                            chunks.push_back(minor);
                        }
                    }
                    sub.clear();
                }
            }
            const std::string tail = trim_copy(sub);
            if (!tail.empty()) {
                if (utf8_codepoint_count(tail) > max_chars) {
                    const auto word_chunks = split_long_text_part(tail, max_chars);
                    chunks.insert(chunks.end(), word_chunks.begin(), word_chunks.end());
                } else {
                    chunks.push_back(tail);
                }
            }
            return;
        }
        if (!can_append_chunk(current, part, max_chars)) {
            flush_current();
        }
        append_joined(current, part);
    };

    for (char c : text) {
        sentence.push_back(c);
        if (c == '.' || c == '!' || c == '?' || c == '\n' || c == '\r') {
            add_part(sentence);
            sentence.clear();
        }
    }
    add_part(sentence);
    flush_current();
    return chunks;
}

// --- VieneuV3OnnxEngine Utils Member Functions ---

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
