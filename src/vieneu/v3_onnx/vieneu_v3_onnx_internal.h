#ifndef VIENEU_V3_ONNX_INTERNAL_H
#define VIENEU_V3_ONNX_INTERNAL_H

#include <cstdint>
#include <string>
#include <vector>
#include <array>
#include "onnxruntime_cxx_api.h"

// Shared Structures
struct NamedArray {
    std::vector<int64_t> shape;
    std::vector<float> data;
};

struct TensorBlob {
    std::vector<int64_t> shape;
    std::vector<float> data;
};

// Endian readers
uint16_t read_u16_le(const uint8_t* p);
uint32_t read_u32_le(const uint8_t* p);
uint64_t read_u64_le(const uint8_t* p);
int16_t read_i16_le(const uint8_t* p);
int32_t read_i24_le(const uint8_t* p);
int32_t read_i32_le(const uint8_t* p);

// Float conversion
float half_to_float(uint16_t h);

// File helpers
std::string read_file_bytes(const std::string& path);

// ONNX utilities
std::vector<int64_t> tensor_shape(const Ort::Value& value);
TensorBlob copy_float_tensor(const Ort::Value& value);
void copy_float_tensor_into(const Ort::Value& value, TensorBlob& blob);
std::vector<std::string> session_input_names(Ort::Session& session);
std::vector<std::string> session_output_names(Ort::Session& session);
std::vector<const char*> name_ptrs(const std::vector<std::string>& names);

// Text splitting utilities
bool is_space_char(char c);
std::string trim_copy(const std::string& value);
size_t utf8_codepoint_count(const std::string& value);
bool can_append_chunk(const std::string& current, const std::string& item, size_t max_chars);
void append_joined(std::string& current, const std::string& item);
std::vector<std::string> split_words(const std::string& text);
std::vector<std::string> split_long_text_part(const std::string& text, size_t max_chars);
std::vector<std::string> split_text_for_v3_chunks(const std::string& text, int max_chars_value);

// Tokenizer helpers
std::string utf8_from_codepoint(uint32_t cp);
std::array<std::string, 256> build_byte_encoder();
std::vector<std::string> byte_level_symbols(const std::string& text);
std::string join_pair_key(const std::string& a, const std::string& b);

// Sampling and Math helpers
void matvec_transposed(const float* vec, const float* matrix_hv, int64_t hidden, int64_t vocab, std::vector<float>& logits);
std::vector<float> softmax(const std::vector<float>& logits);

#endif // VIENEU_V3_ONNX_INTERNAL_H
