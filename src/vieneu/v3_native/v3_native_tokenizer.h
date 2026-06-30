#ifndef V3_NATIVE_TOKENIZER_H
#define V3_NATIVE_TOKENIZER_H

#include <string>
#include <vector>
#include <unordered_map>

class V3NativeTokenizer {
public:
    V3NativeTokenizer() = default;

    bool load(const std::string& path, std::string& error);
    std::vector<int64_t> encode(const std::string& text) const;

    int64_t unk_id() const { return unk_id_; }

private:
    std::vector<int64_t> encode_ordinary(const std::string& text) const;

    std::unordered_map<std::string, int64_t> vocab_;
    std::unordered_map<std::string, int> merge_ranks_;
    std::vector<std::string> special_tokens_;
    int64_t unk_id_ = 43;
};

#endif // V3_NATIVE_TOKENIZER_H
