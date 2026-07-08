#ifndef V3_NATIVE_PROMPT_H
#define V3_NATIVE_PROMPT_H

#include <vector>
#include <string>
#include "v3_native_config.h"
#include "v3_native_tokenizer.h"
#include "v3_native_assets.h"

struct V3PromptRows {
    int64_t rows = 0;
    int64_t cols = 0;
    std::vector<int64_t> data;
};

class V3NativePrompt {
public:
    V3NativePrompt(const V3NativeConfig& config, const V3NativeTokenizer& tokenizer, const V3NativeAssets& assets);

    V3PromptRows build_rows(const std::string& phonemes, const std::vector<int64_t>* ref_codes, int style_token_id) const;
    std::vector<float> embed_rows(const V3PromptRows& rows, const std::vector<float>* speaker_anchor = nullptr) const;
    void embed_rows_into(const V3PromptRows& rows, const std::vector<float>* speaker_anchor, std::vector<float>& embeds) const;

    std::vector<float> embed_slot(const std::vector<int64_t>& codes, const std::vector<float>* speaker_anchor = nullptr) const;
    void embed_slot_into(const std::vector<int64_t>& codes, const std::vector<float>* speaker_anchor, std::vector<float>& embeds) const;

private:
    V3NativeConfig config_;
    const V3NativeTokenizer& tokenizer_;
    const V3NativeAssets& assets_;
};

std::vector<std::string> chunk_text_v3(const std::string& text, int max_chars);

#endif // V3_NATIVE_PROMPT_H
