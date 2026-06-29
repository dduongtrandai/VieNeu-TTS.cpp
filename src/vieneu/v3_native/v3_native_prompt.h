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

    V3PromptRows build_rows(const std::string& phonemes, const std::vector<int64_t>* ref_codes, int leading_token) const;
    std::vector<float> embed_rows(const V3PromptRows& rows) const;
    
    // Embed a single slot/frame of generated audio codes to pass back to the backbone
    std::vector<float> embed_slot(const std::vector<int64_t>& codes) const;

private:
    V3NativeConfig config_;
    const V3NativeTokenizer& tokenizer_;
    const V3NativeAssets& assets_;
};

// Simple text chunking utility matching normalizer
std::vector<std::string> chunk_text_v3(const std::string& text, int max_chars);

#endif // V3_NATIVE_PROMPT_H
