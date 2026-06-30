#include "v3_native_prompt.h"
#include <algorithm>
#include <cctype>
#include <cstring>
#include <sstream>

static bool is_space_char(char c) {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r';
}

static std::string trim_copy(const std::string& value) {
    if (value.empty()) return "";
    size_t start = 0;
    while (start < value.size() && is_space_char(value[start])) {
        start++;
    }
    if (start == value.size()) return "";
    size_t end = value.size() - 1;
    while (end > start && is_space_char(value[end])) {
        end--;
    }
    return value.substr(start, end - start + 1);
}

static size_t utf8_codepoint_count(const std::string& value) {
    size_t count = 0;
    for (size_t i = 0; i < value.size(); ) {
        unsigned char c = value[i];
        if (c < 0x80) i += 1;
        else if ((c & 0xE0) == 0xC0) i += 2;
        else if ((c & 0xF0) == 0xE0) i += 3;
        else if ((c & 0xF8) == 0xF0) i += 4;
        else i += 1;
        count++;
    }
    return count;
}

static bool can_append_chunk(const std::string& current, const std::string& item, size_t max_chars) {
    if (current.empty()) {
        return utf8_codepoint_count(item) <= max_chars;
    }
    return utf8_codepoint_count(current) + 1 + utf8_codepoint_count(item) <= max_chars;
}

static void append_joined(std::string& current, const std::string& item) {
    if (current.empty()) {
        current = item;
    } else {
        current += ' ';
        current += item;
    }
}

static std::vector<std::string> split_words(const std::string& text) {
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

static std::vector<std::string> split_long_text_part(const std::string& text, size_t max_chars) {
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

static std::string normalize_emotion_key(std::string inner) {
    while (!inner.empty() && is_space_char(inner.front())) {
        inner.erase(inner.begin());
    }
    while (!inner.empty() && is_space_char(inner.back())) {
        inner.pop_back();
    }
    std::transform(inner.begin(), inner.end(), inner.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return inner;
}

static bool emotion_token_for_key(const std::string& key, std::string& token) {
    if (key == "chuckle" || key == "cuoi" || key == u8"cười" ||
        key == "sigh" || key == "tho dai" || key == u8"thở dài" ||
        key == "clear throat" || key == "hang giong" || key == u8"hắng giọng") {
        if (key == "chuckle" || key == "cuoi" || key == u8"cười") {
            token = "<|emotion_1|>";
        } else if (key == "sigh" || key == "tho dai" || key == u8"thở dài") {
            token = "<|emotion_2|>";
        } else {
            token = "<|emotion_3|>";
        }
        return true;
    }
    return false;
}

static std::string canonicalize_emotion_markers(const std::string& text) {
    std::string out;
    out.reserve(text.size());
    for (size_t i = 0; i < text.size();) {
        if (text.compare(i, 10, "<|emotion_") == 0) {
            const size_t end = text.find("|>", i + 10);
            if (end != std::string::npos) {
                out.append(text, i, end + 2 - i);
                i = end + 2;
                continue;
            }
        }
        if (text[i] == '[') {
            const size_t end = text.find(']', i + 1);
            if (end != std::string::npos) {
                std::string token;
                const std::string key = normalize_emotion_key(text.substr(i + 1, end - i - 1));
                if (emotion_token_for_key(key, token)) {
                    out += token;
                    i = end + 1;
                    continue;
                }
            }
        }
        out.push_back(text[i]);
        ++i;
    }
    return out;
}

std::vector<std::string> chunk_text_v3(const std::string& text, int max_chars_value) {
    const size_t max_chars = static_cast<size_t>((std::max)(16, max_chars_value));
    std::vector<std::string> chunks;
    std::string current;
    std::string sentence;
    const std::string canonical_text = canonicalize_emotion_markers(text);

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

    for (char c : canonical_text) {
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

V3NativePrompt::V3NativePrompt(const V3NativeConfig& config, const V3NativeTokenizer& tokenizer, const V3NativeAssets& assets)
    : config_(config), tokenizer_(tokenizer), assets_(assets) {}

V3PromptRows V3NativePrompt::build_rows(
    const std::string& phonemes,
    const std::vector<int64_t>* ref_codes,
    int leading_token) const {
    const std::vector<int64_t> phone_ids = tokenizer_.encode(phonemes);
    const int64_t cols = config_.n_vq + 1;
    const int64_t text_rows = static_cast<int64_t>(phone_ids.size()) + 3;
    const int64_t ref_rows = ref_codes ? static_cast<int64_t>(ref_codes->size() / config_.n_vq) : 0;
    
    V3PromptRows rows;
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

std::vector<float> V3NativePrompt::embed_rows(const V3PromptRows& rows) const {
    std::vector<float> embeds(static_cast<size_t>(rows.rows * config_.hidden_size), 0.0f);
    embed_rows_into(rows, embeds);
    return embeds;
}

void V3NativePrompt::embed_rows_into(const V3PromptRows& rows, std::vector<float>& embeds) const {
    embeds.resize(static_cast<size_t>(rows.rows * config_.hidden_size));
    std::fill(embeds.begin(), embeds.end(), 0.0f);
    const auto& text_emb = assets_.text_emb();
    const auto& audio_emb = assets_.audio_emb();
    const int H = config_.hidden_size;

    for (int64_t r = 0; r < rows.rows; ++r) {
        float* dst = embeds.data() + r * H;
        const int64_t text_id = rows.data[static_cast<size_t>(r * rows.cols)];
        if (text_id >= 0 && text_id < config_.text_vocab_size) {
            const float* src = text_emb.data() + text_id * H;
            std::copy(src, src + H, dst);
        }
        for (int ch = 0; ch < config_.n_vq; ++ch) {
            const int64_t id = rows.data[static_cast<size_t>(r * rows.cols + ch + 1)];
            if (id == config_.audio_pad_token_id || id < 0 || id >= config_.audio_vocab_size) {
                continue;
            }
            const float* src = audio_emb.data() + (static_cast<int64_t>(ch) * config_.audio_vocab_size + id) * H;
            for (int h = 0; h < H; ++h) {
                dst[h] += src[h];
            }
        }
    }
}

std::vector<float> V3NativePrompt::embed_slot(const std::vector<int64_t>& codes) const {
    std::vector<float> embeds;
    embed_slot_into(codes, embeds);
    return embeds;
}

void V3NativePrompt::embed_slot_into(const std::vector<int64_t>& codes, std::vector<float>& embeds) const {
    embeds.resize(static_cast<size_t>(config_.hidden_size));
    std::fill(embeds.begin(), embeds.end(), 0.0f);
    const auto& audio_emb = assets_.audio_emb();
    const int H = config_.hidden_size;
    for (int ch = 0; ch < config_.n_vq; ++ch) {
        const int64_t id = codes[static_cast<size_t>(ch)];
        if (id == config_.audio_pad_token_id || id < 0 || id >= config_.audio_vocab_size) {
            continue;
        }
        const float* src = audio_emb.data() + (static_cast<int64_t>(ch) * config_.audio_vocab_size + id) * H;
        for (int h = 0; h < H; ++h) {
            embeds[h] += src[h];
        }
    }
}
