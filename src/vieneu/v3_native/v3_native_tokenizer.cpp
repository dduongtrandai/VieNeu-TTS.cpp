#include "v3_native_tokenizer.h"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <array>
#include <limits>
#include <nlohmann/json.hpp>

static std::string local_utf8_from_codepoint(uint32_t cp) {
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

static std::array<std::string, 256> build_byte_encoder() {
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
        enc[static_cast<size_t>(bs[i])] = local_utf8_from_codepoint(static_cast<uint32_t>(cs[i]));
    }
    return enc;
}

static std::vector<std::string> byte_level_symbols(const std::string& text) {
    static const auto enc = build_byte_encoder();
    std::vector<std::string> out;
    out.reserve(text.size());
    for (unsigned char c : text) {
        out.push_back(enc[static_cast<size_t>(c)]);
    }
    return out;
}

static std::string join_pair_key(const std::string& a, const std::string& b) {
    return a + "\n" + b;
}

bool V3NativeTokenizer::load(const std::string& path, std::string& error) {
    try {
        std::ifstream fs(path, std::ios::binary);
        if (!fs.is_open()) {
            error = "Failed to open tokenizer.json at " + path;
            return false;
        }
        nlohmann::json doc = nlohmann::json::parse(fs);
        const auto& model = doc.at("model");
        vocab_.clear();
        merge_ranks_.clear();
        for (auto it = model.at("vocab").begin(); it != model.at("vocab").end(); ++it) {
            vocab_[it.key()] = it.value().get<int64_t>();
        }
        if (model.contains("unk_token")) {
            const std::string unk = model.at("unk_token").get<std::string>();
            auto it = vocab_.find(unk);
            if (it != vocab_.end()) {
                unk_id_ = it->second;
            }
        }
        int rank = 0;
        for (const auto& merge : model.at("merges")) {
            if (merge.is_array() && merge.size() == 2) {
                merge_ranks_[join_pair_key(merge[0].get<std::string>(), merge[1].get<std::string>())] = rank++;
            } else if (merge.is_string()) {
                std::istringstream ss(merge.get<std::string>());
                std::string a, b;
                if (ss >> a >> b) {
                    merge_ranks_[join_pair_key(a, b)] = rank++;
                }
            }
        }
        return true;
    } catch (const std::exception& e) {
        error = std::string("Failed to load tokenizer: ") + e.what();
        return false;
    }
}

std::vector<int64_t> V3NativeTokenizer::encode(const std::string& text) const {
    std::vector<std::string> word = byte_level_symbols(text);
    if (word.empty()) {
        return {};
    }
    while (word.size() > 1) {
        int best_rank = (std::numeric_limits<int>::max)();
        size_t best_idx = static_cast<size_t>(-1);
        for (size_t i = 0; i + 1 < word.size(); ++i) {
            auto it = merge_ranks_.find(join_pair_key(word[i], word[i + 1]));
            if (it != merge_ranks_.end() && it->second < best_rank) {
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
        auto it = vocab_.find(token);
        ids.push_back(it == vocab_.end() ? unk_id_ : it->second);
    }
    return ids;
}
