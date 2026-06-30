#include "vieneu.h"
#include <regex>
#include <iostream>
#include <map>
#include <algorithm>
#include <sstream>
#include <cmath>
#include <cctype>
#ifdef VIENEU_USE_SEA_G2P
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <mutex>
#include "sea_g2p.h"
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#elif defined(__APPLE__)
#include <mach-o/dyld.h>
#else
#include <unistd.h>
#endif
#endif

// UTF-8 Helper Functions
static std::vector<uint32_t> utf8_to_cps(const std::string& str) {
    std::vector<uint32_t> res;
    size_t i = 0;
    while (i < str.size()) {
        uint8_t c = str[i];
        uint32_t cp = 0;
        int len = 0;
        if (c < 0x80) { cp = c; len = 1; }
        else if ((c & 0xE0) == 0xC0) { cp = c & 0x1F; len = 2; }
        else if ((c & 0xF0) == 0xE0) { cp = c & 0x0F; len = 3; }
        else if ((c & 0xF8) == 0xF0) { cp = c & 0x07; len = 4; }
        else { i++; continue; }

        if (i + len > str.size()) break;
        for (int j = 1; j < len; ++j) {
            cp = (cp << 6) | (str[i + j] & 0x3F);
        }
        res.push_back(cp);
        i += len;
    }
    return res;
}

static std::string cps_to_utf8(const std::vector<uint32_t>& cps) {
    std::string res;
    for (auto cp : cps) {
        if (cp < 0x80) {
            res.push_back((char)cp);
        } else if (cp < 0x800) {
            res.push_back((char)(0xC0 | (cp >> 6)));
            res.push_back((char)(0x80 | (cp & 0x3F)));
        } else if (cp < 0x10000) {
            res.push_back((char)(0xE0 | (cp >> 12)));
            res.push_back((char)(0x80 | ((cp >> 6) & 0x3F)));
            res.push_back((char)(0x80 | (cp & 0x3F)));
        } else {
            res.push_back((char)(0xF0 | (cp >> 18)));
            res.push_back((char)(0x80 | ((cp >> 12) & 0x3F)));
            res.push_back((char)(0x80 | ((cp >> 6) & 0x3F)));
            res.push_back((char)(0x80 | (cp & 0x3F)));
        }
    }
    return res;
}

struct VowelMap {
    uint32_t base;
    int tone;
};

static VowelMap get_vietnamese_vowel_base(uint32_t cp) {
    switch (cp) {
        // Grave (2)
        case 224: return {97, 2};   // à -> a
        case 7857: return {259, 2}; // ằ -> ă
        case 7847: return {226, 2}; // ầ -> â
        case 232: return {101, 2};  // è -> e
        case 7873: return {234, 2}; // ề -> ê
        case 236: return {105, 2};  // ì -> i
        case 242: return {111, 2};  // ò -> o
        case 7891: return {244, 2}; // ồ -> ô
        case 7901: return {417, 2}; // ờ -> ơ
        case 249: return {117, 2};  // ù -> u
        case 7915: return {432, 2}; // ừ -> ư
        case 7923: return {121, 2}; // ỳ -> y

        // Acute (3)
        case 225: return {97, 3};   // á -> a
        case 7855: return {259, 3}; // ắ -> ă
        case 7845: return {226, 3}; // ấ -> â
        case 233: return {101, 3};  // é -> e
        case 7871: return {234, 3}; // ế -> ê
        case 237: return {105, 3};  // í -> i
        case 243: return {111, 3};  // ó -> o
        case 7889: return {244, 3}; // ố -> ô
        case 7899: return {417, 3}; // ớ -> ơ
        case 250: return {117, 3};  // ú -> u
        case 7913: return {432, 3}; // ứ -> ư
        case 253: return {121, 3};  // ý -> y

        // Hỏi (4)
        case 7843: return {97, 4};   // ả -> a
        case 7859: return {259, 4}; // ẳ -> ă
        case 7849: return {226, 4}; // ẩ -> â
        case 7867: return {101, 4};  // ẻ -> e
        case 7875: return {234, 4}; // ể -> ê
        case 7881: return {105, 4};  // ỉ -> i
        case 7887: return {111, 4};  // ỏ -> o
        case 7893: return {244, 4}; // ổ -> ô
        case 7903: return {417, 4}; // ở -> ơ
        case 7911: return {117, 4};  // ủ -> u
        case 7917: return {432, 4}; // ử -> ư
        case 7927: return {121, 4}; // ỷ -> y

        // Tilde (5)
        case 227: return {97, 5};   // ã -> a
        case 7861: return {259, 5}; // ẵ -> ă
        case 7851: return {226, 5}; // ẫ -> â
        case 7869: return {101, 5};  // ẽ -> e
        case 7877: return {234, 5}; // ễ -> ê
        case 297: return {105, 5};  // ĩ -> i
        case 245: return {111, 5};  // õ -> o
        case 7895: return {244, 5}; // ỗ -> ô
        case 7905: return {417, 5}; // ỡ -> ơ
        case 361: return {117, 5};  // ũ -> u
        case 7919: return {432, 5}; // ữ -> ư
        case 7929: return {121, 5}; // ỹ -> y

        // Underdot (6)
        case 7841: return {97, 6};   // ạ -> a
        case 7863: return {259, 6}; // ặ -> ă
        case 7853: return {226, 6}; // ậ -> â
        case 7865: return {101, 6};  // ẹ -> e
        case 7879: return {234, 6}; // ệ -> ê
        case 7883: return {105, 6};  // ị -> i
        case 7885: return {111, 6};  // ọ -> o
        case 7897: return {244, 6}; // ộ -> ô
        case 7907: return {417, 6}; // ợ -> ơ
        case 7909: return {117, 6};  // ụ -> u
        case 7921: return {432, 6}; // ự -> ư
        case 7925: return {121, 6}; // ỵ -> y
    }
    return {cp, 1};
}

static bool has_vietnamese_signal(const std::string& text) {
    for (uint32_t cp : utf8_to_cps(text)) {
        if (cp == 259 || cp == 226 || cp == 234 || cp == 244 || cp == 417 || cp == 432 ||
            cp == 273 || cp == 258 || cp == 194 || cp == 202 || cp == 212 || cp == 416 ||
            cp == 431 || cp == 272 || get_vietnamese_vowel_base(cp).tone > 1) {
            return true;
        }
    }
    return false;
}

#ifdef VIENEU_USE_SEA_G2P
static bool consume_emotion_marker(const std::string& text, size_t pos, std::string& token, size_t& consumed);
static void append_emotion_token(std::stringstream& ss, const std::string& token);

static std::string vieneu_getenv_string(const char* name) {
    const char* value = std::getenv(name);
    return value ? std::string(value) : std::string();
}

static bool vieneu_file_exists(const std::string& path) {
    if (path.empty()) {
        return false;
    }
    std::ifstream fs(path, std::ios::binary);
    return fs.good();
}

static std::string vieneu_dirname(std::string path) {
    const size_t pos = path.find_last_of("/\\");
    if (pos == std::string::npos) {
        return {};
    }
    return path.substr(0, pos);
}

static std::string vieneu_join_path(const std::string& a, const std::string& b) {
    if (a.empty()) {
        return b;
    }
    const char last = a[a.size() - 1];
    if (last == '/' || last == '\\') {
        return a + b;
    }
#ifdef _WIN32
    return a + "\\" + b;
#else
    return a + "/" + b;
#endif
}

static std::string vieneu_executable_dir() {
#ifdef _WIN32
    char path[MAX_PATH + 1] = {};
    const DWORD len = GetModuleFileNameA(nullptr, path, MAX_PATH);
    if (len == 0 || len >= MAX_PATH) {
        return {};
    }
    return vieneu_dirname(path);
#elif defined(__APPLE__)
    uint32_t size = 0;
    _NSGetExecutablePath(nullptr, &size);
    std::string path(size, '\0');
    if (_NSGetExecutablePath(&path[0], &size) != 0) {
        return {};
    }
    path.resize(std::strlen(path.c_str()));
    return vieneu_dirname(path);
#else
    char path[4096] = {};
    const ssize_t len = readlink("/proc/self/exe", path, sizeof(path) - 1);
    if (len <= 0) {
        return {};
    }
    path[len] = '\0';
    return vieneu_dirname(path);
#endif
}

static std::string resolve_sea_g2p_dict_path() {
    std::vector<std::string> candidates;
    const std::string env_path = vieneu_getenv_string("VIENEU_SEA_G2P_DICT");
    if (!env_path.empty()) {
        candidates.push_back(env_path);
    }

    const std::string exe_dir = vieneu_executable_dir();
    if (!exe_dir.empty()) {
        candidates.push_back(vieneu_join_path(exe_dir, "sea_g2p.bin"));
        candidates.push_back(vieneu_join_path(vieneu_join_path(exe_dir, "assets"), "sea_g2p.bin"));
        candidates.push_back(vieneu_join_path(vieneu_join_path(exe_dir, "sea-g2p"), "sea_g2p.bin"));
    }

    candidates.push_back("sea_g2p.bin");
    candidates.push_back(vieneu_join_path("assets", "sea_g2p.bin"));
    candidates.push_back(vieneu_join_path(vieneu_join_path("third_party", "sea-g2p"), vieneu_join_path(vieneu_join_path("python", "sea_g2p"), "sea_g2p.bin")));

#ifdef VIENEU_SEA_G2P_DEFAULT_DICT
    candidates.push_back(VIENEU_SEA_G2P_DEFAULT_DICT);
#endif

    for (const std::string& candidate : candidates) {
        if (vieneu_file_exists(candidate)) {
            return candidate;
        }
    }
    return candidates.empty() ? std::string() : candidates.front();
}

static std::string trim_ascii_copy(const std::string& value) {
    size_t start = 0;
    while (start < value.size() && std::isspace(static_cast<unsigned char>(value[start]))) {
        start++;
    }
    size_t end = value.size();
    while (end > start && std::isspace(static_cast<unsigned char>(value[end - 1]))) {
        end--;
    }
    return value.substr(start, end - start);
}

static void append_phoneme_segment(std::stringstream& ss, const std::string& segment) {
    const std::string trimmed = trim_ascii_copy(segment);
    if (trimmed.empty()) {
        return;
    }
    const std::streampos pos = ss.tellp();
    if (pos > 0) {
        ss << ' ';
    }
    ss << trimmed;
}

static std::string sea_g2p_take_string(char* value) {
    if (!value) {
        return {};
    }
    std::string out(value);
    sea_g2p_free_string(value);
    return out;
}

static SeaG2pContext* sea_g2p_context() {
    static std::mutex mutex;
    static SeaG2pContext* ctx = nullptr;
    static bool attempted = false;
    std::lock_guard<std::mutex> lock(mutex);
    if (attempted) {
        return ctx;
    }
    attempted = true;

    std::string dict_path = resolve_sea_g2p_dict_path();
    if (dict_path.empty()) {
        return nullptr;
    }

    ctx = sea_g2p_create("vi", dict_path.c_str());
    if (!ctx && !vieneu_getenv_string("VIENEU_SEA_G2P_DEBUG").empty()) {
        const std::string err = sea_g2p_take_string(sea_g2p_last_error());
        std::cerr << "[VieNeu sea-g2p] failed to initialize with dict '" << dict_path << "'";
        if (!err.empty()) {
            std::cerr << ": " << err;
        }
        std::cerr << std::endl;
    }
    return ctx;
}

static bool try_sea_g2p_phonemizer(const std::string& text, std::string& out) {
    SeaG2pContext* ctx = sea_g2p_context();
    if (!ctx) {
        return false;
    }

    std::stringstream ss;
    size_t pos = 0;
    while (pos < text.size()) {
        std::string emotion_token;
        size_t consumed = 0;
        if (consume_emotion_marker(text, pos, emotion_token, consumed)) {
            append_emotion_token(ss, emotion_token);
            pos += consumed;
            continue;
        }

        const size_t segment_start = pos;
        while (pos < text.size()) {
            if (consume_emotion_marker(text, pos, emotion_token, consumed)) {
                break;
            }
            pos++;
        }

        const std::string segment = text.substr(segment_start, pos - segment_start);
        if (trim_ascii_copy(segment).empty()) {
            continue;
        }
        char* phonemes_raw = sea_g2p_run(ctx, segment.c_str(), 0);
        if (!phonemes_raw) {
            if (!vieneu_getenv_string("VIENEU_SEA_G2P_DEBUG").empty()) {
                const std::string err = sea_g2p_take_string(sea_g2p_last_error());
                std::cerr << "[VieNeu sea-g2p] phonemize failed";
                if (!err.empty()) {
                    std::cerr << ": " << err;
                }
                std::cerr << std::endl;
            }
            return false;
        }
        append_phoneme_segment(ss, sea_g2p_take_string(phonemes_raw));
    }

    out = ss.str();
    return !out.empty();
}
#endif

static std::string ascii_lower_copy(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return value;
}

static bool consume_emotion_marker(const std::string& text, size_t pos, std::string& token, size_t& consumed) {
    if (text.compare(pos, 10, "<|emotion_") == 0) {
        const size_t end = text.find("|>", pos + 10);
        if (end != std::string::npos) {
            token = text.substr(pos, end + 2 - pos);
            consumed = token.size();
            return true;
        }
    }

    if (pos >= text.size() || text[pos] != '[') {
        return false;
    }
    const size_t end = text.find(']', pos + 1);
    if (end == std::string::npos) {
        return false;
    }

    std::string inner = text.substr(pos + 1, end - pos - 1);
    while (!inner.empty() && std::isspace(static_cast<unsigned char>(inner.front()))) {
        inner.erase(inner.begin());
    }
    while (!inner.empty() && std::isspace(static_cast<unsigned char>(inner.back()))) {
        inner.pop_back();
    }
    const std::string key = ascii_lower_copy(inner);
    if (key == "chuckle" || key == "cuoi" || key == u8"cười") {
        token = "<|emotion_1|>";
    } else if (key == "sigh" || key == "tho dai" || key == u8"thở dài") {
        token = "<|emotion_2|>";
    } else if (key == "clear throat" || key == "hang giong" || key == u8"hắng giọng") {
        token = "<|emotion_3|>";
    } else {
        return false;
    }
    consumed = end + 1 - pos;
    return true;
}

static void append_emotion_token(std::stringstream& ss, const std::string& token) {
    const std::streampos pos = ss.tellp();
    if (pos > 0) {
        ss << ' ';
    }
    ss << token;
}

std::string VieneuProfile::format_prompt(const std::string& phonemes) {
    return "<|speaker_16|><|TEXT_PROMPT_START|>" + phonemes + "<|TEXT_PROMPT_END|><|SPEECH_GENERATION_START|>";
}

std::vector<int64_t> VieneuProfile::extract_speech_ids(const std::string& generated_text) {
    std::vector<int64_t> ids;
    std::regex re("<\\|(?:speech|s)_(\\d+)\\|>");
    auto start = std::sregex_iterator(generated_text.begin(), generated_text.end(), re);
    auto end = std::sregex_iterator();
    for (auto i = start; i != end; ++i) {
        ids.push_back(std::stoll((*i)[1].str()));
    }
    return ids;
}

std::string VieneuProfile::phonemize(const std::string& text) {
#ifdef VIENEU_USE_SEA_G2P
    std::string sea_g2p_phonemes;
    if (try_sea_g2p_phonemizer(text, sea_g2p_phonemes)) {
        return sea_g2p_phonemes;
    }
#endif

    std::stringstream ss;
    std::string word = "";
    const bool vietnamese_context = has_vietnamese_signal(text);
    
    auto is_punc = [](char c) {
        return c == '.' || c == ',' || c == '?' || c == '!' || c == ';' || c == ':' || c == '"' || c == '(' || c == ')';
    };

    auto process_single_word = [&](const std::string& w_raw) -> std::string {
        if (w_raw.empty()) return "";

        // In Vietnamese sentences, unaccented syllables (e.g. "vua", "gian")
        // must not bypass G2P merely because they contain ASCII characters only.
        bool is_ascii = true;
        for (char c : w_raw) {
            if ((unsigned char)c >= 0x80) {
                is_ascii = false;
                break;
            }
        }
        if (is_ascii && !vietnamese_context) {
            // Keep English words as they are, but add a stress symbol at start if it's alphabetical
            if (std::isalpha((unsigned char)w_raw[0])) {
                return "\u02C8" + w_raw;
            }
            return w_raw;
        }

        // Convert to codepoints
        std::vector<uint32_t> cps = utf8_to_cps(w_raw);
        int tone = 1;
        std::vector<uint32_t> base_cps;

        // Extract tone and map accented characters to base vowel + tone
        for (auto cp : cps) {
            // Lowercase conversion for basic letters
            if (cp >= 'A' && cp <= 'Z') {
                cp = cp - 'A' + 'a';
            }
            // Lowercase conversion for accented letters
            if (cp >= 192 && cp <= 220) {
                cp += 32;
            }
            auto map_res = get_vietnamese_vowel_base(cp);
            if (map_res.tone > 1) {
                tone = map_res.tone;
            }
            base_cps.push_back(map_res.base);
        }

        std::string base_word = cps_to_utf8(base_cps);

        // Define onset patterns and their IPA equivalents
        std::vector<std::pair<std::string, std::string>> onsets = {
            {"tr", "t\xca\x83"},   // tʃ
            {"ch", "t\xca\x83"},   // tʃ
            {"kh", "x"},
            {"ph", "f"},
            {"nh", "\xc9\xb2"},    // ɲ
            {"ngh", "\xc5\x8b"},   // ŋ
            {"ng", "\xc5\x8b"},    // ŋ
            {"th", "t"},
            {"gi", "z"},
            {"qu", "kw"},
            {"gh", "\xc9\xa3"},    // ɣ
            {"g", "\xc9\xa3"},     // ɣ
            {"b", "b"},
            {"c", "k"},
            {"k", "k"},
            {"d", "z"},
            {"\xc4\x91", "\xc9\x97"}, // đ -> ɗ
            {"h", "h"},
            {"l", "l"},
            {"m", "m"},
            {"n", "n"},
            {"r", "\xc9\xb9"},     // ɹ
            {"s", "s"},
            {"x", "s"},
            {"t", "t\xcc\xaa"},    // t̪ (dental t)
            {"v", "v"}
        };

        std::string ipa_onset = "";
        std::string rime = base_word;

        for (const auto& pair : onsets) {
            if (base_word.rfind(pair.first, 0) == 0) {
                ipa_onset = pair.second;
                rime = base_word.substr(pair.first.size());
                break;
            }
        }

        // Process rime into vowel and coda
        std::string ipa_vowel = "";
        std::string ipa_coda = "";
        auto map_coda = [](const std::string& rem) -> std::string {
            if (rem == "ng") return "\xc5\x8b";
            if (rem == "nh") return "\xc9\xb2";
            if (rem == "ch") return "t\xca\x83";
            if (rem == "c") return "k";
            if (rem == "t") return "t";
            if (rem == "p") return "p";
            if (rem == "m") return "m";
            if (rem == "n") return "n";
            if (rem == "o" || rem == "u") return "w";
            if (rem == "i" || rem == "y") return "j";
            return "";
        };

        // Standard rime mappings
        if (rime.rfind("ươ", 0) == 0 || rime.rfind("\xc6\xb0\xc6\xa1", 0) == 0) {
            ipa_vowel = "y\xc9\x99"; // yə
            std::string rem = rime.substr(std::min<size_t>(4, rime.size())); // Guard short ASCII variants like "ua"/"ia".
            if (rem == "ng") ipa_coda = "\xc5\x8b";
            else if (rem == "n") ipa_coda = "n";
            else if (rem == "m") ipa_coda = "m";
            else if (rem == "p") ipa_coda = "p";
            else if (rem == "t") ipa_coda = "t";
            else if (rem == "c") ipa_coda = "k";
            else if (rem == "i" || rem == "y") ipa_coda = "j";
            else if (rem == "o" || rem == "u") ipa_coda = "w";
        }
        else if (rime.rfind("iê", 0) == 0 || rime.rfind("yê", 0) == 0 || rime.rfind("ia", 0) == 0 || rime.rfind("ya", 0) == 0) {
            ipa_vowel = "i\xc9\x9b"; // iɛ
            const size_t prefix_bytes = (rime.rfind("ia", 0) == 0 || rime.rfind("ya", 0) == 0) ? 2 : 3;
            std::string rem = rime.substr(std::min(prefix_bytes, rime.size()));
            if (rem == "ng") ipa_coda = "\xc5\x8b";
            else if (rem == "n") ipa_coda = "n";
            else if (rem == "m") ipa_coda = "m";
            else if (rem == "p") ipa_coda = "p";
            else if (rem == "t") ipa_coda = "t";
            else if (rem == "c") ipa_coda = "k";
            else if (rem == "i" || rem == "y") ipa_coda = "j";
            else if (rem == "o" || rem == "u") ipa_coda = "w";
        }
        else if (rime.rfind("uô", 0) == 0 || rime.rfind("ua", 0) == 0) {
            ipa_vowel = "u\xc9\x99"; // uə
            const size_t prefix_bytes = rime.rfind("ua", 0) == 0 ? 2 : 3;
            std::string rem = rime.substr(std::min(prefix_bytes, rime.size()));
            if (rem == "ng") ipa_coda = "\xc5\x8b";
            else if (rem == "n") ipa_coda = "n";
            else if (rem == "m") ipa_coda = "m";
            else if (rem == "p") ipa_coda = "p";
            else if (rem == "t") ipa_coda = "t";
            else if (rem == "c") ipa_coda = "k";
            else if (rem == "i" || rem == "y") ipa_coda = "j";
            else if (rem == "o" || rem == "u") ipa_coda = "w";
        }
        else {
            // General vowel parsing
            std::string v_sp = "";
            std::string c_sp = "";
            if (rime.rfind("oă", 0) == 0) { v_sp = "w\xc9\x90"; rime = rime.substr(std::min<size_t>(3, rime.size())); } // wɐ
            else if (rime.rfind("oa", 0) == 0) { v_sp = "wa\xcb\x90"; rime = rime.substr(std::min<size_t>(2, rime.size())); } // waː
            else if (rime.rfind("uê", 0) == 0) { v_sp = "we"; rime = rime.substr(std::min<size_t>(3, rime.size())); }
            else if (rime.rfind("uy", 0) == 0) { v_sp = "wi"; rime = rime.substr(std::min<size_t>(2, rime.size())); }
            else if (rime.rfind("anh", 0) == 0) { v_sp = "e"; c_sp = "\xc9\xb2"; rime = ""; } // eɲ
            else if (rime.rfind("ach", 0) == 0) { v_sp = "\xc9\x90"; c_sp = "t\xca\x83"; rime = ""; } // ɐtʃ
            else if (rime.rfind("inh", 0) == 0) { v_sp = "i"; c_sp = "\xc9\xb2"; rime = ""; } // iɲ
            else if (rime.rfind("ich", 0) == 0) { v_sp = "\xca\xaa"; c_sp = "k"; rime = ""; } // ɪk

            if (v_sp.empty() && !rime.empty()) {
                // Read single nucleus vowel
                std::vector<uint32_t> r_cps = utf8_to_cps(rime);
                if (!r_cps.empty()) {
                    uint32_t v_cp = r_cps[0];
                    if (v_cp == 'a') v_sp = "a\xcb\x90"; // aː
                    else if (v_cp == 259) v_sp = "a"; // ă -> a
                    else if (v_cp == 226) v_sp = "\xc9\x99"; // â -> ə
                    else if (v_cp == 417) v_sp = "\xc9\x99\xcb\x90"; // ơ -> əː
                    else if (v_cp == 432) v_sp = "y"; // ư -> y
                    else if (v_cp == 'e') v_sp = "\xc9\x9b"; // e -> ɛ
                    else if (v_cp == 'e' + 12 || v_cp == 234) v_sp = "e"; // ê -> e
                    // Wait, ê is 234 decimal.
                    else if (v_cp == 'i' || v_cp == 'y') v_sp = "i";
                    else if (v_cp == 'o') v_sp = "\xc5\x8f"; // o -> ɔ
                    else if (v_cp == 244) v_sp = "o"; // ô -> o
                    else if (v_cp == 'u') v_sp = "u";
                    else v_sp = "a\xcb\x90"; // fallback

                    // Remove vowel from rime to get remaining coda
                    rime = cps_to_utf8(std::vector<uint32_t>(r_cps.begin() + 1, r_cps.end()));
                }
            }

            ipa_vowel = v_sp;
            if (!c_sp.empty()) {
                ipa_coda = c_sp;
            } else {
                ipa_coda = map_coda(rime);
            }
        }

        // Build IPA string
        std::string ipa_tone = "";
        if (tone == 2) ipa_tone = "2";
        else if (tone == 3) ipa_tone = "\xc9\x9c"; // ɜ (sắc)
        else if (tone == 4) ipa_tone = "4";
        else if (tone == 5) ipa_tone = "5";
        else if (tone == 6) ipa_tone = "6";

        // Stress marker \u02C8 (ˈ)
        return ipa_onset + "\u02C8" + ipa_vowel + ipa_tone + ipa_coda;
    };

    // Syllabify text
    for (size_t i = 0; i < text.size(); ++i) {
        std::string emotion_token;
        size_t emotion_consumed = 0;
        if (consume_emotion_marker(text, i, emotion_token, emotion_consumed)) {
            if (!word.empty()) {
                ss << process_single_word(word);
                word = "";
            }
            append_emotion_token(ss, emotion_token);
            i += emotion_consumed - 1;
            continue;
        }

        char c = text[i];
        if (is_punc(c)) {
            if (!word.empty()) {
                ss << process_single_word(word);
                word = "";
            }
            ss << c;
        } else if (c == ' ' || c == '\t' || c == '\n' || c == '\r') {
            if (!word.empty()) {
                ss << process_single_word(word);
                word = "";
            }
            ss << c;
        } else {
            word += c;
        }
    }
    if (!word.empty()) {
        ss << process_single_word(word);
    }

    return ss.str();
}

bool VieneuProfile::synthesize(
    LlamaBackend& llama,
    NeuCodecOnnx& decoder,
    const std::string& text,
    const std::vector<float>& voice_embedding,
    float temperature,
    int top_k,
    int max_tokens,
    bool skip_phonemize,
    std::vector<float>& out_audio,
    const VieneuProgressFn& progress)
{
    out_audio.clear();
    vieneu_report_progress(progress, "prepare", 0, 0, 0.0f, "Preparing v2 synthesis.");

    std::string phonemes = text;
    if (!skip_phonemize) {
        vieneu_report_progress(progress, "phonemize", 0, 1, 0.02f, "Phonemizing input text.");
        phonemes = phonemize(text);
        vieneu_report_progress(progress, "phonemize", 1, 1, 0.05f, "Phonemization complete.");
    }

    std::string prompt = format_prompt(phonemes);
    auto prompt_tokens = llama.tokenize(prompt, true);

    vieneu_report_progress(progress, "prefill", 0, 1, 0.06f, "Running prompt prefill.");
    if (!llama.decode(prompt_tokens, 0, true)) {
        return false;
    }
    vieneu_report_progress(progress, "prefill", 1, 1, 0.10f, "Prompt prefill complete.");

    std::string generated_text = "";
    llama_token curr_token = 0;
    int n_tokens = 0;

    while (n_tokens < max_tokens) {
        curr_token = llama.sample(temperature, top_k);
        if (llama.is_eog(curr_token)) {
            break;
        }

        std::string piece = llama.token_to_piece(curr_token);
        generated_text += piece;

        if (generated_text.find("<|SPEECH_GENERATION_END|>") != std::string::npos) {
            break;
        }

        std::vector<llama_token> next_tokens = { curr_token };
        if (!llama.decode(next_tokens, 0, false)) {
            break;
        }

        n_tokens++;
        const float token_progress = max_tokens > 0 ? static_cast<float>(n_tokens) / static_cast<float>(max_tokens) : 0.0f;
        vieneu_report_progress(progress, "generate_tokens", n_tokens, max_tokens, 0.10f + token_progress * 0.75f, "Generating speech tokens.");
    }

    auto speech_ids = extract_speech_ids(generated_text);
    if (speech_ids.empty()) {
        std::cerr << "[VieNeu] No speech tokens generated!" << std::endl;
        return false;
    }

    // Convert speech ids to int64 for the decoder
    std::vector<int64_t> speech_ids_int64(speech_ids.begin(), speech_ids.end());

    std::vector<float> chunk_audio;
    vieneu_report_progress(progress, "decode_audio", 0, 1, 0.90f, "Decoding speech tokens to audio.");
    if (!decoder.decode_vieneu(speech_ids_int64, voice_embedding, chunk_audio)) {
        return false;
    }

    out_audio = std::move(chunk_audio);
    vieneu_report_progress(progress, "complete", 1, 1, 1.0f, "Synthesis complete.");
    return true;
}
