#include "vieneu.h"
#include <regex>
#include <iostream>
#include <map>
#include <algorithm>
#include <sstream>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <mutex>
#include <unordered_map>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <unistd.h>
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

static std::string getenv_string(const char* name) {
    const char* value = std::getenv(name);
    return value ? std::string(value) : std::string();
}

static bool file_exists_local(const std::string& path) {
    if (path.empty()) {
        return false;
    }
    std::ifstream fs(path, std::ios::binary);
    return fs.good();
}

static std::string dirname_local(std::string path) {
    const size_t pos = path.find_last_of("/\\");
    if (pos == std::string::npos) {
        return {};
    }
    return path.substr(0, pos);
}

static std::string join_path_local(const std::string& a, const std::string& b) {
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

static std::string quote_shell_arg(const std::string& value) {
#ifdef _WIN32
    std::string out = "\"";
    for (char c : value) {
        if (c == '"') {
            out += "\\\"";
        } else {
            out.push_back(c);
        }
    }
    out.push_back('"');
    return out;
#else
    std::string out = "'";
    for (char c : value) {
        if (c == '\'') {
            out += "'\\''";
        } else {
            out.push_back(c);
        }
    }
    out.push_back('\'');
    return out;
#endif
}

static std::string make_temp_text_path() {
#ifdef _WIN32
    char temp_dir[MAX_PATH + 1] = {};
    const DWORD dir_len = GetTempPathA(MAX_PATH, temp_dir);
    if (dir_len == 0 || dir_len > MAX_PATH) {
        return {};
    }
    char temp_file[MAX_PATH + 1] = {};
    if (GetTempFileNameA(temp_dir, "vng", 0, temp_file) == 0) {
        return {};
    }
    return temp_file;
#else
    char tmpl[] = "/tmp/vieneu-g2p-XXXXXX";
    const int fd = mkstemp(tmpl);
    if (fd < 0) {
        return {};
    }
    close(fd);
    return tmpl;
#endif
}

static std::vector<std::string> vieneu_repo_source_candidates() {
    std::vector<std::string> out;
    const std::string explicit_path = getenv_string("VIENEU_PYTHONPATH");
    if (!explicit_path.empty()) {
        out.push_back(explicit_path);
    }
    const std::string explicit_repo = getenv_string("VIENEU_REPO");
    if (!explicit_repo.empty()) {
        out.push_back(join_path_local(explicit_repo, "src"));
    }
    const std::string vieneu_repo = getenv_string("VIENEU_TTS_DIR");
    if (!vieneu_repo.empty()) {
        out.push_back(join_path_local(vieneu_repo, "src"));
    }

    const std::string source_file = __FILE__;
    const std::string profile_dir = dirname_local(source_file);
    const std::string src_dir = dirname_local(profile_dir);
    const std::string project_dir = dirname_local(src_dir);
    const std::string parent_dir = dirname_local(project_dir);
    if (!parent_dir.empty()) {
        out.push_back(join_path_local(join_path_local(parent_dir, "VieNeu-TTS"), "src"));
    }

    out.push_back(join_path_local("VieNeu-TTS", "src"));
    out.push_back(join_path_local(join_path_local("..", "VieNeu-TTS"), "src"));
    out.push_back(join_path_local(join_path_local(join_path_local("..", ".."), "VieNeu-TTS"), "src"));
    return out;
}

static std::vector<std::string> vieneu_python_candidates() {
    std::vector<std::string> out;
    const std::string explicit_python = getenv_string("VIENEU_PYTHON");
    if (!explicit_python.empty()) {
        out.push_back(explicit_python);
    }
    const std::string explicit_python_alt = getenv_string("VIENEU_TTS_PYTHON");
    if (!explicit_python_alt.empty()) {
        out.push_back(explicit_python_alt);
    }
    const std::string venv = getenv_string("VIRTUAL_ENV");
    if (!venv.empty()) {
#ifdef _WIN32
        out.push_back(join_path_local(join_path_local(venv, "Scripts"), "python.exe"));
#else
        out.push_back(join_path_local(join_path_local(venv, "bin"), "python"));
#endif
    }

    const std::string source_file = __FILE__;
    const std::string profile_dir = dirname_local(source_file);
    const std::string src_dir = dirname_local(profile_dir);
    const std::string project_dir = dirname_local(src_dir);
    const std::string parent_dir = dirname_local(project_dir);
    if (!parent_dir.empty()) {
        const std::string vieneu_dir = join_path_local(parent_dir, "VieNeu-TTS");
#ifdef _WIN32
        out.push_back(join_path_local(join_path_local(join_path_local(vieneu_dir, ".venv"), "Scripts"), "python.exe"));
#else
        out.push_back(join_path_local(join_path_local(join_path_local(vieneu_dir, ".venv"), "bin"), "python"));
#endif
    }

#ifdef _WIN32
    out.push_back(join_path_local(join_path_local(join_path_local("VieNeu-TTS", ".venv"), "Scripts"), "python.exe"));
    out.push_back(join_path_local(join_path_local(join_path_local(join_path_local("..", "VieNeu-TTS"), ".venv"), "Scripts"), "python.exe"));
    out.push_back(join_path_local(join_path_local(join_path_local(join_path_local(join_path_local("..", ".."), "VieNeu-TTS"), ".venv"), "Scripts"), "python.exe"));
    out.push_back("python");
#else
    out.push_back(join_path_local(join_path_local(join_path_local("VieNeu-TTS", ".venv"), "bin"), "python"));
    out.push_back(join_path_local(join_path_local(join_path_local(join_path_local("..", "VieNeu-TTS"), ".venv"), "bin"), "python"));
    out.push_back(join_path_local(join_path_local(join_path_local(join_path_local(join_path_local("..", ".."), "VieNeu-TTS"), ".venv"), "bin"), "python"));
    out.push_back("python3");
    out.push_back("python");
#endif
    return out;
}

static bool run_python_vieneu_phonemizer(const std::string& text, std::string& out) {
    const bool debug = !getenv_string("VIENEU_PHONEMIZER_DEBUG").empty();
    const std::string temp_path = make_temp_text_path();
    if (temp_path.empty()) {
        if (debug) {
            std::cerr << "[VieNeu phonemizer] failed to allocate temp file\n";
        }
        return false;
    }
    {
        std::ofstream fs(temp_path, std::ios::binary | std::ios::trunc);
        if (!fs.is_open()) {
            if (debug) {
                std::cerr << "[VieNeu phonemizer] failed to open temp file: " << temp_path << "\n";
            }
            std::remove(temp_path.c_str());
            return false;
        }
        fs.write(text.data(), static_cast<std::streamsize>(text.size()));
    }

    const std::string script =
        "import sys;"
        "sys.stdout.reconfigure(encoding='utf-8');"
        "[sys.path.insert(0,p) for p in sys.argv[2:] if p];"
        "from vieneu_utils.phonemize_text import phonemize_text_with_emotions as p;"
        "sys.stdout.write(p(open(sys.argv[1],encoding='utf-8').read()))";

    const std::vector<std::string> source_paths = vieneu_repo_source_candidates();
    for (const std::string& python : vieneu_python_candidates()) {
        if (python != "python" && python != "python3" && !file_exists_local(python)) {
            if (debug) {
                std::cerr << "[VieNeu phonemizer] missing python candidate: " << python << "\n";
            }
            continue;
        }
#ifdef _WIN32
        std::string command = "call " + quote_shell_arg(python) + " -X utf8 -c " +
            quote_shell_arg(script) + " " + quote_shell_arg(temp_path);
#else
        std::string command = quote_shell_arg(python) + " -X utf8 -c " +
            quote_shell_arg(script) + " " + quote_shell_arg(temp_path);
#endif
        for (const std::string& source_path : source_paths) {
            command += " " + quote_shell_arg(source_path);
        }
#ifdef _WIN32
        command += debug ? " 2>&1" : " 2>NUL";
        FILE* pipe = _popen(command.c_str(), "r");
#else
        command += debug ? " 2>&1" : " 2>/dev/null";
        FILE* pipe = popen(command.c_str(), "r");
#endif
        if (!pipe) {
            if (debug) {
                std::cerr << "[VieNeu phonemizer] popen failed for: " << command << "\n";
            }
            continue;
        }
        std::string result;
        char buffer[4096];
        while (true) {
            const size_t n = std::fread(buffer, 1, sizeof(buffer), pipe);
            if (n > 0) {
                result.append(buffer, n);
            }
            if (n < sizeof(buffer)) {
                break;
            }
        }
#ifdef _WIN32
        const int rc = _pclose(pipe);
#else
        const int rc = pclose(pipe);
#endif
        if (debug) {
            std::cerr << "[VieNeu phonemizer] candidate rc=" << rc
                      << " python=" << python
                      << " bytes=" << result.size() << "\n";
            if (!result.empty()) {
                std::cerr << "[VieNeu phonemizer] output=" << result << "\n";
            }
        }
        if (rc == 0) {
            std::remove(temp_path.c_str());
            out = std::move(result);
            return true;
        }
    }

    std::remove(temp_path.c_str());
    return false;
}

static bool try_python_vieneu_phonemizer(const std::string& text, std::string& out) {
    const std::string disabled = getenv_string("VIENEU_PYTHON_PHONEMIZER");
    if (disabled == "0" || disabled == "false" || disabled == "FALSE" || disabled == "off" || disabled == "OFF") {
        return false;
    }

    static std::mutex mutex;
    static std::unordered_map<std::string, std::string> cache;
    static bool known_unavailable = false;
    std::lock_guard<std::mutex> lock(mutex);
    auto it = cache.find(text);
    if (it != cache.end()) {
        out = it->second;
        return true;
    }
    if (known_unavailable) {
        return false;
    }
    if (run_python_vieneu_phonemizer(text, out)) {
        cache.emplace(text, out);
        return true;
    }
    known_unavailable = true;
    return false;
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
    std::string python_phonemes;
    if (try_python_vieneu_phonemizer(text, python_phonemes)) {
        return python_phonemes;
    }

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
    std::vector<float>& out_audio)
{
    out_audio.clear();

    std::string phonemes = text;
    if (!skip_phonemize) {
        phonemes = phonemize(text);
    }

    std::string prompt = format_prompt(phonemes);
    auto prompt_tokens = llama.tokenize(prompt, true);

    if (!llama.decode(prompt_tokens, 0, true)) {
        return false;
    }

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
    }

    auto speech_ids = extract_speech_ids(generated_text);
    if (speech_ids.empty()) {
        std::cerr << "[VieNeu] No speech tokens generated!" << std::endl;
        return false;
    }

    // Convert speech ids to int64 for the decoder
    std::vector<int64_t> speech_ids_int64(speech_ids.begin(), speech_ids.end());

    std::vector<float> chunk_audio;
    if (!decoder.decode_vieneu(speech_ids_int64, voice_embedding, chunk_audio)) {
        return false;
    }

    out_audio = std::move(chunk_audio);
    return true;
}
