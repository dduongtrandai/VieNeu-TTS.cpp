#include "vieneu/vieneu.h"

#include <cstdlib>
#include <iostream>
#include <string>

static int require_contains(const std::string& value, const std::string& expected, const char* message) {
    if (value.find(expected) != std::string::npos) {
        return 0;
    }
    std::cerr << message << "\nExpected fragment: " << expected << "\nActual: " << value << "\n";
    return 1;
}

static int require_not_contains(const std::string& value, const std::string& rejected, const char* message) {
    if (value.find(rejected) == std::string::npos) {
        return 0;
    }
    std::cerr << message << "\nRejected fragment: " << rejected << "\nActual: " << value << "\n";
    return 1;
}

static int require_contains_any(
    const std::string& value,
    const std::string& expected_a,
    const std::string& expected_b,
    const char* message) {
    if (value.find(expected_a) != std::string::npos || value.find(expected_b) != std::string::npos) {
        return 0;
    }
    std::cerr << message << "\nExpected fragment A: " << expected_a
              << "\nExpected fragment B: " << expected_b << "\nActual: " << value << "\n";
    return 1;
}

int main() {
    const std::string sentence = VieneuProfile::phonemize(
        u8"Tôi thích chơi cờ vua vào thời gian rảnh rỗi");

    int failed = 0;
    failed += require_contains(sentence, u8"vˈuə", "ASCII Vietnamese syllable 'vua' bypassed Vietnamese G2P.");
    failed += require_not_contains(sentence, u8"ˈvua", "ASCII Vietnamese syllable was incorrectly retained as English text.");

    const std::string codas = VieneuProfile::phonemize(u8"miếng loan");
    failed += require_contains(codas, u8"iɛɜŋ", "UTF-8 rime 'iếng' lost its coda.");
    failed += require_contains_any(
        codas,
        u8"lwˈaːn",
        u8"waːn",
        "ASCII rime 'oan' lost its coda.");

    const std::string nguoi = VieneuProfile::phonemize(std::string("ng") + "\xC6\xB0" "\xE1\xBB\x9D" "i");
    failed += require_contains(
        nguoi,
        std::string("\xC5\x8B") + "\xCB\x88" "y" "\xC9\x99" "2j",
        "Vietnamese rime 'uoi with horn' lost its final glide.");

    const std::string normalized = VieneuProfile::phonemize(u8"Giá SP500 hôm nay là 4.200,5 điểm.");
    const bool require_python_backend = std::getenv("VIENEU_REQUIRE_PYTHON_PHONEMIZER") != nullptr;
    if (require_python_backend) {
        failed += require_contains(
            normalized,
            u8"bˈoɜn",
            "Python sea-g2p backend was required but numeric normalization was not observed.");
    }
    if (normalized.find(u8"bˈoɜn") != std::string::npos) {
        failed += require_contains(
            normalized,
            u8"ŋˈi2n",
            "Python sea-g2p backend did not normalize thousands in a numeric expression.");
    }

    const std::string emotion = VieneuProfile::phonemize(u8"[cười] Trời ơi, nghe giống người thật quá!");
    failed += require_contains(
        emotion,
        "<|emotion_1|>",
        "VieNeu phonemizer did not preserve [cười] as <|emotion_1|>.");
    if (require_python_backend) {
        failed += require_contains(
            emotion,
            "<|emotion_1|>",
            "Python VieNeu emotion phonemizer was required but emotion token was not preserved.");
    }
    if (emotion.find("<|emotion_1|>") != std::string::npos) {
        failed += require_contains(
            emotion,
            u8"ŋˈyə2j",
            "Python VieNeu emotion phonemizer did not preserve Vietnamese G2P around emotion tags.");
    }

    return failed == 0 ? 0 : 1;
}
