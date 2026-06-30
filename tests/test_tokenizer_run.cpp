#include "vieneu/v3_native/v3_native_tokenizer.h"
#include <iostream>
#include <vector>

static int require_contains_id(const std::vector<int64_t>& ids, int64_t expected, const char* message) {
    for (const int64_t id : ids) {
        if (id == expected) {
            return 0;
        }
    }
    std::cerr << message << "\nExpected ID: " << expected << "\nActual IDs: ";
    for (const int64_t id : ids) {
        std::cerr << id << " ";
    }
    std::cerr << "\n";
    return 1;
}

int main() {
    V3NativeTokenizer tokenizer;
    std::string error;
    if (!tokenizer.load(".models/vieneu-v3-turbo/tokenizer.json", error)) {
        std::cerr << "Load failed: " << error << "\n";
        return 1;
    }
    std::vector<int64_t> ids = tokenizer.encode("Xin chào, đây là bài kiểm tra");
    std::cout << "IDs: ";
    for (auto id : ids) {
        std::cout << id << " ";
    }
    std::cout << "\n";

    int failed = 0;
    const std::vector<int64_t> emotion_ids = tokenizer.encode("<|emotion_1|> hello <|emotion_2|>. <|emotion_3|>");
    failed += require_contains_id(emotion_ids, 9, "Tokenizer did not preserve <|emotion_1|> as a special token.");
    failed += require_contains_id(emotion_ids, 10, "Tokenizer did not preserve <|emotion_2|> as a special token.");
    failed += require_contains_id(emotion_ids, 11, "Tokenizer did not preserve <|emotion_3|> as a special token.");
    return failed == 0 ? 0 : 1;
}
