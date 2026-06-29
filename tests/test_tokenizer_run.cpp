#include "vieneu/v3_native/v3_native_tokenizer.h"
#include <iostream>
#include <vector>

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
    return 0;
}
