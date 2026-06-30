#include "vieneu/v3_native/v3_native_prompt.h"

#include <iostream>
#include <string>
#include <vector>

static int require_equal(const std::string& actual, const std::string& expected, const char* message) {
    if (actual == expected) {
        return 0;
    }
    std::cerr << message << "\nExpected: " << expected << "\nActual: " << actual << "\n";
    return 1;
}

static int require_starts_with(const std::string& actual, const std::string& expected, const char* message) {
    if (actual.rfind(expected, 0) == 0) {
        return 0;
    }
    std::cerr << message << "\nExpected prefix: " << expected << "\nActual: " << actual << "\n";
    return 1;
}

int main() {
    int failed = 0;

    const std::vector<std::string> chunks = chunk_text_v3(
        u8"Xin chào mọi người! [hắng giọng] Như bạn đang nghe thấy đấy, tốc độ xử lý của mình cực kỳ nhanh và mượt mà, giúp phản hồi gần như ngay lập tức theo thời gian thực. Chính vì vậy, mình rất phù hợp để ứng dụng trực tiếp vào các hệ thống Chatbot thông minh, trợ lý ảo, hoặc làm tổng đài viên tự động cho các doanh nghiệp. Tiện lợi quá đúng không ạ? [cười] Hi vọng phiên bản nâng cấp v3 này sẽ mang lại trải nghiệm tuyệt vời cho dự án của bạn.",
        384);

    if (chunks.size() != 2) {
        std::cerr << "Expected v3 benchmark text to match the source pipeline chunk count; got "
                  << chunks.size() << " chunks.\n";
        for (size_t i = 0; i < chunks.size(); ++i) {
            std::cerr << "chunk[" << i << "]: " << chunks[i] << "\n";
        }
        return 1;
    }

    failed += require_equal(
        chunks[0].find("<|emotion_3|>") == std::string::npos ? "" : "<|emotion_3|>",
        "<|emotion_3|>",
        "[hắng giọng] should be canonicalized before chunking.");
    failed += require_starts_with(
        chunks[1],
        "<|emotion_1|> ",
        "[cười] should start the next chunk, followed by the text after it.");

    const std::vector<std::string> phoneme_chunks = chunk_text_v3(
        u8"Ổn rồi. <|emotion_1|> tiếp tục nào.",
        150);
    if (phoneme_chunks.size() != 1) {
        std::cerr << "Expected pre-tokenized emotion marker to stay inline when the chunk fits; got "
                  << phoneme_chunks.size() << " chunks.\n";
        return 1;
    }
    failed += require_equal(
        phoneme_chunks[0],
        u8"Ổn rồi. <|emotion_1|> tiếp tục nào.",
        "Pre-tokenized emotion marker should be preserved inline.");

    return failed == 0 ? 0 : 1;
}
