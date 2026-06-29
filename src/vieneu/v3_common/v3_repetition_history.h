#ifndef VIENEU_V3_REPETITION_HISTORY_H
#define VIENEU_V3_REPETITION_HISTORY_H

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

struct V3RepetitionHistory {
    std::vector<int32_t> indices;
    std::vector<uint8_t> seen;

    void initialize(size_t vocab_size) {
        indices.clear();
        seen.assign(vocab_size, 0);
    }

    void clear() {
        indices.clear();
        std::fill(seen.begin(), seen.end(), 0);
    }

    void add(int32_t index) {
        if (index < 0) {
            return;
        }
        const size_t pos = static_cast<size_t>(index);
        if (pos >= seen.size() || seen[pos]) {
            return;
        }
        seen[pos] = 1;
        indices.push_back(index);
    }
};

#endif // VIENEU_V3_REPETITION_HISTORY_H
