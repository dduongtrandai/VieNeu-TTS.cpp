#ifndef V3_NATIVE_SAMPLER_H
#define V3_NATIVE_SAMPLER_H

#include <vector>
#include <unordered_set>
#include <random>

class V3NativeSampler {
public:
    V3NativeSampler(uint32_t seed = 42);

    int64_t sample_logits(
        std::vector<float>& logits,
        float temperature,
        int top_k,
        float top_p,
        float repetition_penalty,
        const std::unordered_set<int>* previous);

private:
    std::mt19937 rng_;
    
    // Thread-safe scratch buffers to avoid allocation overhead during sampling
    std::vector<std::pair<float, size_t>> sampling_pairs_;
    std::vector<float> sampling_probs_;
};

void matvec_native(const float* vec, const float* matrix, int64_t hidden, int64_t vocab, std::vector<float>& logits);

#endif // V3_NATIVE_SAMPLER_H
