#include "../vieneu_v3_onnx.h"
#include "vieneu_v3_onnx_internal.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <random>
#include <vector>

// --- Math Kernels ---

void matvec_transposed(const float* vec, const float* matrix_hv, int64_t hidden, int64_t vocab, std::vector<float>& logits) {
    logits.assign(static_cast<size_t>(vocab), 0.0f);
    float* out = logits.data();
    for (int64_t h = 0; h < hidden; ++h) {
        const float scale = vec[h];
        const float* row = matrix_hv + h * vocab;
#if defined(_MSC_VER)
#pragma loop(ivdep)
#elif defined(__GNUC__) || defined(__clang__)
#pragma GCC ivdep
#endif
        for (int64_t v = 0; v < vocab; ++v) {
            out[v] += scale * row[v];
        }
    }
}

std::vector<float> softmax(const std::vector<float>& logits) {
    float max_v = -std::numeric_limits<float>::infinity();
    for (float v : logits) {
        if (v > max_v) max_v = v;
    }
    double sum = 0.0;
    std::vector<float> probs(logits.size());
    for (size_t i = 0; i < logits.size(); ++i) {
        const double e = std::exp(static_cast<double>(logits[i] - max_v));
        probs[i] = static_cast<float>(e);
        sum += e;
    }
    if (sum <= 0.0 || !std::isfinite(sum)) {
        const float uniform = logits.empty() ? 0.0f : 1.0f / static_cast<float>(logits.size());
        std::fill(probs.begin(), probs.end(), uniform);
        return probs;
    }
    for (float& p : probs) {
        p = static_cast<float>(static_cast<double>(p) / sum);
    }
    return probs;
}

// --- VieneuV3OnnxEngine Sampling Member Function ---

int64_t VieneuV3OnnxEngine::sample_logits(
    std::vector<float>& logits,
    float temperature,
    int top_k,
    float top_p,
    float repetition_penalty,
    const std::unordered_set<int>* previous) {
    if (previous && std::fabs(repetition_penalty - 1.0f) > 1e-6f) {
        for (int idx : *previous) {
            if (idx >= 0 && static_cast<size_t>(idx) < logits.size()) {
                logits[static_cast<size_t>(idx)] = logits[static_cast<size_t>(idx)] < 0.0f
                    ? logits[static_cast<size_t>(idx)] * repetition_penalty
                    : logits[static_cast<size_t>(idx)] / repetition_penalty;
            }
        }
    }
    if (!(temperature > 0.0f)) {
        return static_cast<int64_t>(std::distance(logits.begin(), std::max_element(logits.begin(), logits.end())));
    }
    for (float& v : logits) {
        v /= temperature;
    }
    if (top_k > 0 && static_cast<size_t>(top_k) < logits.size()) {
        std::vector<float> tmp = logits;
        std::nth_element(tmp.begin(), tmp.end() - top_k, tmp.end());
        const float kth = *(tmp.end() - top_k);
        for (float& v : logits) {
            if (v < kth) v = -std::numeric_limits<float>::infinity();
        }
    }
    if (top_p > 0.0f && top_p < 1.0f) {
        std::vector<size_t> order(logits.size());
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(), [&](size_t a, size_t b) { return logits[a] > logits[b]; });
        std::vector<float> sorted(order.size());
        for (size_t i = 0; i < order.size(); ++i) sorted[i] = logits[order[i]];
        const std::vector<float> probs = softmax(sorted);
        float cumulative_before = 0.0f;
        for (size_t i = 0; i < order.size(); ++i) {
            if (cumulative_before > top_p) {
                logits[order[i]] = -std::numeric_limits<float>::infinity();
            }
            cumulative_before += probs[i];
        }
    }
    const std::vector<float> probs = softmax(logits);
    std::discrete_distribution<int64_t> dist(probs.begin(), probs.end());
    return dist(rng_);
}
