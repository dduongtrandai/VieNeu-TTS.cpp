#include "../vieneu_v3_onnx.h"
#include "vieneu_v3_onnx_internal.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <random>
#include <vector>
#include "../v3_common/v3_repetition_history.h"

// --- Math Kernels ---

namespace {

void collect_top_k_pairs(const std::vector<float>& logits, size_t k, std::vector<std::pair<float, size_t>>& out) {
    out.clear();
    out.reserve(k);
    const auto by_logit_min_heap = [](const std::pair<float, size_t>& a, const std::pair<float, size_t>& b) {
        return a.first > b.first;
    };
    for (size_t i = 0; i < logits.size(); ++i) {
        const float value = logits[i];
        if (out.size() < k) {
            out.push_back({value, i});
            std::push_heap(out.begin(), out.end(), by_logit_min_heap);
            continue;
        }
        if (!out.empty() && value > out.front().first) {
            std::pop_heap(out.begin(), out.end(), by_logit_min_heap);
            out.back() = {value, i};
            std::push_heap(out.begin(), out.end(), by_logit_min_heap);
        }
    }
    std::sort(
        out.begin(),
        out.end(),
        [](const std::pair<float, size_t>& a, const std::pair<float, size_t>& b) {
            return a.first > b.first;
        });
}

} // namespace

void matvec_transposed(const float* vec, const float* matrix_hv, int64_t hidden, int64_t vocab, std::vector<float>& logits) {
    const size_t out_size = static_cast<size_t>(vocab);
    if (logits.size() != out_size) {
        logits.resize(out_size);
    }
    std::fill(logits.begin(), logits.end(), 0.0f);
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
    const V3RepetitionHistory* previous) {
    if (previous && std::fabs(repetition_penalty - 1.0f) > 1e-6f) {
        for (int32_t idx : previous->indices) {
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
    
    const size_t N = logits.size();
    if (top_k > 0 && static_cast<size_t>(top_k) < N) {
        const size_t k = static_cast<size_t>(top_k);
        collect_top_k_pairs(logits, k, sampling_pairs_);

        if (top_p > 0.0f && top_p < 1.0f && !sampling_pairs_.empty()) {
            const float max_v = sampling_pairs_[0].first;
            double sum = 0.0;
            sampling_probs_.resize(sampling_pairs_.size());
            for (size_t i = 0; i < sampling_pairs_.size(); ++i) {
                const double e = std::exp(static_cast<double>(sampling_pairs_[i].first - max_v));
                sampling_probs_[i] = static_cast<float>(e);
                sum += e;
            }
            if (sum > 0.0 && std::isfinite(sum)) {
                float cumulative_before = 0.0f;
                size_t keep = 0;
                for (size_t i = 0; i < sampling_pairs_.size(); ++i) {
                    if (cumulative_before > top_p) {
                        break;
                    }
                    cumulative_before += static_cast<float>(static_cast<double>(sampling_probs_[i]) / sum);
                    ++keep;
                }
                sampling_pairs_.resize((std::max)(size_t{1}, keep));
            }
        }

        if (sampling_pairs_.empty()) {
            return 0;
        }

        const float max_v = sampling_pairs_[0].first;
        double sum = 0.0;
        sampling_probs_.resize(sampling_pairs_.size());
        for (size_t i = 0; i < sampling_pairs_.size(); ++i) {
            const double e = std::exp(static_cast<double>(sampling_pairs_[i].first - max_v));
            sampling_probs_[i] = static_cast<float>(e);
            sum += e;
        }
        if (sum <= 0.0 || !std::isfinite(sum)) {
            const float uniform = 1.0f / static_cast<float>(sampling_pairs_.size());
            std::fill(sampling_probs_.begin(), sampling_probs_.end(), uniform);
        } else {
            for (size_t i = 0; i < sampling_probs_.size(); ++i) {
                sampling_probs_[i] = static_cast<float>(static_cast<double>(sampling_probs_[i]) / sum);
            }
        }

        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        const float target = dist(rng_);
        float cumulative = 0.0f;
        for (size_t i = 0; i < sampling_pairs_.size(); ++i) {
            cumulative += sampling_probs_[i];
            if (target <= cumulative) {
                return static_cast<int64_t>(sampling_pairs_[i].second);
            }
        }
        return static_cast<int64_t>(sampling_pairs_.back().second);
    }
    if (top_p > 0.0f && top_p < 1.0f) {
        sampling_pairs_.clear();
        for (size_t i = 0; i < N; ++i) {
            if (logits[i] > -std::numeric_limits<float>::infinity()) {
                sampling_pairs_.push_back({logits[i], i});
            }
        }
        if (!sampling_pairs_.empty()) {
            std::sort(sampling_pairs_.begin(), sampling_pairs_.end(),
                      [](const std::pair<float, size_t>& a, const std::pair<float, size_t>& b) {
                          return a.first > b.first;
                      });
            float max_v = sampling_pairs_[0].first;
            double sum = 0.0;
            sampling_probs_.resize(sampling_pairs_.size());
            for (size_t i = 0; i < sampling_pairs_.size(); ++i) {
                double e = std::exp(static_cast<double>(sampling_pairs_[i].first - max_v));
                sampling_probs_[i] = static_cast<float>(e);
                sum += e;
            }
            if (sum > 0.0 && std::isfinite(sum)) {
                for (size_t i = 0; i < sampling_pairs_.size(); ++i) {
                    sampling_probs_[i] = static_cast<float>(static_cast<double>(sampling_probs_[i]) / sum);
                }
            } else {
                float uniform = 1.0f / static_cast<float>(sampling_pairs_.size());
                std::fill(sampling_probs_.begin(), sampling_probs_.end(), uniform);
            }
            float cumulative_before = 0.0f;
            for (size_t i = 0; i < sampling_pairs_.size(); ++i) {
                if (cumulative_before > top_p) {
                    logits[sampling_pairs_[i].second] = -std::numeric_limits<float>::infinity();
                }
                cumulative_before += sampling_probs_[i];
            }
        }
    }
    
    float max_v = -std::numeric_limits<float>::infinity();
    for (float v : logits) {
        if (v > max_v) max_v = v;
    }
    double sum = 0.0;
    sampling_probs_.resize(N);
    for (size_t i = 0; i < N; ++i) {
        double e = std::exp(static_cast<double>(logits[i] - max_v));
        sampling_probs_[i] = static_cast<float>(e);
        sum += e;
    }
    if (sum <= 0.0 || !std::isfinite(sum)) {
        float uniform = N == 0 ? 0.0f : 1.0f / static_cast<float>(N);
        std::fill(sampling_probs_.begin(), sampling_probs_.end(), uniform);
    } else {
        for (size_t i = 0; i < N; ++i) {
            sampling_probs_[i] = static_cast<float>(static_cast<double>(sampling_probs_[i]) / sum);
        }
    }
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    const float target = dist(rng_);
    float cumulative = 0.0f;
    for (size_t i = 0; i < N; ++i) {
        cumulative += sampling_probs_[i];
        if (target <= cumulative) {
            return static_cast<int64_t>(i);
        }
    }
    return N == 0 ? 0 : static_cast<int64_t>(N - 1);
}
