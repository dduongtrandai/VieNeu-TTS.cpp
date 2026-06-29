#ifndef V3_NATIVE_ACOUSTIC_GGML_H
#define V3_NATIVE_ACOUSTIC_GGML_H

#include <vector>
#include <string>
#include <memory>
#include "v3_native_config.h"
#include "v3_native_assets.h"
#include "v3_native_sampler.h"
#include "../v3_common/v3_repetition_history.h"

class V3NativeAcoustic {
public:
    V3NativeAcoustic(const V3NativeConfig& config, const V3NativeAssets& assets, V3NativeSampler& sampler, int n_threads);
    ~V3NativeAcoustic();

    bool initialize(std::string& error);
    void reset_benchmark_stats();
    void print_benchmark_stats() const;

    // Generate a single acoustic frame (16 codes) and the EOS check
    bool generate_frame(
        const std::vector<float>& h,
        float temperature,
        int top_k,
        float top_p,
        float repetition_penalty,
        std::vector<V3RepetitionHistory>& history,
        std::vector<int64_t>& codes,
        bool& eos,
        std::string& error);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

#endif // V3_NATIVE_ACOUSTIC_GGML_H
