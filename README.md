# vieneu-tts.cpp

A native, high-performance C++ inference engine for the VieNeu-TTS model family.

`vieneu-tts.cpp` packages llama.cpp (for v2 GGUF Speech LM inference) and ONNX Runtime (for neural encoder/decoder and v3 single-utterance inference) into a unified C++ orchestration layer. It exposes a clean, stable **C ABI** (`vieneu_tts.h`) that allows easy integration into other languages/applications (such as LA Studio, Python wrappers, or Rust bindings).

---

## Key Features
- **High-Performance Inference**: Tailored for CPU and GPU acceleration by combining `llama.cpp`'s matrix operations and `ONNX Runtime`.
- **Stable C ABI**: Clean API surface designed for cross-language compatibility (pure C99 compatible).
- **Dynamic Model Loading**: Dynamically initialize pipelines with different combinations of GGUF model files, ONNX encoders, and decoders.
- **Zero-Shot Voice Cloning**: Encode speaker embeddings from a reference audio file dynamically to perform instant cloning.
- **Low Footprint**: Minimal overhead, low memory utilization, and multi-thread optimizations.

---

## Directory Structure
- `src/`
  - `vieneu/`: Entry points implementing the stable C ABI wrapper (`vieneu_tts.h` / `vieneu_tts.cpp`) and VieNeu profiles/engines.
  - `codecs/`: ONNX Runtime helpers for neural speech encoders/decoders (e.g. NeuCodec).
  - `backends/`: Wrap `llama.cpp` for parsing GGUF models and generating speech tokens.
- `tools/`
  - `vieneu-tts-cli.cpp`: A command-line utility for standalone TTS generation and smoke tests.
- `tests/`
  - `abi-c.c`: A C99 source file to verify that the C ABI header can compile cleanly without C++ extensions.

---

## Prerequisites & Dependencies

To build `vieneu-tts.cpp`, you will need:
- A C++17 compatible compiler (MSVC, GCC, or Clang)
- CMake (version 3.14 or higher)
- **llama.cpp**: Managed as a Git submodule at `third_party/llama.cpp`, or customize its path using `-DVIENEU_LLAMA_DIR`.
- **ONNX Runtime**: The prebuilt C/C++ SDK library and headers.

---

## Build Instructions

### Easy Local Build (Windows)

We provide a helper PowerShell script [build-local.ps1](build-local.ps1) in the root directory that initializes the `llama.cpp` submodule, downloads ONNX Runtime SDK, sets up MSVC compiler paths, configures CMake, builds the project, and packages it.

To build the project locally, open a PowerShell console and run:

```powershell
.\build-local.ps1
```

**Parameters supported by `build-local.ps1`:**
- `-OnnxRuntimeVersion <string>`: Specify ONNX Runtime SDK version to download (default: `1.24.4`).
- `-Clean`: Clean the `build` directory before configuring CMake.
- `-NoPackage`: Build the project but do not create the final zip package.
- `-LlamaCppRepo <string>`: Custom fallback Git repository URL for `llama.cpp` if submodule initialization is unavailable.
- `-Generator <string>`: CMake generator to use (default: `Ninja`).

### Manual Build via CMake (CLI)

```bash
mkdir build
cd build

# On Windows (MSVC + ONNX Runtime)
cmake .. -DONNXRUNTIME_ROOT="C:/path/to/onnxruntime" -DVIENEU_LLAMA_DIR="../third_party/llama.cpp"

# Build the project
cmake --build . --config Release
```

This will compile:
1. `vieneu-tts-core` (Static Library)
2. `vieneu-tts` (`.dll` on Windows, `.so` on Unix) (Shared Library containing the C ABI)
3. `vieneu-tts-cli` (Command-line tool)
4. `test-abi-c` (ABI test executable)

---

## C ABI API Documentation

The C ABI is defined in [src/vieneu/vieneu_tts.h](src/vieneu/vieneu_tts.h).

### Principal Structures

- **`vieneu_init_params`**: Parameters passed during initialization:
  - `model_path`: Path to the GGUF model file.
  - `encoder_path`: Path to ONNX speaker encoder (optional; needed for cloning).
  - `decoder_path`: Path to ONNX neural decoder (required).
  - `voices_json_path`: Path to voice presets database JSON (optional).
  - `n_threads`: Number of CPU threads to allocate for inference.
  - `n_gpu_layers`: Number of layers to offload to GPU.

- **`vieneu_tts_params`**: Parameters used during synthesis:
  - `text`: Input text string to synthesize.
  - `voice_id`: Preset voice ID to load from `voices.json`.
  - `voice_embedding`: Pointer to a custom 128-dimensional speaker embedding.
  - `temperature`: Control randomness of generation (recommended: `0.3` - `0.5`).
  - `top_k`: Token vocabulary restriction threshold.

- **`vieneu_audio`**: Return structure for synthesized audio:
  - `samples`: Pointer to float PCM array (`[-1.0f, 1.0f]`).
  - `n_samples`: Size of the samples array.
  - `sample_rate`: Sample rate of generated audio (e.g. `24000`).

---

## Usage Examples

### 1. Standalone CLI Usage (`vieneu-tts-cli`)

Use the compiled CLI tool to synthesize text into a WAV file:

```bash
./vieneu-tts-cli \
  --profile vieneu-v2-turbo \
  --model "path/to/vieneu-tts-v2-turbo.gguf" \
  --decoder "path/to/vieneu_decoder.onnx" \
  --encoder "path/to/vieneu_encoder.onnx" \
  --voices-json "path/to/voices.json" \
  --voice "vi-female-preset" \
  --text "Xin chào, tôi là mô hình tiếng nói trí tuệ nhân tạo." \
  --output "hello.wav"
```

### VieNeu v3 Native Voice Cloning Test

The repo includes the same reference clips used by the original VieNeu-TTS UI in
`examples/audio_ref`. The script uses `scripts\v3-native-benchmark-text.txt` by
default, enables native benchmark logging, and writes one output per reference.
After building with `scripts/build-local.ps1`, run:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run-v3-native-voice-clone-test.ps1 -NoBuild
```

To test one reference only:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run-v3-native-voice-clone-test.ps1 -NoBuild -Ref example_3
```

Generated WAV files are written to `outputs\voice-clone` by default.
Benchmark logs are written to `outputs\voice-clone\logs`. Use `-Text "..."` or
`-TextFile path\to\text.txt` to override the benchmark text.

### 2. C ABI Integration Code Example

Below is a minimal C snippet demonstrating how to load the DLL/so and generate audio:

```c
#include "vieneu/vieneu_tts.h"
#include <stdio.h>
#include <stdlib.h>

int main(void) {
    // 1. Configure initialization params
    struct vieneu_init_params init_p;
    vieneu_init_default_params(&init_p);
    init_p.model_path = "vieneu-tts-v2-turbo.gguf";
    init_p.decoder_path = "vieneu_decoder.onnx";
    init_p.n_threads = 4;

    // 2. Initialize the VieNeu context
    struct vieneu_context *ctx = vieneu_init(&init_p);
    if (!ctx) {
        fprintf(stderr, "Initialization failed: %s\n", vieneu_last_error());
        return 1;
    }

    // 3. Set TTS generation parameters
    struct vieneu_tts_params tts_p;
    vieneu_tts_default_params(&tts_p);
    tts_p.text = "Xin chào các bạn.";
    tts_p.temperature = 0.4f;

    // 4. Synthesize speech
    struct vieneu_audio audio;
    if (vieneu_synthesize(ctx, &tts_p, &audio) != 0) {
        fprintf(stderr, "Synthesis failed: %s\n", vieneu_last_error());
        vieneu_free(ctx);
        return 1;
    }

    printf("Generated %d samples at %d Hz successfully!\n", 
           audio.n_samples, audio.sample_rate);

    // 5. Clean up allocated resources
    vieneu_audio_free(&audio);
    vieneu_free(ctx);
    return 0;
}
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
