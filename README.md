# VieNeu-TTS.cpp

Native C++ runtime for [VieNeu-TTS](https://github.com/pnnbao97/VieNeu-TTS), focused on local Vietnamese text-to-speech inference, voice presets, and zero-shot voice cloning.

[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Models-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/pnnbao-ump/VieNeu-TTS-v3-Turbo) [![GitHub](https://img.shields.io/badge/GitHub-Original%20VieNeu--TTS-181717?logo=github)](https://github.com/pnnbao97/VieNeu-TTS) [![llama.cpp](https://img.shields.io/badge/llama.cpp-ggml-0052cc)](https://github.com/ggml-org/llama.cpp) [![ONNX Runtime](https://img.shields.io/badge/ONNX%20Runtime-Inference-blue)](https://onnxruntime.ai/)

## Highlights

- Native C++17 implementation with a small CLI and a stable C ABI.
- VieNeu v3 ONNX pipeline for convenient local inference.
- VieNeu v3 native path using `llama.cpp` for the backbone and native acoustic weights.
- Voice presets through `voices_v3_turbo.json`.
- Zero-shot voice cloning from a reference WAV file.
- Windows helper scripts for building, downloading runtime assets, and running smoke tests.

## Models

Download the model assets from Hugging Face:

- VieNeu v3 Turbo: [pnnbao-ump/VieNeu-TTS-v3-Turbo](https://huggingface.co/pnnbao-ump/VieNeu-TTS-v3-Turbo)
- MOSS audio tokenizer ONNX codec: [OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano-ONNX](https://huggingface.co/OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano-ONNX)

The convenience script `scripts/run-v3-tts-test.ps1` downloads the required v3 ONNX assets into `.models/vieneu-v3-turbo` automatically. Native v3 assets such as `backbone.gguf`, `vieneu_v3_heads.npz`, and `acoustic/vieneu_acoustic_weights.npz` can be exported with the scripts in `scripts/`.

## Build

Requirements:

- C++17 compiler: MSVC, GCC, or Clang
- CMake 3.14+
- `llama.cpp` submodule at `third_party/llama.cpp`
- ONNX Runtime SDK

On Windows, the easiest path is:

```powershell
scripts\build-local.ps1
```

Useful options:

```powershell
scripts\build-local.ps1 -Clean
scripts\build-local.ps1 -NoPackage
scripts\build-local.ps1 -OnnxRuntimeVersion 1.24.4
```

Manual CMake build:

```bash
cmake -S . -B build -DONNXRUNTIME_ROOT="C:/path/to/onnxruntime" -DVIENEU_LLAMA_DIR="third_party/llama.cpp"
cmake --build build --config Release
```

Build outputs include:

- `vieneu-tts-core`: static runtime library
- `vieneu-tts`: shared library exposing the C ABI
- `vieneu-tts-cli`: command-line synthesizer
- `test-abi-c`: C ABI compile test

## Quick Start

Run a VieNeu v3 ONNX smoke test:

```powershell
scripts\run-v3-tts-test.ps1 `
  -Text "Xin chào, đây là bài kiểm tra VieNeu TTS v3 Turbo từ runtime C++." `
  -Output outputs\vieneu-v3-test.wav
```

Run the CLI directly after building:

```powershell
build\Release\vieneu-tts-cli.exe `
  --profile vieneu-v3-onnx `
  --model-dir .models\vieneu-v3-turbo `
  --onnx-dir .models\vieneu-v3-turbo\onnx `
  --codec-dir .models\vieneu-v3-turbo\codec `
  --voices-json .models\vieneu-v3-turbo\voices_v3_turbo.json `
  --text "Xin chào, tôi là giọng nói tiếng Việt được tổng hợp bằng C++." `
  --output outputs\hello.wav
```

Run native v3 voice cloning benchmark with the included reference clips:

```powershell
scripts\run-v3-native-voice-clone-benchmark.ps1 -NoBuild
```

Run one reference only:

```powershell
scripts\run-v3-native-voice-clone-benchmark.ps1 -NoBuild -Ref example_3
```

Generated WAV files are written to `outputs/` by default.

## CLI Profiles

`vieneu-tts-cli` supports these profiles:

- `vieneu-v3-onnx`: v3 model assets plus ONNX Runtime sessions.
- `vieneu-v3-native`: v3 model directory with native GGUF/backbone assets and MOSS codec.
- `vieneu-v2-turbo`: legacy GGUF speech LM path with ONNX encoder/decoder assets.

Common options:

```text
--profile NAME
--model-dir PATH
--onnx-dir PATH
--codec-dir PATH
--voices-json PATH
--ref-audio PATH
--text TEXT
--voice ID
--output PATH
--temperature VALUE
--top-k VALUE
--top-p VALUE
--max-new-frames N
--max-chars N
--threads N
```

## Project Layout

```text
src/
  vieneu/        Stable C ABI and VieNeu v2/v3 runtime orchestration
  codecs/        ONNX Runtime helpers for audio codec models
  backends/      llama.cpp wrappers for GGUF/backbone inference
tools/
  vieneu-tts-cli.cpp
scripts/
  build-local.ps1
  run-v3-tts-test.ps1
  run-v3-native-preset-benchmark.ps1
  run-v3-native-voice-clone-benchmark.ps1
  export-v3-native-assets.py
  export-v3-acoustic-weights.py
examples/
  audio_ref/     Reference clips and transcript manifest for voice cloning
docs/
  v3-native-cpp-pipeline.md
```

## C ABI

The C API is defined in [`src/vieneu/vieneu_tts.h`](src/vieneu/vieneu_tts.h). It is intended for wrappers and host applications that want to embed VieNeu-TTS.cpp without depending on C++ symbols.

Minimal shape:

```c
#include "vieneu/vieneu_tts.h"
#include <stdio.h>

int main(void) {
    struct vieneu_init_params_v2 init;
    vieneu_init_v2_default_params(&init);
    init.profile = "vieneu-v3-onnx";
    init.model_dir = ".models/vieneu-v3-turbo";
    init.onnx_dir = ".models/vieneu-v3-turbo/onnx";
    init.codec_dir = ".models/vieneu-v3-turbo/codec";
    init.voices_json_path = ".models/vieneu-v3-turbo/voices_v3_turbo.json";

    struct vieneu_context *ctx = vieneu_init_v2(&init);
    if (!ctx) {
        fprintf(stderr, "init failed: %s\n", vieneu_last_error());
        return 1;
    }

    struct vieneu_tts_params_v2 tts;
    vieneu_tts_v2_default_params(&tts);
    tts.text = "Xin chào các bạn.";
    tts.temperature = 0.8f;
    tts.top_k = 25;

    struct vieneu_audio audio;
    if (vieneu_synthesize_v2(ctx, &tts, &audio) != 0) {
        fprintf(stderr, "synthesis failed: %s\n", vieneu_last_error());
        vieneu_free(ctx);
        return 1;
    }

    printf("Generated %d samples at %d Hz\n", audio.n_samples, audio.sample_rate);
    vieneu_audio_free(&audio);
    vieneu_free(ctx);
    return 0;
}
```

## Notes

- Keep large model files out of Git. Store them under `.models/` or download them from Hugging Face.
- The included `.models/`, `build/`, `dist/`, and `outputs/` directories are local working directories.
- For implementation details of the v3 native path, see [`docs/v3-native-cpp-pipeline.md`](docs/v3-native-cpp-pipeline.md).

## Credits

Special thanks to [pnnbao97](https://github.com/pnnbao97) and the original [pnnbao97/VieNeu-TTS](https://github.com/pnnbao97/VieNeu-TTS) project for the Vietnamese TTS model and reference implementation that this C++ runtime is based on.

Thanks also to the authors and maintainers of [llama.cpp](https://github.com/ggml-org/llama.cpp), [ONNX Runtime](https://onnxruntime.ai/), and [MOSS Audio Tokenizer](https://huggingface.co/OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano-ONNX).

## License

This project is licensed under the MIT License. See [`LICENSE`](LICENSE) for details.
