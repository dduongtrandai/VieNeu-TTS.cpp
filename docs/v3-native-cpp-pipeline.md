# VieNeu v3 Native C++ Pipeline

This document defines a new `vieneu-v3-native` runtime path. It is intentionally
separate from the existing `vieneu-v3-onnx` path and treats the PyTorch/GPU
implementation in `VieNeu-TTS` as the behavioral reference.

## Goal

Build a native C++ inference pipeline for VieNeu-TTS v3 Turbo with no ONNX
Runtime dependency in the hot path:

1. Text normalization, chunking, phonemization, tokenizer, prompt construction.
2. Qwen3 semantic backbone prefill/decode with KV cache.
3. Local acoustic decoder that autoregressively samples 16 RVQ codes per frame.
4. MOSS Audio Tokenizer Nano encode/decode for reference cloning and waveform
   output.
5. Streaming and batched serving paths modeled after the PyTorch/GPU runtime.
6. Optimized CPU execution for laptops without a discrete GPU.

## Reference PyTorch Pipeline

The reference execution order is:

```text
text
  -> normalize_to_chunks_v3
  -> phonemize_text_with_emotions
  -> build_prompt_2d(text phonemes + optional ref MOSS codes)
  -> _build_inputs_embeds(text embedding + per-channel audio embeddings)
  -> semantic_backbone prefill(inputs_embeds, use_cache=True)
  -> repeat per frame:
       acoustic_decoder.cached_step(global hidden, sgs text token, previous RVQ)
       sample 16 audio codebooks with top-k/top-p/temp/repetition penalty
       text_lm_head(slot0) decides speech EOS
       semantic_backbone decode_step(generated slot embedding, past_kv)
  -> MOSS decode(codes) -> 48 kHz mono waveform
```

Native C++ must preserve this shape. Replacing ONNX files with C++ wrappers is
not enough; the win comes from persistent GPU memory, reusable KV caches,
batched decode, fused small ops, and CUDA graph capture where shapes are stable.

## Proposed Runtime Layout

```text
src/vieneu/v3_native/
  vieneu_v3_native.h/.cpp          public engine orchestrator
  v3_native_config.h               config.json mapping
  v3_native_assets.h/.cpp          safetensors/GGUF/NPZ loading
  v3_native_tokenizer.h/.cpp       byte-BPE tokenizer.json loader
  v3_native_prompt.h/.cpp          prompt rows + chunking glue
  v3_native_sampler.h/.cpp         top-k/top-p/temp/repetition penalty
  v3_native_backbone_llama.cpp     Qwen3 backbone via llama.cpp input embeddings
  v3_native_acoustic_ggml.cpp      acoustic decoder via ggml/CUDA backend
  v3_native_moss_codec.cpp         MOSS codec native graph
  v3_native_stream.cpp             streaming chunk scheduler
  v3_native_batch.cpp              static/dynamic batching scheduler
```

The public profile should be named `vieneu-v3-native`. Keep
`vieneu-v3-onnx` available as a compatibility path, but do not share execution
classes with the new runtime.

## Native Backend Mapping

The native pipeline should be split into orchestration code and compute backends.
VieNeu-specific logic stays in C++; tensor execution is delegated to proven
native backends.

```text
VieNeu C++ orchestration
  text/chunk/phoneme/tokenizer/prompt/sampler/session state
  |
  +-- Qwen3 semantic backbone  -> llama.cpp / GGML / CUDA / cuBLAS
  +-- acoustic decoder         -> GGML graph or custom CUDA kernels
  +-- MOSS codec               -> GGML graph or custom CPU/CUDA kernels
  +-- sampling heads           -> CPU SIMD or CUDA kernels depending on device
```

### Qwen3 Semantic Backbone

Use `llama.cpp` as the runtime for the Qwen3 backbone.

Why:

- VieNeu v3 backbone is Qwen3-like: RMSNorm, RoPE, GQA, SiLU gated FFN.
- llama.cpp already supports Qwen-family models, GGUF loading, CPU SIMD, CUDA,
  Vulkan, Metal, KV cache, quantization, and batching.
- llama.cpp exposes `llama_batch.embd`, which is required because VieNeu v3 feeds
  summed text/audio embeddings, not plain token ids.

Execution:

```text
prompt rows
  -> VieNeu C++ embed_rows()
  -> llama_batch with embd != nullptr
  -> llama_decode()
  -> llama_get_embeddings_ith() or equivalent hidden output
  -> h_last
```

For each generated frame:

```text
codes[16]
  -> VieNeu C++ embed_generated_slot()
  -> llama_batch.embd with one token position
  -> llama_decode() using existing KV cache
  -> next h_last
```

Backend selection inside llama.cpp:

- CPU laptop: GGML CPU kernels, SIMD, quantized GGUF.
- NVIDIA GPU: llama.cpp CUDA backend, cuBLAS for matmul, CUDA kernels for
  attention/rope/norm where supported.
- AMD/Intel GPU: llama.cpp Vulkan backend if stable for the target.
- Apple: Metal backend.

### Embeddings And Output Heads

VieNeu v3 cannot rely on normal LLM token embedding lookup because every row has
one text token plus up to 16 audio-code tokens.

Keep this in VieNeu-native code:

```text
row_embedding =
    text_embeddings[text_id]
  + sum(audio_embeddings[channel][audio_code[channel]])
```

CPU:

- Store `text_emb` as `[text_vocab, hidden]`.
- Store `audio_emb` as `[n_vq, audio_vocab, hidden]`.
- Store transposed heads as `[n_vq, hidden, audio_vocab]` for fast matvec.
- Use persistent scratch buffers and SIMD-friendly loops.

GPU:

- Keep embeddings and heads resident on device.
- Build prompt/generated-slot embeddings with a small CUDA kernel.
- Compute codebook logits with cuBLAS GEMV/GEMM or a fused custom kernel.

### Acoustic Decoder

The acoustic decoder is small but called many times:

```text
per generated audio frame:
  1 local prefill over 2 tokens
  15 local cached steps
  16 output-head projections
  16 sampling calls
```

This is where tiny-op overhead matters most.

CPU implementation:

- Use one reusable GGML graph or hand-written fused kernels.
- Do not instantiate one graph per linear layer.
- Fuse at least:
  - RMSNorm + qkv projection input preparation
  - local attention softmax over the tiny local sequence
  - gated FFN intermediate handling
  - output-head matvec scratch reuse
- Keep local KV cache in contiguous `[layer, token, hidden]` or
  `[layer, head, token, head_dim]` layout.

GPU implementation:

- Option A: GGML CUDA backend for the acoustic graph.
- Option B, faster target: custom CUDA kernels plus cuBLAS.
- Use cuBLAS for large projections:
  - qkv: `H -> 3H`
  - FFN gate/up/down
  - audio/text heads
- Use custom CUDA kernels for:
  - RMSNorm
  - small causal local attention
  - SiLU multiply
  - top-k/top-p sampling
- Capture the fixed-shape acoustic frame with CUDA Graph when
  `batch_size`, `top_k`, and dtype are stable.

### Sampler

Sampling remains VieNeu-owned because it must match PyTorch behavior:

- temperature
- top-k
- top-p
- repetition penalty per codebook
- deterministic greedy path when `temperature <= 0`

CPU:

- Use partial top-k selection over 1024 audio vocab.
- Reuse probability buffers.
- Avoid sorting the full vocab unless top-p requires it.

GPU:

- For batch serving, run top-k/top-p per row/codebook on device.
- For batch size 1, CPU sampling may be acceptable initially, but it creates
  device sync overhead; final GPU path should keep sampling on device.

### MOSS Codec

MOSS is mandatory for a fully native path.

Decode path:

```text
generated codes (T, 16)
  -> MOSS native decoder
  -> stereo/mono waveform
  -> average channels to mono
  -> 48 kHz float32
```

Encode path for reference cloning:

```text
wav 48 kHz stereo/mono
  -> MOSS native encoder
  -> ref codes (T, 16)
```

CPU:

- GGML graph is preferred first because it gives portable CPU execution.
- Later, fuse convolution/residual blocks if profiling shows codec cost is high.

GPU:

- GGML CUDA or custom CUDA kernels.
- Keep codec decode on GPU if the acoustic codes are already on GPU.
- Transfer only final float waveform to CPU.

### Device Profiles

Recommended concrete profiles:

```text
cpu-portable:
  backbone: llama.cpp GGML CPU, portable build
  backbone weights: Q4_K_M / Q5_K_M
  acoustic: fused CPU GGML/fused C++
  codec: GGML CPU

cpu-native:
  backbone: llama.cpp GGML CPU, native SIMD build
  backbone weights: Q4_K_M / Q5_K_M
  acoustic: fused CPU kernels
  codec: GGML CPU

cuda:
  backbone: llama.cpp CUDA + cuBLAS
  backbone weights: fp16, q8, or suitable GGUF quant
  acoustic: custom CUDA/cuBLAS or GGML CUDA
  codec: GGML CUDA or custom CUDA

vulkan/metal:
  backbone: llama.cpp Vulkan/Metal
  acoustic: CPU first, then backend-specific graph when stable
  codec: CPU first, then backend-specific graph when stable
```

## Asset Format

The PyTorch checkpoint has custom components, so use a native asset pack instead
of ONNX:

```text
.models/vieneu-v3-native/
  config.json
  tokenizer.json
  voices_v3_turbo.json
  backbone.gguf            semantic_backbone Qwen3 weights
  heads.gguf or heads.npz  text embeddings and 16 audio embeddings/heads
  acoustic.gguf            local acoustic decoder weights
  moss_codec.gguf          MOSS encoder/decoder weights
```

`backbone.gguf` should be converted from `model.safetensors` using only
`semantic_backbone.*` tensors and Qwen3 metadata:

- `hidden_size = 768`
- `num_hidden_layers = 12`
- `num_attention_heads = 12`
- `num_key_value_heads = 4`
- `head_dim = 64`
- `intermediate_size = 3072`
- `rope_theta = 1000000.0`
- `rms_norm_eps = 1e-6`

The backbone must be driven with external embeddings, because VieNeu input rows
are not plain text tokens. llama.cpp supports `llama_batch.embd`; the native
engine should build prompt and generated-slot embeddings itself, then feed those
embeddings into the Qwen3 graph.

## Native Inference Loop

```cpp
prompt_rows = build_prompt_2d(phonemes, ref_codes, leading_token);
prompt_embeds = embed_rows(prompt_rows);        // text_emb + audio_emb sums

h = backbone.prefill(prompt_embeds);            // returns last hidden + KV

for frame in 0..max_new_frames:
    AcousticFrame frame = acoustic.generate_frame(
        h,
        speech_generation_start_token_id,
        sampling_params,
        repetition_history);

    frames.push_back(frame.codes);
    if (frame.eos) break;

    slot_embed = embed_generated_slot(frame.codes);
    h = backbone.decode_step(slot_embed);       // advances semantic KV

wav = moss_codec.decode(frames);
```

The acoustic `generate_frame` must mirror PyTorch:

1. Prefill two local tokens: global hidden and `sgs` text embedding.
2. Sample codebook 0 from hidden slot 1.
3. For codebooks 1..15, feed the previous codebook embedding with cached local
   KV and sample the next code.
4. Compute speech EOS from `text_lm_head(slot0)`.

## Dual Backend Strategy

The native runtime must support both:

- `cpu`: highest-priority path for laptops without a discrete GPU.
- `gpu`: acceleration path for CUDA/Vulkan/Metal-capable machines.

The default device policy should be:

```text
auto:
  use CUDA if available and explicitly built
  else use Vulkan/Metal if available and stable
  else use optimized CPU
```

CPU support is not a fallback afterthought. It is a first-class deployment target
and should be benchmarked on low-power laptop CPUs.

## CPU-Laptop Optimization Strategy

For laptops without a discrete GPU, the main risks are memory bandwidth, small-op
overhead, thread oversubscription, and thermal throttling. The CPU path should
therefore prefer fewer large fused kernels over many tiny graph launches.

Target CPU backend:

1. Use llama.cpp/ggml CPU for the Qwen3 semantic backbone.
   - Build with native SIMD enabled for local releases when possible.
   - Provide portable binaries separately.
   - Prefer quantized backbone weights for CPU: start with `Q4_K_M`/`Q5_K_M`,
     validate quality, then keep fp16/f32 as debug/reference modes.
   - Tune `n_threads` and `n_threads_batch` separately; on laptop CPUs, using all
     logical threads can be slower than using physical cores.
2. Keep embeddings, sampler, prompt rows, and generated slot construction native
   C++ with persistent buffers.
   - Avoid per-frame allocation.
   - Keep `text_emb` and `audio_emb` in cache-friendly contiguous layouts.
   - Precompute transposed output heads for fast matvec.
3. Implement acoustic decoder as a fused CPU graph/kernel, not per-linear graphs.
   - The current experimental acoustic ggml path creates many tiny operations and
     is not the final CPU design.
   - Fuse RMSNorm + qkv, local attention, FFN, output-head matvec where practical.
   - Batch the 16 codebook steps inside one executor call with reusable scratch.
4. Implement MOSS decode as a native CPU graph with persistent workspaces.
   - Preset voices only need decode.
   - Reference cloning additionally needs encode; this can ship after decode.
5. Add streaming decode for perceived latency.
   - Emit after a small lead-in, matching the PyTorch stream scheduler.
   - For CPU, choose chunk sizes that reduce codec invocations while keeping
     first-audio latency acceptable.

Recommended CPU defaults for laptop-class machines:

```text
backbone quantization: Q4_K_M or Q5_K_M
acoustic weights: fp16 or q8/fp16 mixed, benchmark both
threads: physical cores by default, not logical threads
batch: 1 for interactive use, small static batches for server mode
memory: mmap weights where possible, preallocate KV and scratch buffers
```

Acceptance for CPU:

- On a 4-core/8-thread laptop CPU, the native CPU path should beat the existing
  ONNX CPU path for the same text, voice, and sampling settings.
- `temperature=0` must match PyTorch deterministic code generation before quality
  and speed tuning.
- RTF, first-audio latency, peak memory, and sustained speed after thermal
  warm-up must be reported.

## GPU Optimization Strategy

The fastest native route is:

1. Use llama.cpp for the Qwen3 backbone first.
   - It already has CUDA, Metal, Vulkan, CPU, KV cache, input embeddings, flash
     attention controls, quantization support, and batching primitives.
   - This avoids writing a full transformer backend from scratch.
2. Implement acoustic decoder as a single persistent ggml graph or custom CUDA
   module.
   - Current experimental per-linear ggml implementation is not the target: it
     creates too many tiny operations and is CPU-oriented.
   - The target graph should fuse qkv, RMSNorm, attention, FFN, heads, and
     sampling where practical.
3. Implement MOSS codec natively.
   - This is mandatory for "no ONNX".
   - Decode is required for all synthesis; encode is required for reference
     voice cloning.
   - If time is constrained, ship preset voices first, then add codec encode.
4. Add batched serving after single-request parity.
   - Recreate `V3TurboBatchEngine`: prefill B prompts, run acoustic frame for B,
     EOS-mask finished rows, feed generated slots back to the backbone.
   - Add CUDA graph capture for fixed `(B, n_vq, hidden)` acoustic frame shapes.

## Milestones

### M1: Native Asset Export

Create `scripts/export-v3-native-assets.py`:

- Load `model.safetensors`.
- Export `semantic_backbone` to `backbone.gguf`.
- Export `text_embeddings`, `audio_embeddings`, and tied output heads.
- Export `acoustic_decoder` to `acoustic.gguf`.
- Export or convert MOSS Audio Tokenizer Nano to `moss_codec.gguf`.

Acceptance:

- A small asset inspector prints all tensor names, shapes, dtypes, and config.
- Shapes match the PyTorch config exactly.

### M2: Backbone Parity

Implement `V3NativeBackbone` using llama.cpp `llama_batch.embd`.

Acceptance:

- Feed PyTorch-exported prompt embeddings into C++ and Python.
- Compare last hidden state for prefill and one decode step.
- Target tolerance: fp32 CPU close enough for debug; fp16/bf16 GPU tolerance for
  production.

### M3: Acoustic Parity

Implement `V3NativeAcoustic`.

Acceptance:

- Given the same hidden vector and deterministic sampling (`temperature=0`),
  C++ emits the same 16 codes as PyTorch.
- EOS decision matches PyTorch.
- Batched shape `(B,H)` works before optimizing.

### M4: MOSS Codec Native Decode

Implement MOSS decode from `(T, n_vq)` to 48 kHz waveform.

Acceptance:

- Same input codes produce waveform close to PyTorch MOSS decode.
- Full preset-voice synthesis works without ONNX Runtime.

### M5: Reference Encode

Implement MOSS encode for `ref_audio_path`.

Acceptance:

- Encoded reference codes have same `(T, 16)` layout as PyTorch.
- Voice cloning path matches PyTorch behavior.

### M6: GPU Serving Optimizations

Add request batching, persistent buffers, pinned host memory, GPU-resident
embeddings, and CUDA graph capture for acoustic frame generation.

Acceptance:

- Single-request RTF improves over current ONNX CPU path on NVIDIA GPU.
- Batched throughput follows the PyTorch `v3_turbo_serve` path.

### M7: CPU Laptop Tuning

Optimize the CPU path after correctness parity:

- Quantize backbone GGUF and benchmark quality/speed tradeoffs.
- Replace per-op acoustic execution with fused CPU kernels or one reusable ggml
  graph.
- Tune thread defaults for physical-core laptop CPUs.
- Add sustained benchmark mode that runs multiple utterances to expose thermal
  throttling.

Acceptance:

- Native CPU path is faster than `vieneu-v3-onnx` on a laptop without dGPU.
- Speed remains stable across repeated runs.
- CLI prints enough timing breakdown to identify backbone/acoustic/codec cost.

## C ABI Changes

Add a new profile string:

```text
vieneu-v3-native
```

The existing `vieneu_init_params_v2` can be reused, but names should be
interpreted differently for the native profile:

- `model_dir`: native asset root.
- `onnx_dir`: ignored.
- `codec_dir`: optional override for native MOSS codec assets.
- `n_threads`: CPU fallback thread count.

Add future ABI fields only if needed:

- `device`: `auto`, `cpu`, `cuda`, `vulkan`, `metal`.
- `batch_size`: max active requests for serving.
- `gpu_layers`: for llama.cpp offload control.
- `cpu_threads`: generation threads.
- `cpu_batch_threads`: prompt/batch threads.
- `quantization`: preferred native asset quantization.

## Non-Goals

- Do not port the old ONNX graph runner.
- Do not make the acoustic-only ggml path the final architecture.
- Do not treat CPU as a slow compatibility fallback; laptop CPU is a first-class
  target.
- Do not change the public v2/v3 ONNX profiles while building this path.

## Immediate Implementation Order

1. Add `vieneu-v3-native` scaffolding and asset loader interfaces.
2. Build `export-v3-native-assets.py`.
3. Implement backbone prefill/decode with `llama_batch.embd` on CPU first.
4. Implement deterministic acoustic decode (`temperature=0`) for parity.
5. Add MOSS native decode.
6. Wire CLI/C ABI synthesis end to end.
7. Add GPU backends and batched serving after the CPU path is correct.
