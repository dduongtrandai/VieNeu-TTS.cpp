#!/usr/bin/env python
r"""Export VieNeu v3 Turbo semantic backbone and assets for the C++ native pipeline.

Example:
  python scripts/export-v3-native-assets.py \
    --model-dir .models/vieneu-v3-turbo \
    --safetensors C:\Users\daidu\.cache\huggingface\hub\models--pnnbao-ump--VieNeu-TTS-v3-Turbo\snapshots\d363ab07bbe11547528b3847386dc3d3273e5934\model.safetensors
"""

import argparse
import os
import sys
from pathlib import Path
import numpy as np
import safetensors.torch

# Add llama.cpp/gguf-py to path
sys.path.insert(0, str(Path(__file__).parent.parent / "third_party" / "llama.cpp" / "gguf-py"))
from gguf import GGUFWriter


def export_backbone_gguf(safetensors_path: Path, output_path: Path):
    sd = {k: v.float() for k, v in safetensors.torch.load_file(str(safetensors_path)).items()}

    print("Mapping semantic backbone tensors to GGUF...")
    
    # Qwen2/Qwen3 metadata
    hidden_size = 768
    num_layers = 12
    num_heads = 12
    num_kv_heads = 4
    intermediate_size = 3072
    text_vocab_size = sd["text_embeddings.weight"].shape[0]

    writer = GGUFWriter(str(output_path), "qwen3")
    writer.add_name("VieNeu v3 Turbo Backbone")
    writer.add_block_count(num_layers)
    writer.add_context_length(2048)
    writer.add_embedding_length(hidden_size)
    writer.add_feed_forward_length(intermediate_size)
    writer.add_head_count(num_heads)
    writer.add_head_count_kv(num_kv_heads)
    writer.add_layer_norm_rms_eps(1e-6)
    writer.add_rope_freq_base(10000.0)

    # Add a dummy vocabulary to satisfy llama.cpp loader
    dummy_tokens = [f"<t_{i}>" for i in range(text_vocab_size)]
    dummy_scores = [0.0] * text_vocab_size
    dummy_toktypes = [1] * text_vocab_size  # 1 = Normal
    writer.add_tokenizer_model("llama")
    writer.add_token_list(dummy_tokens)
    writer.add_token_scores(dummy_scores)
    writer.add_token_types(dummy_toktypes)

    # Map text embedding and Qwen layers
    # token_embd.weight
    text_emb = sd["text_embeddings.weight"].cpu().numpy()
    writer.add_tensor("token_embd.weight", text_emb)
    writer.add_tensor("output.weight", text_emb) # Tied

    # blk.{i}.*
    for i in range(num_layers):
        # Norms
        writer.add_tensor(f"blk.{i}.attn_norm.weight", sd[f"semantic_backbone.layers.{i}.input_layernorm.weight"].cpu().numpy())
        writer.add_tensor(f"blk.{i}.ffn_norm.weight", sd[f"semantic_backbone.layers.{i}.post_attention_layernorm.weight"].cpu().numpy())

        # QKV Projections
        writer.add_tensor(f"blk.{i}.attn_q.weight", sd[f"semantic_backbone.layers.{i}.self_attn.q_proj.weight"].cpu().numpy())
        writer.add_tensor(f"blk.{i}.attn_k.weight", sd[f"semantic_backbone.layers.{i}.self_attn.k_proj.weight"].cpu().numpy())
        writer.add_tensor(f"blk.{i}.attn_v.weight", sd[f"semantic_backbone.layers.{i}.self_attn.v_proj.weight"].cpu().numpy())
        writer.add_tensor(f"blk.{i}.attn_output.weight", sd[f"semantic_backbone.layers.{i}.self_attn.o_proj.weight"].cpu().numpy())

        # QK norms (if present in the model)
        qk_norm_keys = [
            (f"blk.{i}.attn_q_norm.weight", f"semantic_backbone.layers.{i}.self_attn.q_norm.weight"),
            (f"blk.{i}.attn_k_norm.weight", f"semantic_backbone.layers.{i}.self_attn.k_norm.weight"),
        ]
        for gguf_key, sf_key in qk_norm_keys:
            if sf_key in sd:
                writer.add_tensor(gguf_key, sd[sf_key].cpu().numpy())

        # MLP/FFN
        writer.add_tensor(f"blk.{i}.ffn_gate.weight", sd[f"semantic_backbone.layers.{i}.mlp.gate_proj.weight"].cpu().numpy())
        writer.add_tensor(f"blk.{i}.ffn_up.weight", sd[f"semantic_backbone.layers.{i}.mlp.up_proj.weight"].cpu().numpy())
        writer.add_tensor(f"blk.{i}.ffn_down.weight", sd[f"semantic_backbone.layers.{i}.mlp.down_proj.weight"].cpu().numpy())

    # Final norm
    writer.add_tensor("output_norm.weight", sd["semantic_backbone.norm.weight"].cpu().numpy())

    print(f"Writing GGUF file to {output_path}...")
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print("Backbone GGUF export complete.")


def export_heads_npz(safetensors_path: Path, output_path: Path):
    print(f"Exporting embedding heads from {safetensors_path}...")
    sd = {k: v.float() for k, v in safetensors.torch.load_file(str(safetensors_path)).items()}

    text_emb = sd["text_embeddings.weight"].cpu().numpy()
    
    # Pack audio embeddings [16, 1024, 768]
    audio_embeddings = []
    for ch in range(16):
        key = f"audio_embeddings.{ch}.weight"
        audio_embeddings.append(sd[key].cpu().numpy())
    audio_emb = np.stack(audio_embeddings, axis=0)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, text_emb=text_emb, audio_emb=audio_emb)
    print(f"Heads NPZ exported to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Export VieNeu v3 Turbo Native Assets")
    parser.add_argument("--model-dir", default=".models/vieneu-v3-turbo", help="Output model directory")
    parser.add_argument("--safetensors", required=True, help="Path to input model.safetensors")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    safetensors_path = Path(args.safetensors)

    if not safetensors_path.exists():
        print(f"Error: {safetensors_path} does not exist.")
        return 1

    model_dir.mkdir(parents=True, exist_ok=True)

    backbone_path = model_dir / "backbone.gguf"
    heads_path = model_dir / "vieneu_v3_heads.npz"

    export_backbone_gguf(safetensors_path, backbone_path)
    export_heads_npz(safetensors_path, heads_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
