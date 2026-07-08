#!/usr/bin/env python
r"""Export VieNeu v3 Turbo acoustic decoder weights for the native C++ backend.

Example:
  python scripts/export-v3-acoustic-weights.py ^
    --source-repo E:\dduongtrandai-github\VieNeu-TTS ^
    --checkpoint pnnbao-ump/VieNeu-TTS-v3-Turbo ^
    --output .models\vieneu-v3-turbo\acoustic\vieneu_acoustic_weights.npz
"""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path

import numpy as np


def _array(tensor):
    return tensor.detach().cpu().float().contiguous().numpy().astype(np.float32, copy=False)


def _savez_stored(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, mode="w", compression=zipfile.ZIP_STORED, allowZip64=True) as zf:
        for name, arr in arrays.items():
            import io

            buf = io.BytesIO()
            np.save(buf, arr, allow_pickle=False)
            zf.writestr(f"{name}.npy", buf.getvalue())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-repo", default=r"E:\dduongtrandai-github\VieNeu-TTS", help="Path to the Python VieNeu-TTS repo.")
    parser.add_argument("--checkpoint", default="pnnbao-ump/VieNeu-TTS-v3-Turbo", help="HF repo id or local checkpoint directory.")
    parser.add_argument("--output", default=r".models\vieneu-v3-turbo\acoustic\vieneu_acoustic_weights.npz", help="Output NPZ path.")
    parser.add_argument("--hf-token", default=None, help="Optional Hugging Face token.")
    parser.add_argument("--subfolder", default="update", help="Checkpoint subfolder to load.")
    args = parser.parse_args()

    source_repo = Path(args.source_repo).resolve()
    src_dir = source_repo / "src"
    if not src_dir.exists():
        raise SystemExit(f"Cannot find Python source directory: {src_dir}")
    sys.path.insert(0, str(src_dir))

    from vieneu._v3_turbo_engine.configuration_v3_turbo import VieNeuV3TurboConfig
    from vieneu._v3_turbo_engine.hub_load_v3_turbo import load_v3_turbo_checkpoint

    cfg = VieNeuV3TurboConfig.from_pretrained(args.checkpoint, token=args.hf_token, subfolder=args.subfolder or "")
    model = load_v3_turbo_checkpoint(args.checkpoint, token=args.hf_token, device="cpu", dtype="float32", subfolder=args.subfolder or None).eval()
    dec = model.acoustic_decoder

    if int(getattr(cfg, "hidden_size")) <= 0 or int(getattr(cfg, "n_vq")) <= 0:
        raise SystemExit("Invalid v3 acoustic config loaded from checkpoint.")

    arrays: dict[str, np.ndarray] = {
        "slot_pos_emb": _array(dec.slot_pos_emb.weight),
        "norm": _array(dec.norm.weight),
    }

    for i, layer in enumerate(dec.layers):
        prefix = f"layers.{i}."
        arrays[prefix + "norm1"] = _array(layer.norm1.weight)
        arrays[prefix + "attn.qkv"] = _array(layer.attn.qkv.weight)
        arrays[prefix + "attn.q_norm"] = _array(layer.attn.q_norm.weight)
        arrays[prefix + "attn.k_norm"] = _array(layer.attn.k_norm.weight)
        arrays[prefix + "attn.o_proj"] = _array(layer.attn.o_proj.weight)
        arrays[prefix + "norm2"] = _array(layer.norm2.weight)
        arrays[prefix + "ff_up"] = _array(layer.ff_up.weight)
        arrays[prefix + "ff_gate"] = _array(layer.ff_gate.weight)
        arrays[prefix + "ff_down"] = _array(layer.ff_down.weight)

    _savez_stored(Path(args.output), arrays)
    print(f"Exported {len(arrays)} tensors to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
