"""Atomic, validated inference blocks and model-state fingerprints for crop studies."""
from __future__ import annotations

import hashlib
import json

import numpy as np


def signature(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def atomic_npz(path, **arrays):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def validate_probs(probs):
    if probs.ndim not in (2, 3) or not np.isfinite(probs).all():
        raise ValueError("invalid/non-finite probabilities")
    if np.any(probs < 0) or np.any(probs > 1) or not np.allclose(probs.sum(-1), 1, atol=2e-6):
        raise ValueError("probabilities must be in [0,1] and sum to one")


def load_blocks(folder, expected_signature, n, batch, n_views, n_classes):
    """Only a contiguous prefix with correct signature, bounds and finite scores can resume."""
    out, timings, peaks, done = [], [], [], 0
    for path in sorted(folder.glob("block_*.npz")):
        with np.load(path, allow_pickle=False) as block:
            start, end = int(block["start"]), int(block["end"])
            if (str(block["signature"]) != expected_signature or start != done
                    or end != min(start + batch, n) or path.name != f"block_{start:06d}.npz"):
                raise ValueError(f"{path}: incompatible or non-contiguous inference checkpoint")
            p = block["probs"]
            if p.shape != (end-start, n_views, n_classes):
                raise ValueError(f"{path}: unexpected probability shape")
            validate_probs(p)
            t, peak = block["seconds"], block["peak_bytes"]
            if t.shape != (n_views,) or peak.shape != t.shape or not np.isfinite(t).all() or (t < 0).any():
                raise ValueError(f"{path}: invalid profiling values")
            out.append(p.copy()); timings.append(t.copy()); peaks.append(peak.copy()); done = end
    return out, timings, peaks, done


def state_digest(state):
    """Includes all parameters AND buffers, including scalar BatchNorm counters."""
    import torch
    h = hashlib.sha256()
    for name, value in sorted(state.items()):
        t = value.detach().cpu().contiguous()
        h.update(json.dumps([name, str(t.dtype), list(t.shape)]).encode())
        h.update(t.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()
