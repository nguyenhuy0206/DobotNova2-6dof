"""The kernel API: the only way the teacher (Astra) touches the student's weights.

Astra's python tool imports exactly the names in __all__. It never runs the model: it reads
tensors, writes edited tensors into the work folder, and asks the runner (student.weights.runner)
for a forward pass with a fixed budget.

Weights: the frozen student (student.paths.FROZEN_DIR, Qwen3.5-4B 4-bit). Linear layers are
stored quantized (weight + scales + biases); `load` gives them back dequantized as float32
numpy arrays with the shape the layer uses (out, in). An edited tensor is saved in full precision and
the runner puts it into the model as a float layer, so small edits are not lost to re-quantization.

Rules (as in MIT exp8):
  * no global knobs: the token embedding (also the output head, tied), the final norm and the vision
    tower are out of scope
  * no whole-tensor scaling (W' = c·W)
  * the runner refuses an edited model that changes more than MAX_CHANGED_FRACTION of the language
    model's weights
"""
from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

import numpy as np

from student.paths import FROZEN_DIR, STUDENT_DIR

WORK = STUDENT_DIR / "weightlab" / "work"
EDITED = WORK / "edited"
FORBIDDEN = {"language_model.model.embed_tokens.weight", "language_model.model.norm.weight"}
FORBIDDEN_PREFIX = ("vision_tower.",)
MAX_CHANGED_FRACTION = 0.005

__all__ = ["NAMES", "info", "load", "load_original", "save_edit", "drop_edit", "edited", "ablate_direction",
           "MAX_CHANGED_FRACTION", "WORK"]


@lru_cache(maxsize=1)
def _raw() -> dict:
    """Every tensor of the frozen model (one file or several shards)."""
    import mlx.core as mx
    out = {}
    for f in sorted(FROZEN_DIR.glob("*.safetensors")):
        out.update(mx.load(str(f)))
    return out


@lru_cache(maxsize=1)
def text_config() -> dict:
    """num_hidden_layers, hidden_size, … of the language model."""
    cfg = json.loads((FROZEN_DIR / "config.json").read_text(encoding="utf-8"))
    return cfg.get("text_config") or cfg


@lru_cache(maxsize=1)
def _quant() -> dict:
    cfg = json.loads((FROZEN_DIR / "config.json").read_text(encoding="utf-8"))
    return cfg.get("quantization") or {"group_size": 64, "bits": 8}


def _names() -> list[str]:
    return sorted(k for k in _raw() if not k.endswith((".scales", ".biases")))


NAMES = _names()


def _quantized(name: str) -> bool:
    return name.endswith(".weight") and f"{name[:-len('.weight')]}.scales" in _raw()


def info(name: str) -> dict:
    """{shape, quantized, in_scope, edited} of one tensor (shape as `load` returns it)."""
    raw, q = _raw(), _quantized(name)
    if q:
        w = raw[name]
        shape = (w.shape[0], w.shape[1] * 32 // _quant()["bits"])
    else:
        shape = tuple(raw[name].shape)
    return {"shape": list(shape), "quantized": q, "in_scope": _in_scope(name), "edited": name in edited()}


def load_original(name: str) -> np.ndarray:
    import mlx.core as mx
    raw = _raw()
    if name not in raw:
        raise KeyError(f"unknown tensor {name}")
    if _quantized(name):
        base = name[:-len(".weight")]
        q = _quant()
        w = mx.dequantize(raw[name], raw[f"{base}.scales"], raw[f"{base}.biases"], group_size=q["group_size"],
                          bits=q["bits"])
    else:
        w = raw[name]
    return np.array(w.astype(mx.float32))


def load(name: str) -> np.ndarray:
    """The current tensor: your edit if there is one, else the original."""
    p = EDITED / f"{name}.npy"
    return np.load(p) if p.exists() else load_original(name)


def _in_scope(name: str) -> bool:
    return name not in FORBIDDEN and not name.startswith(FORBIDDEN_PREFIX)


def save_edit(name: str, tensor) -> str:
    if name not in NAMES:
        raise ValueError(f"unknown tensor {name}")
    if not _in_scope(name):
        raise PermissionError(f"{name} is out of scope")
    orig = load_original(name)
    t = np.asarray(tensor, dtype=np.float32)
    if t.shape != orig.shape or not np.isfinite(t).all():
        raise ValueError(f"shape {t.shape} != {orig.shape} or non-finite values")
    mask = np.abs(orig) > 1e-6
    ratio = t[mask] / orig[mask]
    if ratio.size and np.allclose(ratio, ratio.flat[0], rtol=1e-4):
        raise PermissionError("whole-tensor scaling is not allowed")
    EDITED.mkdir(parents=True, exist_ok=True)
    np.save(EDITED / f"{name}.npy", t)
    changed = float((t != orig).mean())
    return f"saved {name}: {changed:.4%} of its entries changed, max |delta| {float(np.abs(t - orig).max()):.3g}"


def drop_edit(name: str) -> None:
    p = EDITED / f"{name}.npy"
    if p.exists():
        os.remove(p)


def edited() -> list[str]:
    return sorted(p.name[:-4] for p in EDITED.glob("*.npy")) if EDITED.exists() else []


def ablate_direction(W, u, alpha: float = 1.0, cols=None) -> np.ndarray:
    """W' = W - alpha * u (u^T W) for W of shape (out, in) and u in the OUTPUT space (len out):
    removes the component along u from what the layer writes. cols: only these input columns."""
    W = np.asarray(W, dtype=np.float32)
    u = np.asarray(u, dtype=np.float32)
    u = u / np.linalg.norm(u)
    proj = np.outer(u, u @ W)
    if cols is not None:
        mask = np.zeros(W.shape[1], dtype=np.float32)
        mask[np.asarray(cols)] = 1.0
        proj = proj * mask
    return W - alpha * proj


def _changed_fraction() -> tuple[float, int, int]:
    """(fraction, changed entries, total) over the language model's weights; used by the runner."""
    total = changed = 0
    for n in NAMES:
        if n.startswith("language_model."):
            total += int(np.prod(info(n)["shape"]))
    for n in edited():
        changed += int((load(n) != load_original(n)).sum())
    return changed / max(total, 1), changed, total
