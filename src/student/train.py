"""Fine-tune the frozen student with a LoRA adapter (the base weights never change).

The loss is computed ONLY on the target tool call: the model runs over the whole prompt (system,
tools, history: 6–9k tokens) but logits are projected for the last few dozen positions only
(`logits_to_keep`). Projecting all of them would need a 7k x 248k logit array (~8 GB, over the
Metal buffer limit on this M1).

    run = train(TrainConfig(dataset="<id>", iters=300))      -> data/student/runs/<run>/
"""
from __future__ import annotations

import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from student import dataset as ds
from student.context import call_message
from student.paths import FROZEN_DIR, RUNS_DIR


@dataclass
class TrainConfig:
    dataset: str
    iters: int = 300
    lr: float = 1e-4
    warmup: int = 20
    rank: int = 8
    alpha: float = 16.0
    dropout: float = 0.0
    grad_accum: int = 4             # examples per optimizer step
    max_tokens: int = 9000          # longer examples are skipped
    eval_every: int = 50
    eval_examples: int = 40
    seed: int = 0
    base: str = str(FROZEN_DIR)
    note: str = ""                  # why this run (the teacher writes it)


def encode(tok, messages: list[dict], tools: list[dict], name: str, args: dict) -> tuple[list[int], int]:
    """(token ids of prompt + target, number of target tokens at the end)."""
    prompt = tok.apply_chat_template(messages, tools=tools, add_generation_prompt=True, tokenize=False,
                                     enable_thinking=False)
    full = tok.apply_chat_template(messages + [call_message(name, args)], tools=tools, add_generation_prompt=False,
                                   tokenize=False, enable_thinking=False)
    if not full.startswith(prompt):
        raise ValueError("chat template: the prompt is not a prefix of prompt + answer")
    p_ids = tok.encode(prompt, add_special_tokens=False)
    f_ids = tok.encode(full, add_special_tokens=False)
    if f_ids[:len(p_ids)] != p_ids:
        raise ValueError("tokenization of the prompt changed when the answer was appended")
    return f_ids, len(f_ids) - len(p_ids)


def train(cfg: TrainConfig, log=print) -> Path:
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    from mlx_vlm import load
    from mlx_vlm.trainer import grad_checkpoint, prepare_model_for_training, save_adapter

    train_rows, valid_rows, tools = ds.load(cfg.dataset)
    model, processor = load(cfg.base)
    tok = processor.tokenizer
    model = prepare_model_for_training(model, train_type="lora", lora_rank=cfg.rank, lora_alpha=cfg.alpha,
                                       lora_dropout=cfg.dropout, verbose=False)
    lm = model.language_model
    grad_checkpoint(lm.model.layers[0])

    def encoded(rows):
        out, skipped = [], 0
        for r in rows:
            ids, n = encode(tok, r["messages"], tools, r["name"], r["args"])
            if len(ids) > cfg.max_tokens:
                skipped += 1
                continue
            out.append((mx.array(ids), n))
        return out, skipped

    t0 = time.perf_counter()
    train_set, skipped = encoded(train_rows)
    valid_set, _ = encoded(valid_rows)
    log(f"{len(train_set)} train / {len(valid_set)} valid examples ({skipped} longer than {cfg.max_tokens} skipped), "
        f"encoded in {time.perf_counter() - t0:.0f} s")

    def loss_fn(m, ids, n):
        logits = m.language_model(ids[None], logits_to_keep=n + 1).logits[:, :-1].astype(mx.float32)
        return nn.losses.cross_entropy(logits, ids[None, -n:]).mean()

    def valid_loss() -> float:
        rows = valid_set[:cfg.eval_examples]
        return float(sum(loss_fn(model, ids, n).item() for ids, n in rows) / max(len(rows), 1))

    run = RUNS_DIR / time.strftime("%Y%m%d-%H%M%S")
    run.mkdir(parents=True, exist_ok=True)
    (run / "config.json").write_text(json.dumps(asdict(cfg), indent=1) + "\n", encoding="utf-8")
    logf = (run / "log.jsonl").open("w", encoding="utf-8")

    schedule = optim.join_schedules([optim.linear_schedule(1e-7, cfg.lr, cfg.warmup),
                                     optim.cosine_decay(cfg.lr, max(cfg.iters - cfg.warmup, 1))], [cfg.warmup])
    opt = optim.AdamW(learning_rate=schedule, weight_decay=0.0)
    value_and_grad = nn.value_and_grad(model, loss_fn)
    rng = random.Random(cfg.seed)
    order = []
    v0 = valid_loss()
    log(f"step 0 · valid loss {v0:.3f}")
    logf.write(json.dumps({"step": 0, "valid_loss": v0}) + "\n")
    t0, running = time.perf_counter(), []
    for step in range(1, cfg.iters + 1):
        grads_sum, loss_sum = None, 0.0
        for _ in range(cfg.grad_accum):
            if not order:
                order = list(range(len(train_set)))
                rng.shuffle(order)
            ids, n = train_set[order.pop()]
            loss, grads = value_and_grad(model, ids, n)
            grads_sum = grads if grads_sum is None else _tree_add(grads_sum, grads)
            loss_sum += loss.item()
        opt.update(model, _tree_scale(grads_sum, 1 / cfg.grad_accum))
        mx.eval(model.trainable_parameters(), opt.state)
        running.append(loss_sum / cfg.grad_accum)
        if step % 10 == 0 or step == cfg.iters:
            rec = {"step": step, "train_loss": sum(running) / len(running), "s_per_step": (time.perf_counter() - t0) / step,
                   "peak_gb": mx.get_peak_memory() / 1e9}
            if step % cfg.eval_every == 0 or step == cfg.iters:
                rec["valid_loss"] = valid_loss()
            log(" · ".join(f"{k} {v:.3f}" if isinstance(v, float) else f"{k} {v}" for k, v in rec.items()))
            logf.write(json.dumps(rec) + "\n")
            logf.flush()
            running = []
    save_adapter(model, run / "adapters.safetensors")
    logf.close()
    return run


def _tree_add(a, b):
    from mlx.utils import tree_map
    return tree_map(lambda x, y: x + y, a, b)


def _tree_scale(a, s):
    from mlx.utils import tree_map
    return tree_map(lambda x: x * s, a)
