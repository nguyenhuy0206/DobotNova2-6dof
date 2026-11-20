#!/usr/bin/env python3
"""The student model (local Qwen) taught by Astra: freeze, data, fine-tune, evaluate. No robot.

    .venv/bin/python scripts/train_student.py freeze                 read-only copy of the base model + checksums
    .venv/bin/python scripts/train_student.py verify                 the frozen copy is unchanged?
    .venv/bin/python scripts/train_student.py dataset                build training data from the episode logs (+ edits)
    .venv/bin/python scripts/train_student.py train DATASET [--iters 300 --lr 1e-4 --rank 8 --note "..."]
    .venv/bin/python scripts/train_student.py eval [--adapter data/student/runs/<id>] [--steps N]
    .venv/bin/python scripts/train_student.py compare base <run id>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("freeze")
    sub.add_parser("verify")
    d = sub.add_parser("dataset")
    d.add_argument("--window", type=int, default=None)
    t = sub.add_parser("train")
    t.add_argument("dataset")
    t.add_argument("--iters", type=int, default=300)
    t.add_argument("--lr", type=float, default=1e-4)
    t.add_argument("--rank", type=int, default=8)
    t.add_argument("--grad-accum", type=int, default=4)
    t.add_argument("--note", default="")
    e = sub.add_parser("eval")
    e.add_argument("--adapter", default=None)
    e.add_argument("--steps", type=int, default=0, help="at most this many steps per episode (0 = all)")
    c = sub.add_parser("compare")
    c.add_argument("a")
    c.add_argument("b")
    args = ap.parse_args()

    if args.cmd == "freeze":
        from student.freeze import freeze
        m = freeze()
        print(f"frozen {m['repo']}@{m['revision'][:8]}: {len(m['files'])} files")
    elif args.cmd == "verify":
        from student.freeze import verify
        print(json.dumps(verify(), indent=1))
    elif args.cmd == "dataset":
        from student.context import WINDOW
        from student.dataset import build
        print(json.dumps(build(window=args.window or WINDOW), indent=1))
    elif args.cmd == "train":
        from student.train import TrainConfig, train
        run = train(TrainConfig(dataset=args.dataset, iters=args.iters, lr=args.lr, rank=args.rank,
                                grad_accum=args.grad_accum, note=args.note))
        print("adapter:", run)
    elif args.cmd == "eval":
        from student.evaluate import evaluate
        s = evaluate(adapter=Path(args.adapter) if args.adapter else None, max_steps=args.steps)
        print(json.dumps({k: v for k, v in s.items() if k != "per_tool"}, indent=1))
    elif args.cmd == "compare":
        from student.evaluate import compare
        for k, a, b in compare(args.a, args.b):
            print(f"  {k:30s} {str(a):>12s}  {str(b):>12s}")


if __name__ == "__main__":
    main()
