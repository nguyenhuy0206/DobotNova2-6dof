#!/usr/bin/env python3
"""Astra edits the weights of the student (local Qwen, student.paths) through a kernel API. No robot.

    .venv/bin/python scripts/weight_lab.py data                 build dev (teacher) + test (held out) items
    .venv/bin/python scripts/weight_lab.py reset [--budget 40]  drop all edits, refill the forward budget
    .venv/bin/python scripts/weight_lab.py forward '{"type": "eval", "model": "original"}'   operator check
    .venv/bin/python scripts/weight_lab.py astra [--steps 80]   the teacher session (calls the API)
    .venv/bin/python scripts/weight_lab.py test                 held-out: original vs edited
    .venv/bin/python scripts/weight_lab.py status
"""
from __future__ import annotations

import argparse
import json

import _bootstrap  # noqa: F401


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("data")
    d.add_argument("--dev", type=int, default=60)
    d.add_argument("--test", type=int, default=60)
    r = sub.add_parser("reset")
    r.add_argument("--budget", type=int, default=40)
    f = sub.add_parser("forward")
    f.add_argument("request")
    a = sub.add_parser("astra")
    a.add_argument("--steps", type=int, default=80)
    sub.add_parser("test")
    sub.add_parser("status")
    args = ap.parse_args()

    if args.cmd == "data":
        from student.weights.data import build
        print(json.dumps(build(dev_size=args.dev, test_size=args.test), indent=1))
    elif args.cmd == "reset":
        from student.weights.runner import reset
        reset(args.budget)
        print(f"edits dropped, budget {args.budget}")
    elif args.cmd == "forward":
        from student.weights.runner import forward
        print(json.dumps(forward(json.loads(args.request)), indent=1))
    elif args.cmd == "astra":
        from student.weights.teacher import session
        print(json.dumps(session(max_steps=args.steps), indent=1))
    elif args.cmd == "test":
        from student.weights.runner import test
        print(json.dumps(test(), indent=1))
    elif args.cmd == "status":
        from student.weights import kernel, runner
        print(json.dumps({"edited": kernel.edited(), "budget": runner.budget(), "edit_sha": runner.edit_sha()}, indent=1))


if __name__ == "__main__":
    main()
