"""Where the student (local Qwen) lives. Everything under data/ (git-ignored).

    data/models/<name>-frozen/      the frozen base model: read-only copy + manifest.json (sha256)
    data/student/datasets/<id>/     train.jsonl / valid.jsonl built from episode logs (+ teacher edits)
    data/student/edits.jsonl        the teacher's (Astra's) edits to the training data
    data/student/runs/<id>/         one fine-tuning run: adapters.safetensors, config.json, log.jsonl
    data/student/evals/<id>.jsonl   one replay evaluation (per step) + <id>.json (summary)
"""
from __future__ import annotations

from common.paths import DATA_DIR

BASE_REPO = "mlx-community/Qwen3.5-4B-4bit"
MODELS_DIR = DATA_DIR / "models"
FROZEN_DIR = MODELS_DIR / "qwen3.5-4b-4bit-frozen"
STUDENT_DIR = DATA_DIR / "student"
DATASETS_DIR = STUDENT_DIR / "datasets"
RUNS_DIR = STUDENT_DIR / "runs"
EVALS_DIR = STUDENT_DIR / "evals"
EDITS_FILE = STUDENT_DIR / "edits.jsonl"

# held out from training: the replay benchmark (four 3/3 successes of 17 Nov + two runs of 18 Nov)
EVAL_EPISODES = ("20261117-165638-c1c822", "20261117-172537-5f5c16", "20261117-173052-a287ec",
                 "20261117-173616-86e54e", "20261118-164411-bda29f", "20261118-171824-339d3a")
