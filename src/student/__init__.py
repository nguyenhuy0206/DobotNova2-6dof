"""The student: a small local model (Qwen3.5-4B 4-bit on MLX; student.paths) that is taught by the teacher (Astra).

The base weights are frozen (freeze); learning happens in LoRA adapters trained on data the
teacher provides and edits (dataset + data/student/edits.jsonl), and every adapter is judged
against the frozen base on held-out episodes (evaluate). Nothing here moves the robot.

    paths      where models, datasets, runs and evaluations live
    freeze     the read-only base copy + checksums
    context    what the student sees at one step; reading its tool calls
    dataset    episode logs (+ teacher edits) -> training examples
    train      LoRA fine-tuning, loss on the target tool call only
    evaluate   replay benchmark and comparison
"""
