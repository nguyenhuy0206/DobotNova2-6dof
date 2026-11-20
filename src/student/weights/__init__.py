"""Weight lab: the teacher (Astra) edits the student's (local Qwen) weights directly, MIT exp8 style.

    kernel   the only API the teacher gets: read tensors, save edits (rules enforced)
    data     dev items (teacher sees labels) and the held-out test items (runner only)
    runner   builds the edited student, the budgeted forward tool, the held-out test
    teacher  the Astra session: python / forward / finish

Folders (data/student/weightlab/): data/ (dev.jsonl, tools.json: the teacher may read),
work/ (the teacher's: notes, edited/<tensor>.npy, fwd_<k>/), runner/ (budget, test.jsonl, logs).
"""
