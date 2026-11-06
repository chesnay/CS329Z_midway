"""Read the Codeforces subset in use and its Train / Dev / Test split.

Data: data/interim/codeforces_official_subset/ (problems.parquet,
submissions.parquet) and data/splits/codeforces/ (train.txt, dev.txt,
test.txt). See data/README.md for how they were built.
"""

import os
from functools import lru_cache

import pyarrow.parquet as pq

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
SUBSET_DIR = os.path.join(PROJECT_ROOT, "data", "interim", "codeforces_official_subset")
SPLIT_DIR = os.path.join(PROJECT_ROOT, "data", "splits", "codeforces")
SPLITS = ("train", "dev", "test")


def load_problem(problem_id):
    """Return one problem as a dict (all columns), or None if not in the subset."""
    table = pq.read_table(os.path.join(SUBSET_DIR, "problems.parquet"), filters=[("id", "=", problem_id)])
    rows = table.to_pylist()
    return rows[0] if rows else None


def load_submission(submission_id):
    """Return one submission as a dict, or None if not in the subset."""
    table = pq.read_table(os.path.join(SUBSET_DIR, "submissions.parquet"),
                          filters=[("submission_id", "=", str(submission_id))])
    rows = table.to_pylist()
    return rows[0] if rows else None


@lru_cache(maxsize=1)
def split_membership():
    """Return {problem id: "train" | "dev" | "test"}."""
    membership = {}
    for split in SPLITS:
        with open(os.path.join(SPLIT_DIR, f"{split}.txt")) as handle:
            for line in handle:
                if line.strip():
                    membership[line.strip()] = split
    return membership


def split_of(problem_id):
    """Return the Tagent split of a problem, or None if it is in none."""
    return split_membership().get(problem_id)
