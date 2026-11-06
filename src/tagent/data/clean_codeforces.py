"""Read the Codeforces problems Qwen3-Coder solved, and their submissions.

The tables are data/clean/problems/{train,dev,test}.parquet and
data/clean/submissions/{train,dev,test}.parquet. They are a subset of
data/interim/codeforces_official_subset/, with the same columns. A problem is
included only when qwen/qwen3-coder wrote a program that passed the stored
official tests. Passes by the 30B or 32B coders are not enough.
scripts/build_clean_codeforces.py writes the files.
"""

import os
from functools import lru_cache

import pyarrow.parquet as pq

from tagent.data.instruction_check import CODER_SMART

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
CLEAN_DIR = os.path.join(PROJECT_ROOT, "data", "clean")
PROBLEMS_DIR = os.path.join(CLEAN_DIR, "problems")
SUBMISSIONS_DIR = os.path.join(CLEAN_DIR, "submissions")
SPLITS = ("train", "dev", "test")


def solved_by_large_coder(record):
    """Return whether this audit record is a pass by qwen/qwen3-coder.

    A solve step counts when that model wrote a program and the official tests
    passed. The statement-review rollup has no steps; ``smart_solve`` of
    ``usable`` is that same pass. An error row does not count, and neither
    does a pass by another model.
    """
    if not isinstance(record, dict) or record.get("status") == "error":
        return False
    for step in record.get("steps") or []:
        if (
            step.get("model") == CODER_SMART
            and step.get("role") == "solve"
            and step.get("passed") is True
        ):
            return True
    return record.get("smart_solve") == "usable"


def problem_path(split):
    """Path of the problem table for one split."""
    _check_split(split)
    return os.path.join(PROBLEMS_DIR, f"{split}.parquet")


def submission_path(split):
    """Path of the submission table for one split."""
    _check_split(split)
    return os.path.join(SUBMISSIONS_DIR, f"{split}.parquet")


def problems_table(split):
    """Return the problem table for one split."""
    return pq.read_table(problem_path(split))


def submissions_table(split):
    """Return every submission to the problems in one split."""
    return pq.read_table(submission_path(split))


def load_problem(problem_id):
    """Return one problem as a dict, or None if the large coder did not solve it."""
    split = split_of(problem_id)
    if split is None:
        return None
    table = pq.read_table(problem_path(split), filters=[("id", "=", problem_id)])
    rows = table.to_pylist()
    return rows[0] if rows else None


def load_submissions(problem_id):
    """Return the submissions for one solved problem, or an empty list."""
    split = split_of(problem_id)
    if split is None:
        return []
    table = pq.read_table(
        submission_path(split), filters=[("problem_id", "=", problem_id)]
    )
    return table.to_pylist()


def load_submission(submission_id):
    """Return one submission as a dict, or None if it is not in this subset."""
    wanted = str(submission_id)
    for split in SPLITS:
        table = pq.read_table(
            submission_path(split), filters=[("submission_id", "=", wanted)]
        )
        rows = table.to_pylist()
        if rows:
            return rows[0]
    return None


@lru_cache(maxsize=1)
def split_membership():
    """Return {problem id: "train" | "dev" | "test"} for solved problems."""
    membership = {}
    for split in SPLITS:
        table = pq.read_table(problem_path(split), columns=["id"])
        for problem_id in table["id"].to_pylist():
            if problem_id in membership:
                raise RuntimeError(f"{problem_id} is in more than one clean split")
            membership[problem_id] = split
    return membership


def split_of(problem_id):
    """Return the split of a solved problem, or None if it is not in this subset."""
    return split_membership().get(problem_id)


def problem_ids(split):
    """Return the solved problem ids in one split, in table order."""
    return problems_table(split)["id"].to_pylist()


def _check_split(split):
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, not {split!r}")
