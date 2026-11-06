"""The clean subset keeps only problems Qwen3-Coder solved."""

from tagent.data.clean_codeforces import solved_by_large_coder
from tagent.data.instruction_check import CODER_FIRST, CODER_SMART


def test_large_coder_pass_counts():
    record = {
        "status": "usable",
        "steps": [{"model": CODER_SMART, "role": "solve", "passed": True}],
    }
    assert solved_by_large_coder(record)


def test_smaller_coder_pass_does_not_count():
    record = {
        "status": "usable",
        "passed_at": "first",
        "steps": [{"model": CODER_FIRST, "role": "solve", "passed": True}],
    }
    assert not solved_by_large_coder(record)


def test_review_rollup_counts_without_steps():
    assert solved_by_large_coder({"smart_solve": "usable"})
    assert not solved_by_large_coder({"smart_solve": "unresolved"})


def test_error_and_failed_solve_do_not_count():
    failed = {
        "status": "unresolved",
        "steps": [{"model": CODER_SMART, "role": "solve", "passed": False, "trial": 1}],
    }
    errored = {
        "status": "error",
        "steps": [{"model": CODER_SMART, "role": "solve", "passed": True}],
    }
    assert not solved_by_large_coder(failed)
    assert not solved_by_large_coder(errored)


def test_one_passing_trial_is_enough():
    record = {
        "status": "usable",
        "steps": [
            {"model": CODER_SMART, "role": "solve", "passed": False, "trial": 1},
            {"model": CODER_SMART, "role": "solve", "passed": True, "trial": 2},
        ],
    }
    assert solved_by_large_coder(record)
