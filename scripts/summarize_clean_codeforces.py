"""Explain the clean Codeforces subset and print its statistics.

Reads data/clean/ and the finished Harbor jobs. Qwen3-Coder passed every
problem in the subset. The three small models were run on the problems that
were exported, five attempts each at temperature 0.7. The note
data/clean/CLEANING.md explains why the subset was cut this way.

Examples:
    uv run python scripts/summarize_clean_codeforces.py
"""

import os
import statistics
import sys
from collections import Counter

import pyarrow.compute as pc

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from tagent.data.clean_codeforces import SPLITS, problem_ids, problems_table, submissions_table  # noqa: E402
from tagent.data.clean_runs import LARGE_CODER, SMALL_MODELS, small_model_runs  # noqa: E402


def rating_band(rating):
    """Group a Codeforces rating into the bands used in the run reports."""
    if rating is None:
        return "unrated"
    if rating <= 1100:
        return "800-1100"
    if rating <= 1500:
        return "1200-1500"
    if rating <= 1900:
        return "1600-1900"
    return "2000+"


def counts_per_problem(table, verdict=None):
    """Return {problem id: number of submissions}, optionally one verdict."""
    if verdict is not None:
        table = table.filter(pc.equal(table["verdict"], verdict))
    if table.num_rows == 0:
        return {}
    grouped = table.group_by("problem_id").aggregate([("problem_id", "count")])
    return dict(zip(grouped["problem_id"].to_pylist(), grouped["problem_id_count"].to_pylist()))


def describe(values):
    """Return total, mean, median, min, max for a list of counts."""
    if not values:
        return (0, 0, 0, 0, 0)
    return (
        sum(values),
        statistics.mean(values),
        statistics.median(values),
        min(values),
        max(values),
    )


def main():
    print("Clean Codeforces subset")
    print("The tables in data/clean are problems Qwen3-Coder solved from the public")
    print("statement, plus every Python 3 submission to those problems. A pass by the")
    print("30B or 32B coder is not enough, so some problems a smaller coder solved are")
    print("absent. Samples and official tests are stored for grading. They were not")
    print("shown to the coder that had to solve the statement. The full rationale is")
    print("data/clean/CLEANING.md.")
    print()
    print(f"Large coder already run on every problem here: {LARGE_CODER[1]} ({LARGE_CODER[2]}).")
    print("It passed the stored tests. That pass is why the problem is in the subset.")
    print("Small-model runs are five separate attempts at temperature 0.7, not five")
    print("rounds of test feedback. A problem counts as solved for a small model when")
    print("any attempt passes. The stopped condition-01 job is not included.")
    print()

    grand_problems = 0
    grand_submissions = 0
    for split in SPLITS:
        ids = problem_ids(split)
        problems = problems_table(split).select(["id", "rating", "index", "testset_size"])
        submissions = submissions_table(split).select(["problem_id", "verdict"])
        grand_problems += len(ids)
        grand_submissions += submissions.num_rows
        print(f"=== {split}: {len(ids)} problems, {submissions.num_rows:,} submissions ===")

        bands = Counter(rating_band(rating) for rating in problems["rating"].to_pylist())
        letters = Counter((index or "?")[:1] for index in problems["index"].to_pylist())
        print("Rating: " + ", ".join(f"{band} {bands[band]}" for band in ("800-1100", "1200-1500", "1600-1900", "2000+", "unrated") if bands[band]))
        print("Index:  " + ", ".join(f"{letter} {letters[letter]}" for letter in sorted(letters)))
        sizes = [size or 0 for size in problems["testset_size"].to_pylist()]
        print(f"Official tests per problem: median {statistics.median(sizes):g}, min {min(sizes)}, max {max(sizes)}")

        all_counts = counts_per_problem(submissions)
        ok_counts = counts_per_problem(submissions, "OK")
        print(f"{'Submissions per problem':28} {'total':>9} {'average':>9} {'median':>7} {'min':>5} {'max':>7}")
        for label, counts in (("all", all_counts), ("accepted (OK)", ok_counts)):
            total, mean, median, low, high = describe(list(counts.values()) + [0] * (len(ids) - len(counts)))
            print(f"{label:28} {total:9,} {mean:9.1f} {median:7g} {low:5,} {high:7,}")
        verdicts = Counter(submissions["verdict"].to_pylist())
        print("Verdicts: " + ", ".join(f"{name} {count:,}" for name, count in verdicts.most_common(6)))

        runs, missing = small_model_runs(split)
        if missing:
            print("Missing Harbor jobs: " + ", ".join(missing))
        print(f"{'Small model':16} {'problems run':>14} {'solved':>14} {'passing trials':>18}")
        for slug, label, _model in SMALL_MODELS:
            run_rows = [runs[problem_id][slug] for problem_id in ids if problem_id in runs and slug in runs[problem_id]]
            solved = sum(1 for row in run_rows if row["passes"])
            passes = sum(row["passes"] for row in run_rows)
            trials = sum(row["trials"] for row in run_rows)
            run_text = f"{len(run_rows)} of {len(ids)}"
            solved_text = f"{solved} of {len(run_rows)}"
            trial_text = f"{passes} of {trials}"
            print(f"{label:16} {run_text:>14} {solved_text:>14} {trial_text:>18}")
        not_run = sum(1 for problem_id in ids if problem_id not in runs)
        print(f"Problems with no small-model run: {not_run}")
        print()

    print(f"All splits: {grand_problems} problems, {grand_submissions:,} submissions.")
    print("List one split with scripts/list_clean_codeforces.py. The per-problem")
    print("model grid is scripts/show_clean_runs.py.")


if __name__ == "__main__":
    main()
