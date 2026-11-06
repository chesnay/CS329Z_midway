"""List clean Codeforces problems and, for one problem, its submissions.

Reads data/clean/. Each problem line says which small models have already
been run and how many of the five attempts passed. Qwen3-Coder has already
passed every problem listed. Pass --problem to list that problem's
submissions. Pass --source to print one submitted program.

Examples:
    uv run python scripts/list_clean_codeforces.py --split train
    uv run python scripts/list_clean_codeforces.py --split dev --problem 26/A
    uv run python scripts/list_clean_codeforces.py --split train --problem 21/B --source 1
"""

import argparse
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from tagent.data.clean_codeforces import (  # noqa: E402
    SPLITS,
    load_problem,
    load_submission,
    problem_ids,
    problems_table,
    split_of,
    submissions_table,
)
from tagent.data.clean_runs import LARGE_CODER, SMALL_MODELS, format_run, small_model_runs  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--split", required=True, choices=SPLITS)
    parser.add_argument("--problem", help="one problem id, such as 21/B. Lists its submissions.")
    parser.add_argument("--source", help="submission id whose program to print. Requires --problem.")
    return parser.parse_args()


def shorten(text, limit):
    """Return text on one line, cut off past limit."""
    flat = " ".join((text or "").split())
    if len(flat) <= limit:
        return flat
    return flat[: limit - 1] + "…"


def print_split(split):
    """Print one line per problem, including the small-model runs."""
    problems = {row["id"]: row for row in problems_table(split).select(["id", "title", "rating", "testset_size"]).to_pylist()}
    submissions = submissions_table(split).select(["problem_id", "verdict"])
    counts = {}
    accepted = {}
    for problem_id, verdict in zip(submissions["problem_id"].to_pylist(), submissions["verdict"].to_pylist()):
        counts[problem_id] = counts.get(problem_id, 0) + 1
        if verdict == "OK":
            accepted[problem_id] = accepted.get(problem_id, 0) + 1
    runs, missing = small_model_runs(split)
    if missing:
        print("Missing Harbor jobs: " + ", ".join(missing))
    headers = " ".join(f"{label:>14}" for _slug, label, _model in SMALL_MODELS)
    print(f"{split}: problems Qwen3-Coder ({LARGE_CODER[2]}) already solved.")
    print("Small-model cells are passing attempts out of the attempts that finished.")
    print(f"{'id':8} {'rating':7} {'tests':5} {'subs':6} {'ok':5} {headers}  title")
    for problem_id in problem_ids(split):
        row = problems[problem_id]
        cells = []
        for slug, _label, _model in SMALL_MODELS:
            run = (runs.get(problem_id) or {}).get(slug)
            cells.append(f"{format_run(run):>14}")
        rating = "-" if row["rating"] is None else str(int(row["rating"]))
        print(
            f"{problem_id:8} {rating:>7} {int(row['testset_size'] or 0):5} "
            f"{counts.get(problem_id, 0):6} {accepted.get(problem_id, 0):5} "
            + " ".join(cells)
            + "  "
            + shorten(row["title"], 48)
        )


def print_problem(split, problem_id, source_id):
    """Print one problem, the models that ran, and its submissions."""
    if split_of(problem_id) != split:
        raise SystemExit(f"{problem_id} is not in the clean {split} set.")
    problem = load_problem(problem_id)
    runs, _missing = small_model_runs(split)
    print(f"{problem_id}  {problem['title']}")
    rating = "unrated" if problem["rating"] is None else str(int(problem["rating"]))
    print(f"Split {split}. Rating {rating}. Official tests {int(problem['testset_size'] or 0)}.")
    print(f"{LARGE_CODER[1]} ({LARGE_CODER[2]}): already passed. The problem is in this set for that reason.")
    for slug, label, model in SMALL_MODELS:
        run = (runs.get(problem_id) or {}).get(slug)
        if not run:
            print(f"{label} ({model}): not run")
            continue
        print(f"{label} ({model}): {format_run(run)} passed, job {run['job']}")
    rows = submissions_table(split).select(
        ["problem_id", "submission_id", "verdict", "passedTestCount"]
    )
    chosen = [
        row for row in rows.to_pylist() if row["problem_id"] == problem_id
    ]
    print(f"\nSubmissions: {len(chosen)}")
    print(f"{'submission_id':16} {'verdict':24} {'tests passed':>12}")
    for row in chosen:
        passed = row["passedTestCount"]
        passed_text = "-" if passed is None else str(int(passed))
        print(f"{row['submission_id']:16} {row['verdict']:24} {passed_text:>12}")
    if source_id is None:
        print("\nPass --source SUBMISSION_ID to print one program.")
        return
    submission = load_submission(source_id)
    if submission is None or submission.get("problem_id") != problem_id:
        raise SystemExit(f"Submission {source_id} is not one of the submissions for {problem_id}.")
    print(f"\n--- submission {source_id} ({submission['verdict']}) ---")
    print(submission.get("source") or "(none)")


def main():
    args = parse_args()
    if args.source and not args.problem:
        raise SystemExit("--source requires --problem.")
    try:
        if args.problem:
            print_problem(args.split, args.problem, args.source)
        else:
            print_split(args.split)
    except BrokenPipeError:
        sys.stdout.close()
        raise SystemExit(0)


if __name__ == "__main__":
    main()
