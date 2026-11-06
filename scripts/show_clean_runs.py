"""Show which models have already been run on the clean Codeforces subset.

Qwen3-Coder passed every problem. The three small models each had five
attempts at temperature 0.7 on the problems exported to Harbor. A cell is
the number of passing attempts out of the attempts that finished, or
"not run" when that problem was not exported. The stopped condition-01 job
is not included.

Examples:
    uv run python scripts/show_clean_runs.py --split train
    uv run python scripts/show_clean_runs.py --split test --problem 84/A
"""

import argparse
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from tagent.data.clean_codeforces import SPLITS, problem_ids, split_of  # noqa: E402
from tagent.data.clean_runs import (  # noqa: E402
    JOB_PREFIXES,
    LARGE_CODER,
    SMALL_MODELS,
    format_run,
    small_model_runs,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--split", required=True, choices=SPLITS)
    parser.add_argument("--problem", help="one problem id, such as 84/A")
    return parser.parse_args()


def main():
    try:
        _main()
    except BrokenPipeError:
        sys.stdout.close()
        raise SystemExit(0)


def _main():
    args = parse_args()
    ids = problem_ids(args.split)
    if args.problem:
        if split_of(args.problem) != args.split:
            raise SystemExit(f"{args.problem} is not in the clean {args.split} set.")
        ids = [args.problem]
    runs, missing = small_model_runs(args.split)
    prefixes = ", ".join(JOB_PREFIXES[args.split])
    print(f"{args.split}: {LARGE_CODER[1]} ({LARGE_CODER[2]}) already passed every problem below.")
    print(f"Small-model jobs: {prefixes}. Five attempts, temperature 0.7.")
    if missing:
        print("Missing Harbor jobs: " + ", ".join(missing))
    header = " ".join(f"{label:>16}" for _slug, label, _model in SMALL_MODELS)
    print(f"{'id':8} {header}  job")
    for problem_id in ids:
        cells = []
        jobs = []
        for slug, _label, _model in SMALL_MODELS:
            run = (runs.get(problem_id) or {}).get(slug)
            cells.append(f"{format_run(run):>16}")
            if run and run["job"] not in jobs:
                jobs.append(run["job"])
        print(f"{problem_id:8} " + " ".join(cells) + "  " + (", ".join(jobs) or "-"))


if __name__ == "__main__":
    main()
