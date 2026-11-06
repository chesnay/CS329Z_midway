"""Show one Codeforces problem from the official subset.

Run ``uv run python data/example_code/show_example.py --help`` for the commands.
The longer write-up is data/example_code/README.md.
"""

import argparse
import os
import sys

import pyarrow.parquet as pq

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "src"))

from tagent.data.statement_math import restore_statement_math

# This file lives at data/example_code/show_example.py, so the repository root
# is three directories up.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SUBSET_DIR = os.path.join(PROJECT_ROOT, "data", "interim", "codeforces_official_subset")
SPLIT_DIR = os.path.join(PROJECT_ROOT, "data", "splits", "codeforces")
DEFAULT_PROBLEM = "630/K"

HELP = """\
Show one Codeforces problem, the programs submitted for it, the judge tests,
and one accepted program.

Data read
  problems and submissions
      data/interim/codeforces_official_subset/
  Train / Dev / Test label
      data/splits/codeforces/{train,dev,test}.txt

There is no golden-solution column. A working solution is the shortest
submission whose verdict is OK. An incorrect submission is any row whose
verdict is not OK (WRONG_ANSWER, RUNTIME_ERROR, TIME_LIMIT_EXCEEDED, and the
other non-OK verdicts). The saved text is the program that was submitted, not
the output that program printed. An editorial, when the problem has one, is a
written explanation and is printed apart from the program.

Commands
  show_example.py [PROBLEM]
      Full report. PROBLEM defaults to 630/K (Indivisibility, train split):
      statement, samples, every submission, the incorrect ones again, the
      official tests, the shortest accepted program, and the editorial.

  show_example.py PROBLEM --incorrect
      Only the list of submissions that were not accepted. Columns are
      submission number, verdict, how many tests passed, time, and memory.

  show_example.py PROBLEM --submission NUMBER
      The program for that submission number on this problem. If the verdict
      is OK, the command says the submission was accepted. If the number
      belongs to another problem, the command stops with an error.

  show_example.py --help
  show_example.py -h
  show_example.py -help
      This explanation. --submission is ignored when help is requested.

Examples
  uv run python data/example_code/show_example.py
  uv run python data/example_code/show_example.py 630/K
  uv run python data/example_code/show_example.py 630/K --incorrect
  uv run python data/example_code/show_example.py 630/K --submission 217126120
  uv run python data/example_code/show_example.py 4/A --submission 208939622

630/K submission 217126120 is a wrong answer: it uses float division.
4/A submission 208939622 is a runtime error: it calls a function that
uses a name that was never defined.
"""


def split_of(problem_id):
    """Return 'train', 'dev', or 'test', or None when the id is in none of them."""
    for name in ("train", "dev", "test"):
        path = os.path.join(SPLIT_DIR, f"{name}.txt")
        with open(path, encoding="utf-8") as handle:
            if any(line.strip() == problem_id for line in handle):
                return name
    return None


def load_problem(problem_id):
    """Return one problem row, or None when the id is not in the official subset."""
    table = pq.read_table(os.path.join(SUBSET_DIR, "problems.parquet"), filters=[("id", "=", problem_id)])
    rows = table.to_pylist()
    return rows[0] if rows else None


def load_submissions(problem_id):
    """Return every saved submission for this problem.

    Accepted rows come first. Within each group, rows are ordered by submission
    number so the incorrect list stays stable from run to run.
    """
    table = pq.read_table(
        os.path.join(SUBSET_DIR, "submissions.parquet"),
        filters=[("problem_id", "=", problem_id)],
    )
    rows = table.to_pylist()
    rows.sort(key=lambda row: (row["verdict"] != "OK", int(row["submission_id"])))
    return rows


def working_solution(submissions):
    """Return the shortest accepted program.

    The subset has no reference solution. Length is the tie break that prefers
    a compact program; the smaller submission number breaks a remaining tie.
    """
    accepted = [row for row in submissions if row["verdict"] == "OK"]
    if not accepted:
        return None
    return min(accepted, key=lambda row: (len(row["source"]), int(row["submission_id"])))


def incorrect_submissions(submissions):
    """Return submissions Codeforces did not accept.

    ``source`` on each row is the submitted program. The dataset does not store
    the text that program printed on the failing test.
    """
    return [row for row in submissions if row["verdict"] != "OK"]


def find_submission(submissions, submission_id):
    """Return the row with this submission number, or None if it is not in the list.

    The number may be passed as an int or a string. Comparison is on the text
    form, because the Parquet column stores ids as strings.
    """
    wanted = str(submission_id)
    for row in submissions:
        if str(row["submission_id"]) == wanted:
            return row
    return None


def print_block(title, text):
    print(f"\n--- {title} ---")
    print((text or "").strip() or "(none)")


def print_submission_table(rows):
    """Print submission number, verdict, tests passed, time, and memory. Not the program."""
    if not rows:
        print("(none)")
        return
    print(f"{'submission_id':<14} {'verdict':<24} {'passed':>7} {'time_ms':>8} {'memory_bytes':>14}")
    for row in rows:
        print(
            f"{row['submission_id']:<14} {row['verdict']:<24} "
            f"{int(row['passedTestCount']):>7} {int(row['timeConsumedMillis']):>8} "
            f"{int(row['memoryConsumedBytes']):>14}"
        )


def print_program(row):
    print(
        f"submission {row['submission_id']}  problem {row['problem_id']}  "
        f"verdict {row['verdict']}  passed {int(row['passedTestCount'])}  "
        f"{len(row['source'])} characters"
    )
    print(row["source"].rstrip())


def print_pairs(title, pairs):
    """Print sample or official tests. Each item has input and expected output."""
    print(f"\n--- {title} ({len(pairs)}) ---")
    if not pairs:
        print("(none)")
        return
    for index, pair in enumerate(pairs, start=1):
        print(f"[{index}] input:")
        print(pair["input"] if pair["input"] is not None else "(none)")
        print(f"[{index}] output:")
        print(pair["output"] if pair["output"] is not None else "(none)")


def report_incorrect(problem_id, submissions):
    incorrect = incorrect_submissions(submissions)
    print(f"===== {problem_id}: incorrect submissions ({len(incorrect)} of {len(submissions)}) =====")
    print_submission_table(incorrect)


def report_one_submission(problem_id, submissions, submission_id):
    """Print one program. Say so when that program was accepted."""
    row = find_submission(submissions, submission_id)
    if row is None:
        raise SystemExit(f"Submission {submission_id} is not among the saved submissions for {problem_id}.")
    if row["verdict"] == "OK":
        print(f"Submission {submission_id} was accepted. It is not an incorrect answer.")
    else:
        print(f"Incorrect submission for {problem_id}:")
    print_program(row)


def main():
    # RawDescriptionHelpFormatter keeps the line breaks in HELP.
    # allow_abbrev=False stops --sub from being accepted as --submission.
    parser = argparse.ArgumentParser(
        description=HELP,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        allow_abbrev=False,
    )
    parser.add_argument("problem_id", nargs="?", default=DEFAULT_PROBLEM, help="problem id such as 630/K (default: 630/K)")
    parser.add_argument("--incorrect", action="store_true", help="list only submissions whose verdict is not OK")
    parser.add_argument("--submission", metavar="NUMBER", help="print the program for this submission number on PROBLEM")
    # argparse provides -h and --help. Also accept the single token -help.
    parser.add_argument("-help", action="help", help="show this explanation and exit")
    args = parser.parse_args()

    problem_id = args.problem_id
    problem = load_problem(problem_id)
    if problem is None:
        raise SystemExit(f"{problem_id} is not in data/interim/codeforces_official_subset/problems.parquet")

    submissions = load_submissions(problem_id)
    # One submission is a different report from the incorrect list. When both
    # flags are passed, the single program is the one that is printed.
    if args.submission:
        report_one_submission(problem_id, submissions, args.submission)
        return
    if args.incorrect:
        report_incorrect(problem_id, submissions)
        return

    solution = working_solution(submissions)
    split = split_of(problem_id)

    print(f"===== {problem['id']}: {problem['title']} =====")
    print(f"split            {split or '(not in data/splits/codeforces)'}")
    print(f"contest          {problem['contest_id']} {problem['contest_name']}")
    print(f"rating           {problem['rating']}")
    print(f"tags             {', '.join(problem['tags'] or []) or '(none)'}")
    print(f"time limit       {problem['time_limit']} s")
    print(f"memory limit     {problem['memory_limit']} MB")
    print(f"input mode       {problem['input_mode']}")
    print_block("Description", restore_statement_math(problem["description"] or ""))
    print_block("Input format", restore_statement_math(problem["input_format"] or ""))
    print_block("Output format", restore_statement_math(problem["output_format"] or ""))
    if problem["interaction_format"]:
        print_block("Interaction format", restore_statement_math(problem["interaction_format"]))
    if problem["note"]:
        print_block("Note", restore_statement_math(problem["note"]))
    print_pairs("Samples", problem["examples"] or [])

    print(f"\n--- Submissions ({len(submissions)}) ---")
    print_submission_table(submissions)
    incorrect = incorrect_submissions(submissions)
    print(f"\n--- Incorrect submissions ({len(incorrect)}) ---")
    print_submission_table(incorrect)

    print_pairs("Official tests", problem["official_tests"] or [])

    print("\n--- Working solution ---")
    if solution is None:
        print("No submission with verdict OK is in this subset.")
    else:
        print(
            f"submission {solution['submission_id']}  verdict {solution['verdict']}  "
            f"{len(solution['source'])} characters"
        )
        print(solution["source"].rstrip())

    if (problem["editorial"] or "").strip():
        print_block("Editorial", problem["editorial"])


if __name__ == "__main__":
    main()
