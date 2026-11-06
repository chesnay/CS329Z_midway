"""Write a Harbor task that starts from a failed submission plus the statement.

The student sees the problem statement and their own failing program in
/app/solution.py. Official tests and one accepted program stay in tests/ and
solution/, which Harbor does not show to the student. Every submission inherits
the split of its problem, so submissions for one problem cannot fall into two
splits. See data/example_code/HARBOR.md.

Examples:
    uv run python data/example_code/export_harbor_task.py --check-split
    uv run python data/example_code/export_harbor_task.py --all-failed
    uv run python data/example_code/export_harbor_task.py 630/K --submission 217126120 --check
"""

import argparse
import json
import os
import stat
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "src"))

import pyarrow.parquet as pq

from tagent.data.statement_math import restore_statement_math

from show_example import (
    SPLIT_DIR,
    SUBSET_DIR,
    find_submission,
    load_problem,
    load_submissions,
    split_of,
    working_solution,
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_OUT_ROOT = os.path.join(PROJECT_ROOT, "data", "example_code", "harbor_tasks")
CASES_DIR = os.path.join(PROJECT_ROOT, "data", "example_code", "cases")
# Minimal-feedback control. Pass --feedback to replace this comment.
DEFAULT_TEACHER_FEEDBACK = "try again"


def tokens_match(actual, expected):
    """Codeforces ignores extra whitespace. Same rule as the project grader."""
    return (actual or "").split() == (expected or "").split()


def task_dir_name(problem_id, submission_id):
    """630/K and submission 217126120 become 630_K__217126120."""
    return f"{problem_id.replace('/', '_')}__{submission_id}"


def load_split_ids():
    """Return {split: [problem_id, ...]}. One file, one problem id per line."""
    splits = {}
    for name in ("train", "dev", "test"):
        path = os.path.join(SPLIT_DIR, f"{name}.txt")
        with open(path, encoding="utf-8") as handle:
            splits[name] = [line.strip() for line in handle if line.strip()]
    return splits


def check_problem_split():
    """Return a list of leakage errors. An empty list means the split is safe.

    Safe means every official-subset problem is in exactly one of train, dev,
    or test. A submission is not assigned its own split: it belongs to its
    problem, so it cannot land in a second split.
    """
    splits = load_split_ids()
    errors = []
    seen = {}
    for name, problem_ids in splits.items():
        for problem_id in problem_ids:
            previous = seen.get(problem_id)
            if previous is not None:
                errors.append(f"{problem_id} is in both {previous} and {name}")
            else:
                seen[problem_id] = name
    table = pq.read_table(os.path.join(SUBSET_DIR, "problems.parquet"), columns=["id"])
    official = set(table.column("id").to_pylist())
    assigned = set(seen)
    missing = sorted(official - assigned)
    extra = sorted(assigned - official)
    if missing:
        errors.append(f"{len(missing)} official problems have no split, including {missing[:5]}")
    if extra:
        errors.append(f"{len(extra)} split ids are not in the official subset, including {extra[:5]}")
    return errors


def submission_split(problem_id):
    """The split of every submission for this problem. None when the problem is unassigned."""
    return split_of(problem_id)


def eligible(problem, submissions, faulty):
    """Return an error string when this pair cannot be a student-repair task."""
    if faulty is None:
        return "that submission is not saved for this problem"
    if faulty["verdict"] == "OK":
        return f"submission {faulty['submission_id']} was accepted, so it is not a failed start"
    if (problem.get("input_mode") or "") != "stdio":
        return f"input_mode is {problem.get('input_mode')!r}, not stdio"
    if (problem.get("interaction_format") or "").strip():
        return "the problem is interactive"
    if (problem.get("generated_checker") or "").strip():
        return "the problem has a custom checker, so token comparison is not the judge"
    if not problem.get("official_tests_complete"):
        return "official tests are incomplete"
    if not (problem.get("official_tests") or []):
        return "the problem has no official tests"
    if working_solution(submissions) is None:
        return "the subset has no accepted submission to score the repair against"
    return None


def render_instruction(problem, faulty, feedback=""):
    """Statement plus the fact that /app/solution.py is the student's failing program.

    The instruction always ends with a teacher comment. An empty feedback
    argument uses ``try again``. A non-empty feedback argument replaces that
    comment. This text must not contain the accepted program or the hidden
    tests. The failing program itself is the starter file, not a paste in this
    instruction.
    """
    samples = []
    for index, pair in enumerate(problem.get("examples") or [], start=1):
        samples.append(
            f"Sample {index} input:\n{(pair.get('input') or '').rstrip()}\n\n"
            f"Sample {index} output:\n{(pair.get('output') or '').rstrip()}"
        )
    sample_text = "\n\n".join(samples) if samples else "(no samples)"
    note = restore_statement_math(problem.get("note") or "").strip()
    note_block = f"\n\n## Note\n\n{note}" if note else ""
    feedback = (feedback or "").strip() or DEFAULT_TEACHER_FEEDBACK
    feedback_block = f"\n## Teacher feedback\n\n{feedback}"
    return f"""# {problem['id']}: {problem['title']}

Your Python 3 program is `/app/solution.py`. It reads standard input and writes the answer to standard output.

Codeforces judged submission {faulty['submission_id']} as {faulty['verdict']}. It passed {int(faulty['passedTestCount'])} tests. Fix the program in place. Do not change the file path.

Time limit: {problem['time_limit']} seconds. Memory limit: {problem['memory_limit']} MB.

## Statement

{restore_statement_math(problem.get('description') or '').strip()}

## Input

{restore_statement_math(problem.get('input_format') or '').strip() or '(none)'}

## Output

{restore_statement_math(problem.get('output_format') or '').strip() or '(none)'}
{note_block}

## Samples

{sample_text}
{feedback_block}
"""


def hidden_output_leaks(instruction, problem):
    """Hidden expected outputs that appear as their own line or block.

    A short output such as ``1`` is not a leak merely because the statement
    contains that digit inside another number.
    """
    sample_outputs = set()
    sample_inputs = set()
    sample_input_lines = set()
    for pair in problem.get("examples") or []:
        sample_outputs.add((pair.get("output") or "").strip())
        sample_input = (pair.get("input") or "").strip()
        sample_inputs.add(sample_input)
        sample_input_lines.update(line.strip() for line in sample_input.splitlines())
    lines = {line.strip() for line in instruction.splitlines()}
    leaked = []
    for test in problem.get("official_tests") or []:
        output = (test.get("output") or "").strip()
        # A sample input line such as "4" is the public sample, even when some
        # hidden test's whole answer is that same token.
        if not output or output in sample_outputs or output in sample_inputs:
            continue
        if "\n" not in output and output in sample_input_lines:
            continue
        if "\n" in output:
            if output in instruction:
                leaked.append(output)
        elif output in lines:
            leaked.append(output)
    return leaked


def render_task_toml(problem, split, faulty):
    """Harbor metadata. The split is the problem's split, never chosen per submission."""
    description = _toml_string(
        f"Repair Codeforces {problem['id']} starting from submission {faulty['submission_id']}"
    )
    tags = ", ".join(_toml_string(tag) for tag in (problem.get("tags") or []))
    verifier_timeout = max(60.0, float(problem["time_limit"] or 2) * max(len(problem["official_tests"]), 1) * 3)
    return f"""schema_version = "1.3"
artifacts = []

[task]
name = "codeforces/{task_dir_name(problem['id'], faulty['submission_id'])}"
description = {description}
authors = [{{ name = "Codeforces authors" }}, {{ name = "Tagent export", role = "Harbor port" }}]
keywords = ["codeforces", "python", "repair", {tags}]

[metadata]
difficulty = {problem.get('rating') or 0}
category = "repair"
source = "open-r1/codeforces"
source_task_id = {_toml_string(problem['id'])}
faulty_submission_id = {_toml_string(faulty['submission_id'])}
faulty_verdict = {_toml_string(faulty['verdict'])}
split = {_toml_string(split or "unassigned")}

[verifier]
timeout_sec = {verifier_timeout:.1f}
collect = []

[verifier.env]

[agent]
timeout_sec = 600.0

[environment]
network_mode = "public"
build_timeout_sec = 600.0
os = "linux"
mcp_servers = []

[environment.env]

[solution.env]
"""


def _toml_string(text):
    escaped = str(text).replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def render_verifier(problem):
    """A checker that runs /app/solution.py on each hidden test."""
    per_test = float(problem["time_limit"] or 2) * 3
    return f'''"""Run /app/solution.py on the official tests. Exit 0 only when all pass.

Whitespace is compared the same way as tagent.execution.sandbox.tokens_match.
"""

import subprocess
import sys
from pathlib import Path

TESTS = Path(__file__).with_name("official_tests.json")
PROGRAM = Path("/app/solution.py")
PER_TEST_SEC = {per_test:.1f}


def tokens_match(actual, expected):
    return (actual or "").split() == (expected or "").split()


def main():
    tests = __import__("json").loads(TESTS.read_text(encoding="utf-8"))
    if not PROGRAM.is_file():
        print("missing /app/solution.py", file=sys.stderr)
        return 1
    for index, test in enumerate(tests, start=1):
        try:
            result = subprocess.run(
                [sys.executable, str(PROGRAM)],
                input=test.get("input") or "",
                text=True,
                capture_output=True,
                timeout=PER_TEST_SEC,
            )
        except subprocess.TimeoutExpired:
            print(f"test {{index}}: timeout", file=sys.stderr)
            return 1
        if result.returncode != 0:
            print(f"test {{index}}: exit {{result.returncode}}", file=sys.stderr)
            print(result.stderr, file=sys.stderr)
            return 1
        if not tokens_match(result.stdout, test.get("output") or ""):
            print(f"test {{index}}: wrong answer", file=sys.stderr)
            return 1
    print(f"{{len(tests)}} tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''


def render_test_sh():
    return """#!/bin/bash
set -uo pipefail
mkdir -p /logs/verifier
if python /tests/test_outputs.py; then
  echo 1 > /logs/verifier/reward.txt
else
  echo 0 > /logs/verifier/reward.txt
fi
"""


def render_solve_sh():
    """The oracle replaces the student's program with an accepted one, for scoring only."""
    return """#!/bin/bash
set -euo pipefail
cp /solution/solution.py /app/solution.py
"""


def render_dockerfile():
    return """FROM python:3.12-slim-bookworm
WORKDIR /app
COPY solution.py /app/solution.py
"""


def refresh_instruction(out_dir):
    """Rewrite instruction.md from the stored statement, with flattened math restored.

    Tests, the starter program, and the task metadata stay as they are. The teacher
    comment already in the file is kept. Returns ``updated``, ``same``, ``leak``,
    ``golden``, ``missing``, or ``skip``.
    """
    import tomllib

    path = os.path.join(out_dir, "instruction.md")
    toml_path = os.path.join(out_dir, "task.toml")
    if not (os.path.isfile(path) and os.path.isfile(toml_path)):
        return "skip"
    with open(toml_path, "rb") as handle:
        metadata = tomllib.load(handle).get("metadata") or {}
    problem_id = metadata.get("source_task_id")
    submission_id = metadata.get("faulty_submission_id")
    if not problem_id or not submission_id:
        return "skip"
    with open(path, encoding="utf-8") as handle:
        current = handle.read()
    marker = "\n## Teacher feedback\n\n"
    feedback = current.split(marker, 1)[1].strip() if marker in current else ""
    problem = load_problem(problem_id)
    submissions = load_submissions(problem_id) if problem is not None else None
    faulty = find_submission(submissions, submission_id) if submissions else None
    if problem is None or faulty is None:
        return "missing"
    instruction = render_instruction(problem, faulty, feedback)
    if instruction == current:
        return "same"
    if hidden_output_leaks(instruction, problem):
        return "leak"
    golden = working_solution(submissions)
    if golden and golden["source"].strip() and golden["source"].strip() in instruction:
        return "golden"
    temporary = path + ".tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        handle.write(instruction)
    os.replace(temporary, path)
    return "updated"


def _directory_is_this_task(out_dir, problem_id, submission_id):
    """True when out_dir was already exported for this problem and submission."""
    path = os.path.join(out_dir, "task.toml")
    if not os.path.isfile(path):
        return False
    import tomllib

    with open(path, "rb") as handle:
        metadata = tomllib.load(handle).get("metadata") or {}
    return (
        str(metadata.get("source_task_id")) == str(problem_id)
        and str(metadata.get("faulty_submission_id")) == str(submission_id)
    )


def export_harbor_task(problem_id, submission_id, out_dir, feedback="", allow_test=False):
    """Write a repair task for one failed submission. Return the output directory.

    The directory is not chosen freely relative to the split: callers should
    pass a path under that problem's split. This function still records the
    split in task.toml and refuses a test-split problem unless allow_test is set.
    """
    problem = load_problem(problem_id)
    if problem is None:
        raise SystemExit(f"{problem_id} is not in the official subset")
    errors = check_problem_split()
    if errors:
        raise SystemExit("Problem split is not safe:\n" + "\n".join(errors))
    split = submission_split(problem_id)
    if split is None:
        raise SystemExit(f"{problem_id} is not in train, dev, or test")
    if split == "test" and not allow_test:
        raise SystemExit(f"{problem_id} is in the test split. Pass --allow-test to export it.")

    submissions = load_submissions(problem_id)
    faulty = find_submission(submissions, submission_id)
    reason = eligible(problem, submissions, faulty)
    if reason:
        raise SystemExit(f"{problem_id} submission {submission_id} was not exported: {reason}")
    # The folder name is the split. A train problem cannot be written under test/.
    out_dir = os.path.abspath(out_dir)
    parent = os.path.basename(os.path.dirname(out_dir))
    if parent != split:
        raise SystemExit(
            f"{problem_id} belongs to {split}. Write the task in a directory named {split}, not {parent or '(none)'}."
        )

    instruction = render_instruction(problem, faulty, feedback)
    golden = working_solution(submissions)
    if golden["source"].strip() and golden["source"].strip() in instruction:
        raise SystemExit("The instruction contains the accepted program.")
    leaked = hidden_output_leaks(instruction, problem)
    if leaked:
        raise SystemExit("The instruction contains a hidden expected output.")

    # A second export of the same task updates the instruction, including a new
    # --feedback comment. A different problem or submission stays refused.
    if os.path.isdir(out_dir) and os.listdir(out_dir):
        if not _directory_is_this_task(out_dir, problem["id"], faulty["submission_id"]):
            raise SystemExit(f"Refusing to overwrite a directory that is not this task: {out_dir}")

    _write(os.path.join(out_dir, "instruction.md"), instruction)
    _write(os.path.join(out_dir, "task.toml"), render_task_toml(problem, split, faulty))
    _write(os.path.join(out_dir, "environment", "Dockerfile"), render_dockerfile())
    _write(os.path.join(out_dir, "environment", "solution.py"), faulty["source"])
    _write(os.path.join(out_dir, "tests", "test.sh"), render_test_sh(), executable=True)
    _write(os.path.join(out_dir, "tests", "test_outputs.py"), render_verifier(problem))
    _write(
        os.path.join(out_dir, "tests", "official_tests.json"),
        json.dumps(problem["official_tests"], ensure_ascii=False, indent=2) + "\n",
    )
    _write(os.path.join(out_dir, "solution", "solve.sh"), render_solve_sh(), executable=True)
    _write(os.path.join(out_dir, "solution", "solution.py"), golden["source"])
    return out_dir


def _write(path, text, executable=False):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)
    if executable:
        mode = os.stat(path).st_mode
        os.chmod(path, mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def problem_block_reason(problem):
    """Why this problem cannot supply repair cases. None when it can."""
    if (problem.get("input_mode") or "") != "stdio":
        return f"input_mode is {problem.get('input_mode')!r}, not stdio"
    if (problem.get("interaction_format") or "").strip():
        return "interactive"
    if (problem.get("generated_checker") or "").strip():
        return "custom checker"
    if not problem.get("official_tests_complete"):
        return "official tests incomplete"
    return None


def build_failed_dataset(out_dir=CASES_DIR):
    """Write every non-OK submission into the jsonl file of its problem's split.

    The program text is stored on the case. The official tests are not copied:
    they belong to the problem, and copying them once per submission would
    repeat the same answer key. Accepted programs are not stored here.
    """
    errors = check_problem_split()
    if errors:
        raise SystemExit("Problem split is not safe:\n" + "\n".join(errors))

    problem_to_split = {}
    for name, problem_ids in load_split_ids().items():
        for problem_id in problem_ids:
            problem_to_split[problem_id] = name

    columns = ["id", "input_mode", "interaction_format", "generated_checker", "official_tests_complete"]
    problems = pq.read_table(os.path.join(SUBSET_DIR, "problems.parquet"), columns=columns).to_pylist()
    usable = set()
    skipped_problems = {}
    for problem in problems:
        reason = problem_block_reason(problem)
        if reason is None:
            usable.add(problem["id"])
        else:
            skipped_problems[reason] = skipped_problems.get(reason, 0) + 1

    submissions = pq.read_table(
        os.path.join(SUBSET_DIR, "submissions.parquet"),
        columns=["problem_id", "submission_id", "verdict", "passedTestCount", "source"],
    ).to_pylist()
    problems_with_ok = {row["problem_id"] for row in submissions if row["verdict"] == "OK"}

    os.makedirs(out_dir, exist_ok=True)
    handles = {}
    counts = {name: {} for name in ("train", "dev", "test")}
    problems_used = {name: set() for name in counts}
    skipped_submissions = {"problem not usable": 0, "problem has no accepted submission": 0}
    try:
        for name in counts:
            handles[name] = open(os.path.join(out_dir, f"{name}.jsonl"), "w", encoding="utf-8")
        for row in submissions:
            if row["verdict"] == "OK":
                continue
            problem_id = row["problem_id"]
            if problem_id not in usable:
                skipped_submissions["problem not usable"] += 1
                continue
            if problem_id not in problems_with_ok:
                skipped_submissions["problem has no accepted submission"] += 1
                continue
            split = problem_to_split[problem_id]
            record = {
                "split": split,
                "problem_id": problem_id,
                "submission_id": str(row["submission_id"]),
                "verdict": row["verdict"],
                "passed_test_count": int(row["passedTestCount"]),
                "source": row["source"],
            }
            handles[split].write(json.dumps(record, ensure_ascii=False) + "\n")
            verdict_counts = counts[split]
            verdict_counts[row["verdict"]] = verdict_counts.get(row["verdict"], 0) + 1
            problems_used[split].add(problem_id)
    finally:
        for handle in handles.values():
            handle.close()

    # A problem id in two case files would mean a submission leaked across splits.
    overlap = []
    for left, right in (("train", "dev"), ("train", "test"), ("dev", "test")):
        shared = problems_used[left] & problems_used[right]
        if shared:
            overlap.append(f"{left} and {right} share {sorted(shared)[:5]}")
    if overlap:
        raise SystemExit("Failed-submission files cross splits:\n" + "\n".join(overlap))

    manifest = {
        "rule": "Every submission whose verdict is not OK, on a usable problem, goes to the split of its problem.",
        "usable_problem": "stdio, not interactive, no custom checker, official tests complete, and at least one accepted submission",
        "cases": {name: sum(counts[name].values()) for name in counts},
        "problems": {name: len(problems_used[name]) for name in counts},
        "verdicts": counts,
        "skipped_problems": skipped_problems,
        "skipped_submissions": skipped_submissions,
        "files": {name: f"{name}.jsonl" for name in counts},
    }
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    return manifest


def program_passes(program_path, tests_path, timeout=30):
    """Return True when the program matches every official test."""
    with open(tests_path, encoding="utf-8") as handle:
        tests = json.load(handle)
    for index, test in enumerate(tests, start=1):
        try:
            result = subprocess.run(
                [sys.executable, program_path],
                input=test.get("input") or "",
                text=True,
                capture_output=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            print(f"test {index}: timeout", file=sys.stderr)
            return False
        if result.returncode != 0 or not tokens_match(result.stdout, test.get("output") or ""):
            return False
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("problem_id", nargs="?", help="problem id such as 630/K")
    parser.add_argument("--submission", help="failed submission number to use as the student's program")
    parser.add_argument(
        "--feedback",
        default="",
        help='teacher comment in the instruction; an empty value uses "try again"',
    )
    parser.add_argument("--out", help="task directory (default: harbor_tasks/<split>/<problem>__<submission>)")
    parser.add_argument("--allow-test", action="store_true", help="allow a problem from the test split")
    parser.add_argument("--check-split", action="store_true", help="check that every problem is in exactly one split")
    parser.add_argument("--all-failed", action="store_true", help="write every failed submission into cases/<split>.jsonl")
    parser.add_argument("--check", action="store_true", help="check the oracle passes and the student's program does not")
    args = parser.parse_args()

    if args.all_failed:
        manifest = build_failed_dataset()
        print(f"Wrote {CASES_DIR}")
        print("cases: " + ", ".join(f"{name} {count}" for name, count in manifest["cases"].items()))
        print("problems: " + ", ".join(f"{name} {count}" for name, count in manifest["problems"].items()))
        print("skipped problems:", manifest["skipped_problems"])
        print("skipped submissions:", manifest["skipped_submissions"])
        if not args.problem_id:
            return

    if args.check_split:
        errors = check_problem_split()
        splits = load_split_ids()
        print(
            "problems: "
            + ", ".join(f"{name} {len(ids)}" for name, ids in splits.items())
        )
        if errors:
            print("\n".join(errors))
            raise SystemExit(1)
        print("Every official problem is in exactly one split. Submissions follow their problem.")
        if not args.problem_id:
            return

    if not args.problem_id or not args.submission:
        raise SystemExit("Give a problem id and --submission, or pass --check-split.")

    split = submission_split(args.problem_id)
    if split is None:
        raise SystemExit(f"{args.problem_id} is not in train, dev, or test")
    out_dir = args.out or os.path.join(DEFAULT_OUT_ROOT, split, task_dir_name(args.problem_id, args.submission))
    export_harbor_task(args.problem_id, args.submission, out_dir, feedback=args.feedback, allow_test=args.allow_test)
    print(f"Wrote {out_dir}")
    print(f"Split {split}: every submission of {args.problem_id} belongs to {split}.")
    if args.check:
        tests_path = os.path.join(out_dir, "tests", "official_tests.json")
        oracle_ok = program_passes(os.path.join(out_dir, "solution", "solution.py"), tests_path)
        student_ok = program_passes(os.path.join(out_dir, "environment", "solution.py"), tests_path)
        if not oracle_ok:
            raise SystemExit("The accepted program did not pass the official tests.")
        if student_ok:
            raise SystemExit("The failed submission passed every official test, so it is not a faulty start.")
        print("Accepted program passed. The student's program did not.")


if __name__ == "__main__":
    main()
