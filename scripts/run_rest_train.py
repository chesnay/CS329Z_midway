"""Check the remaining usable train problems, then run the three small models.

The explanation for a colleague is data/example_code/REST_TRAIN.md.

The set is every usable train problem a strong coder has not already passed.
That is the train problems outside repair200, plus the repair200 problems that
are still unsolved. Dev and test are not read. 44/H is left out: every
accepted program crashes on its first stored test.

Stage 1 asks Qwen3-Coder up to 3 times, at temperature 0.7, and stops when one
program passes. The problem table is not changed.

Stage 2 runs only on the misses. Qwen3-Coder sees an accepted program that
already passes the stored tests and may insert math markup. It then solves the
repaired statement, up to 3 times, without that program. The repair is written
only when one of those attempts passes. A reply that changes the words is
discarded.

Stage 3 exports a condition-01 Harbor task for each problem this run solved
and runs Llama 3.2 1B, Ministral 3B, and Llama 3.2 3B for five attempts at
temperature 0.7. Job names start with rest_train, so this does not resume
solved158 or the stopped cond01 job.

``--split dev`` and ``--split test`` run the same three stages on every usable
problem in that split. They do not subtract repair200. Test export sets
allow_test. Job names start with heldout_dev or heldout_test. The task parent
directory is named for the split.

Examples:
    uv run python scripts/run_rest_train.py --list
    uv run python scripts/run_rest_train.py --smart-only --workers 8
    uv run python scripts/run_rest_train.py --clean-only --workers 8
    uv run python scripts/run_rest_train.py --small-only
    uv run python scripts/run_rest_train.py --workers 8
    uv run python scripts/run_rest_train.py --split dev --list
    uv run python scripts/run_rest_train.py --split test --workers 8
"""

import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import pyarrow.parquet as pq

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from tagent.data.codeforces import split_of
from tagent.data.instruction_check import (
    CODER_SMART,
    clean_unsolved,
    ensure_train,
    first_passing_submission,
    ids_already_checked,
    require_models,
    rest_train_problem_ids,
    run_checks,
    save_statement_updates,
    smart_trials,
    updates_from_records,
)

CASES_PATH = os.path.join(PROJECT_ROOT, "data", "example_code", "cases", "train.jsonl")
REPAIR_PATH = os.path.join(PROJECT_ROOT, "data", "splits", "codeforces", "repair200.jsonl")
ANCHOR_PATH = os.path.join(PROJECT_ROOT, "data", "example_code", "instruction_anchors.jsonl")
PROBLEMS_PATH = os.path.join(PROJECT_ROOT, "data", "interim", "codeforces_official_subset", "problems.parquet")
SUBMISSIONS_PATH = os.path.join(
    PROJECT_ROOT, "data", "interim", "codeforces_official_subset", "submissions.parquet"
)
SMART_AUDIT = os.path.join(PROJECT_ROOT, "data", "example_code", "instruction_audit_rest_train.jsonl")
CLEAN_AUDIT = os.path.join(PROJECT_ROOT, "data", "example_code", "instruction_audit_rest_train_clean.jsonl")
EXPORT_PATH = os.path.join(PROJECT_ROOT, "data", "example_code", "export_harbor_task.py")
TASK_ROOT = os.path.join(PROJECT_ROOT, "data", "example_code", "harbor_tasks", "rest_train", "train")
MANIFEST_PATH = os.path.join(PROJECT_ROOT, "data", "example_code", "harbor_tasks", "rest_train", "manifest.json")
JOBS_DIR = os.path.join(PROJECT_ROOT, "data", "example_code", "harbor_jobs")
SUMMARY_PATH = os.path.join(JOBS_DIR, "rest_train_summary.json")
DEFAULT_ENV = os.path.join(PROJECT_ROOT, ".env")

SMART_TRIALS = 3
SMART_TEMPERATURE = 0.7
SMALL_ATTEMPTS = 5
SMALL_TEMPERATURE = 0.7
SMALL_CONCURRENT = 4
ANCHOR_TIMEOUT = 8
SPLIT = "train"
JOB_PREFIX = "rest_train"
ALLOW_TEST = False
STATEMENT_COLUMNS = [
    "id", "title", "rating", "time_limit", "memory_limit", "description", "input_format",
    "output_format", "note", "interaction_format", "examples", "official_tests",
]
PREFERRED = ("WRONG_ANSWER", "RUNTIME_ERROR", "COMPILATION_ERROR")
SMALL_MODELS = (
    ("llama-3.2-1b", "openrouter/meta-llama/llama-3.2-1b-instruct"),
    ("ministral-3b", "openrouter/mistralai/ministral-3b-2512"),
    ("llama-3.2-3b", "openrouter/meta-llama/llama-3.2-3b-instruct"),
)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--list", action="store_true", help="print the set and stop")
    parser.add_argument("--smart-only", action="store_true", help="run the 3 smart-coder trials and stop")
    parser.add_argument("--clean-only", action="store_true", help="repair the smart-coder misses and stop")
    parser.add_argument("--small-only", action="store_true", help="run the three small models on the solved problems")
    parser.add_argument("--workers", type=int, default=8, help="smart-coder calls in flight (default 8)")
    parser.add_argument("--split", choices=("train", "dev", "test"), default="train")
    parser.add_argument("--env-file", default=DEFAULT_ENV)
    args = parser.parse_args()
    chosen = [args.smart_only, args.clean_only, args.small_only]
    if sum(chosen) > 1:
        raise SystemExit("Pass only one of --smart-only, --clean-only, and --small-only.")
    if args.workers < 1:
        raise SystemExit("workers must be at least 1")
    apply_split(args.split)

    problem_ids = load_problem_ids()
    if args.list:
        _print_list(problem_ids)
        return

    if not os.path.isfile(args.env_file):
        raise SystemExit(f"Env file not found: {args.env_file}")
    if not args.small_only:
        checker = _load_script("check_instruction_solvable")
        api_key = checker._load_openrouter_key(args.env_file)
        require_models(checker._catalog_ids(), (CODER_SMART,))
        complete = lambda model, prompt: checker._complete(model, prompt, api_key, SMART_TEMPERATURE)
    else:
        complete = None

    run_smart = not args.clean_only and not args.small_only
    run_clean = not args.smart_only and not args.small_only
    run_small = not args.smart_only and not args.clean_only
    if run_smart:
        _run_smart(problem_ids, complete, args.workers)
    if run_clean:
        _run_clean(problem_ids, complete, args.workers)
    if run_small:
        _run_small(problem_ids, args.env_file)


def apply_split(name):
    """Point the paths and the Harbor job prefix at one split.

    The default train paths stay as they were. Dev and test use their own
    case file, audits, tasks, and job names, and they do not resume train.
    """
    global SPLIT, JOB_PREFIX, ALLOW_TEST, CASES_PATH, SMART_AUDIT, CLEAN_AUDIT
    global TASK_ROOT, MANIFEST_PATH, SUMMARY_PATH
    if name == "train":
        return
    example = os.path.join(PROJECT_ROOT, "data", "example_code")
    SPLIT = name
    JOB_PREFIX = f"heldout_{name}"
    ALLOW_TEST = name == "test"
    CASES_PATH = os.path.join(example, "cases", f"{name}.jsonl")
    SMART_AUDIT = os.path.join(example, f"instruction_audit_heldout_{name}.jsonl")
    CLEAN_AUDIT = os.path.join(example, f"instruction_audit_heldout_{name}_clean.jsonl")
    TASK_ROOT = os.path.join(example, "harbor_tasks", JOB_PREFIX, name)
    MANIFEST_PATH = os.path.join(example, "harbor_tasks", JOB_PREFIX, "manifest.json")
    SUMMARY_PATH = os.path.join(JOBS_DIR, f"{JOB_PREFIX}_summary.json")


def load_problem_ids():
    """Problems for this split. Train keeps the unsolved-only set."""
    if SPLIT != "train":
        return _heldout_problem_ids()
    train_ids = set()
    with open(CASES_PATH, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                train_ids.add(json.loads(line)["problem_id"])
    solved = _load_script("run_solved158").solved_ids()
    skipped = _no_anchor_ids()
    problem_ids = rest_train_problem_ids(train_ids, solved, skipped)
    for problem_id in problem_ids:
        ensure_train(problem_id, split_of(problem_id))
    return problem_ids


def _heldout_problem_ids():
    """Every usable problem in the dev or test case file."""
    if not os.path.isfile(CASES_PATH):
        raise SystemExit(f"Case file not found: {CASES_PATH}")
    found = set()
    with open(CASES_PATH, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("split") != SPLIT:
                raise SystemExit(f"{row.get('problem_id')} row is split {row.get('split')}, not {SPLIT}")
            found.add(row["problem_id"])
    problem_ids = sorted(found, key=_sort_key)
    for problem_id in problem_ids:
        actual = split_of(problem_id)
        if actual != SPLIT:
            raise SystemExit(f"{problem_id} is {actual or 'not in a split'}, not {SPLIT}")
    return problem_ids


def _print_list(problem_ids):
    if SPLIT != "train":
        print(f"heldout {SPLIT} problems {len(problem_ids)}")
        return
    repair = _repair_ids()
    outside = [problem_id for problem_id in problem_ids if problem_id not in repair]
    inside = [problem_id for problem_id in problem_ids if problem_id in repair]
    print(f"rest train problems {len(problem_ids)}")
    print(f"outside repair200 {len(outside)}")
    print(f"unsolved repair200 {len(inside)}")
    print("skipped, no accepted program passes: " + ", ".join(sorted(_no_anchor_ids())))


def _run_smart(problem_ids, complete, workers):
    done = ids_already_checked(_read_jsonl(SMART_AUDIT))
    pending_ids = [problem_id for problem_id in problem_ids if problem_id not in done]
    print(f"smart trials: {len(pending_ids)} pending, {len(problem_ids) - len(pending_ids)} already recorded", flush=True)
    problems = _load_problems(pending_ids)

    def check(problem, complete, write=False, timeout=30):
        return smart_trials(problem, complete, trials=SMART_TRIALS, timeout=timeout)

    def on_record(record):
        record["temperature"] = SMART_TEMPERATURE
        _append(SMART_AUDIT, record)
        print(
            f"{record['problem_id']} {record['status']} tried {record.get('n_tried', '-')} "
            f"passed {record.get('n_passed', '-')}",
            flush=True,
        )

    records = run_checks(problems, complete, workers, on_record, write=False, check=check)
    usable = sum(1 for record in records if record["status"] == "usable")
    print(f"PROGRESS: smart done, {usable} solved in this batch of {len(records)}", flush=True)


def _run_clean(problem_ids, complete, workers):
    missed = [
        record["problem_id"]
        for record in _latest(SMART_AUDIT).values()
        if record.get("status") == "unresolved" and record["problem_id"] in set(problem_ids)
    ]
    done = ids_already_checked(_read_jsonl(CLEAN_AUDIT))
    pending_ids = [problem_id for problem_id in missed if problem_id not in done]
    print(f"clean: {len(pending_ids)} misses, {len(missed) - len(pending_ids)} already recorded", flush=True)
    if not pending_ids:
        return
    problems = {problem["id"]: problem for problem in _load_problems(pending_ids)}
    anchors = _attach_anchors(problems, workers)
    ready = []
    for problem_id in pending_ids:
        problem = problems[problem_id]
        source = anchors.get(problem_id)
        if not source:
            _append(CLEAN_AUDIT, {
                "problem_id": problem_id,
                "status": "unresolved",
                "passed_at": None,
                "steps": [{"model": CODER_SMART, "role": "correct", "passed": False, "reason": "no_anchor"}],
                "diff": None,
                "written": False,
                "temperature": SMART_TEMPERATURE,
            })
            print(f"{problem_id} unresolved no_anchor", flush=True)
            continue
        problem["_working_source"] = source
        ready.append(problem)

    def check(problem, complete, write=False, timeout=30):
        return clean_unsolved(problem, complete, write=True, timeout=timeout, trials=SMART_TRIALS)

    saved = []

    def on_record(record):
        record["temperature"] = SMART_TEMPERATURE
        if record["status"] == "amended":
            save_statement_updates(PROBLEMS_PATH, updates_from_records([record]))
            saved.append(record["problem_id"])
        _append(CLEAN_AUDIT, record)
        reason = ""
        if record.get("steps"):
            reason = record["steps"][0].get("reason") or ""
        print(f"{record['problem_id']} {record['status']} {reason}", flush=True)

    if ready:
        run_checks(ready, complete, workers, on_record, write=True, check=check)
    print(f"PROGRESS: clean done, saved {len(saved)} statement repairs", flush=True)


def _attach_anchors(problems, workers):
    grouped = _ok_programs(list(problems))
    found = {}
    lock = threading.Lock()

    def search(problem_id):
        submission_id, _tried = first_passing_submission(
            grouped.get(problem_id, []),
            problems[problem_id].get("official_tests") or [],
            timeout=ANCHOR_TIMEOUT,
        )
        source = ""
        if submission_id is not None:
            for row in grouped.get(problem_id, []):
                if row["submission_id"] == submission_id:
                    source = row["source"]
                    break
        with lock:
            found[problem_id] = source
        print(f"anchor {problem_id} {submission_id or 'none'}", flush=True)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(search, problem_id) for problem_id in problems]
        for future in as_completed(futures):
            future.result()
    return found


def _run_small(problem_ids, env_file):
    solved = _solved_here(problem_ids)
    if not solved:
        print("PROGRESS: no problem solved by the smart coder in this run", flush=True)
        return
    starters = _starters(solved)
    missing = [problem_id for problem_id in solved if problem_id not in starters]
    if missing:
        raise SystemExit("No failed starter for: " + ", ".join(missing[:10]))
    exporter = _load_script_path(EXPORT_PATH, "export_harbor_task")
    os.makedirs(TASK_ROOT, exist_ok=True)
    left_out = []
    for index, problem_id in enumerate(solved, start=1):
        starter = starters[problem_id]
        name = f"{problem_id.replace('/', '_')}__{starter['submission_id']}"
        out_dir = os.path.join(TASK_ROOT, name)
        if _already_exported(out_dir, starter["submission_id"]):
            continue
        print(f"export {index}/{len(solved)} {problem_id}", flush=True)
        try:
            exporter.export_harbor_task(
                problem_id, starter["submission_id"], out_dir, feedback="", allow_test=ALLOW_TEST
            )
        except SystemExit as exc:
            message = exc.code if isinstance(exc.code, str) else "export refused"
            left_out.append({"problem_id": problem_id, "reason": message})
            print(f"note: skipped {problem_id}: {message}", flush=True)
            continue
        if exporter.program_passes(
            os.path.join(out_dir, "environment", "solution.py"),
            os.path.join(out_dir, "tests", "official_tests.json"),
        ):
            left_out.append({"problem_id": problem_id, "reason": "starter already passes"})
            shutil.rmtree(out_dir)
            print(f"note: skipped {problem_id}: starter already passes", flush=True)
    present = [name for name in os.listdir(TASK_ROOT) if os.path.isdir(os.path.join(TASK_ROOT, name))]
    manifest = {
        "solved_by_smart_coder": len(solved),
        "tasks": len(present),
        "left_out": left_out,
        "n_attempts": SMALL_ATTEMPTS,
        "temperature": SMALL_TEMPERATURE,
        "models": [{"slug": slug, "model": model} for slug, model in SMALL_MODELS],
    }
    os.makedirs(os.path.dirname(MANIFEST_PATH), exist_ok=True)
    with open(MANIFEST_PATH, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")
    print(f"PROGRESS: export done, {len(present)} tasks, {len(left_out)} left out", flush=True)
    if not present:
        print("PROGRESS: no Harbor tasks to run", flush=True)
        return
    codes = []
    for slug, model in SMALL_MODELS:
        codes.append(_run_model(slug, model, env_file))
    summary = _summarize(starters)
    print(f"Wrote {os.path.relpath(SUMMARY_PATH, PROJECT_ROOT)}", flush=True)
    for job in summary["jobs"]:
        print(
            f"result {job['model_slug']} any-attempt {job['n_solved_any_attempt']} of {job['n_problems']}"
            f" problems, trials {job['n_solved_trials']} of {job['n_trials']}",
            flush=True,
        )
    expected = len(present) * SMALL_ATTEMPTS
    if any(code != 0 for code in codes) or any(job["n_trials"] != expected for job in summary["jobs"]):
        print(f"FAILED: a model job did not finish all {expected} trials", flush=True)
        raise SystemExit(1)
    print(f"DONE: {JOB_PREFIX} on {len(present)} problems, {SMALL_ATTEMPTS} attempts each", flush=True)


def _run_model(slug, model, env_file):
    token = str(SMALL_TEMPERATURE).replace(".", "")
    name = f"{JOB_PREFIX}__{slug}__k{SMALL_ATTEMPTS}__t{token}"
    if name.startswith(("cond01", "solved158")) or (JOB_PREFIX != "rest_train" and name.startswith("rest_train")):
        raise SystemExit(f"Refusing to resume an earlier job: {name}")
    command = [
        "uv", "run", "harbor", "run",
        "--path", TASK_ROOT,
        "--agent", "tagent.agents.rewrite_student:RewriteStudent",
        "--model", model,
        "--env-file", env_file,
        "--yes",
        "--n-concurrent", str(SMALL_CONCURRENT),
        "--n-attempts", str(SMALL_ATTEMPTS),
        "--agent-kwarg", f"temperature={SMALL_TEMPERATURE}",
        "--job-name", name,
        "--jobs-dir", JOBS_DIR,
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = os.path.join(PROJECT_ROOT, "src")
    print(f"PROGRESS: harbor {name}", flush=True)
    completed = subprocess.run(command, cwd=PROJECT_ROOT, env=env)
    print(f"PROGRESS: {name} exit {completed.returncode}", flush=True)
    return completed.returncode


def _summarize(starters):
    by_task = {
        f"{problem_id.replace('/', '_')}__{starter['submission_id']}": problem_id
        for problem_id, starter in starters.items()
    }
    jobs = []
    for slug, model in SMALL_MODELS:
        token = str(SMALL_TEMPERATURE).replace(".", "")
        name = f"{JOB_PREFIX}__{slug}__k{SMALL_ATTEMPTS}__t{token}"
        job_dir = os.path.join(JOBS_DIR, name)
        trials = []
        if os.path.isdir(job_dir):
            for entry in sorted(os.listdir(job_dir)):
                result_path = os.path.join(job_dir, entry, "result.json")
                if not os.path.isfile(result_path):
                    continue
                with open(result_path, encoding="utf-8") as handle:
                    result = json.load(handle)
                task_name = (result.get("task_name") or "").split("/", 1)[-1]
                reward = (result.get("verifier_result") or {}).get("reward")
                if reward is None:
                    reward = ((result.get("verifier_result") or {}).get("rewards") or {}).get("reward")
                trials.append({
                    "task": task_name,
                    "problem_id": by_task.get(task_name),
                    "reward": reward,
                    "exception": (result.get("exception_info") or {}).get("exception_type"),
                })
        by_problem = {}
        for trial in trials:
            by_problem.setdefault(trial["problem_id"], []).append(trial["reward"])
        solved_counts = [
            sum(1 for reward in rewards if reward in (1, 1.0))
            for rewards in by_problem.values()
        ]
        solved_counts = [count for count in solved_counts if count]
        jobs.append({
            "job_name": name,
            "model_slug": slug,
            "model": model,
            "n_attempts": SMALL_ATTEMPTS,
            "temperature": SMALL_TEMPERATURE,
            "n_trials": len(trials),
            "n_problems": len(by_problem),
            "n_solved_trials": sum(1 for trial in trials if trial["reward"] in (1, 1.0)),
            "n_solved_any_attempt": len(solved_counts),
            "mean_successful_attempts_when_solved": (
                round(sum(solved_counts) / len(solved_counts), 2) if solved_counts else None
            ),
            "trials": trials,
        })
    summary = {
        "set": JOB_PREFIX,
        "split": SPLIT,
        "attempts": SMALL_ATTEMPTS,
        "temperature": SMALL_TEMPERATURE,
        "jobs": jobs,
    }
    os.makedirs(JOBS_DIR, exist_ok=True)
    with open(SUMMARY_PATH, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")
    return summary


def _solved_here(problem_ids):
    wanted = set(problem_ids)
    solved = []
    for problem_id, record in _latest(SMART_AUDIT).items():
        if problem_id in wanted and record.get("status") == "usable":
            solved.append(problem_id)
    for problem_id, record in _latest(CLEAN_AUDIT).items():
        if problem_id in wanted and record.get("status") == "amended" and problem_id not in solved:
            solved.append(problem_id)
    return sorted(solved, key=_sort_key)


def _starters(problem_ids):
    """One failed submission per problem. Repair200 keeps its recorded starter."""
    wanted = set(problem_ids)
    chosen = {}
    with open(REPAIR_PATH, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row["problem_id"] in wanted:
                chosen[row["problem_id"]] = {
                    "submission_id": str(int(float(row["submission_id"]))),
                    "verdict": row["verdict"],
                }
    best = {}
    fallback = {}
    with open(CASES_PATH, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            problem_id = row["problem_id"]
            if problem_id not in wanted or problem_id in chosen:
                continue
            candidate = {
                "submission_id": str(int(float(row["submission_id"]))),
                "verdict": row["verdict"],
                "passed_test_count": int(row["passed_test_count"]),
                "source_chars": len(row["source"]),
            }
            key = (problem_id, row["verdict"])
            current = fallback.get(key)
            fallback[key] = candidate if current is None else _better(current, candidate)
            if row["verdict"] == "WRONG_ANSWER" and candidate["passed_test_count"] < 1:
                continue
            if row["verdict"] not in PREFERRED:
                continue
            current = best.get(key)
            best[key] = candidate if current is None else _better(current, candidate)
    for problem_id in wanted:
        if problem_id in chosen:
            continue
        starter = None
        for verdict in PREFERRED:
            starter = best.get((problem_id, verdict))
            if starter:
                break
        if starter is None:
            rows = [row for (pid, _verdict), row in fallback.items() if pid == problem_id]
            starter = None
            for row in rows:
                starter = row if starter is None else _better(starter, row)
        if starter:
            chosen[problem_id] = {"submission_id": starter["submission_id"], "verdict": starter["verdict"]}
    return chosen


def _better(left, right):
    left_key = (-left["passed_test_count"], left["source_chars"], int(left["submission_id"]))
    right_key = (-right["passed_test_count"], right["source_chars"], int(right["submission_id"]))
    return left if left_key <= right_key else right


def _ok_programs(problem_ids):
    table = pq.read_table(
        SUBMISSIONS_PATH,
        columns=["problem_id", "submission_id", "verdict", "source"],
        filters=[("problem_id", "in", problem_ids), ("verdict", "==", "OK")],
    )
    grouped = {}
    for row in table.to_pylist():
        source = row.get("source") or ""
        if not source.strip():
            continue
        grouped.setdefault(row["problem_id"], []).append({
            "submission_id": int(float(row["submission_id"])),
            "source": source,
        })
    for rows in grouped.values():
        rows.sort(key=lambda item: (len(item["source"]), item["submission_id"]))
    return grouped


def _load_problems(problem_ids):
    if not problem_ids:
        return []
    table = pq.read_table(PROBLEMS_PATH, columns=STATEMENT_COLUMNS, filters=[("id", "in", problem_ids)])
    by_id = {row["id"]: row for row in table.to_pylist()}
    missing = [problem_id for problem_id in problem_ids if problem_id not in by_id]
    if missing:
        raise SystemExit("Not in the official subset: " + ", ".join(missing[:10]))
    return [by_id[problem_id] for problem_id in problem_ids]


def _repair_ids():
    ids = set()
    with open(REPAIR_PATH, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                ids.add(json.loads(line)["problem_id"])
    return ids


def _no_anchor_ids():
    skipped = set()
    if not os.path.isfile(ANCHOR_PATH):
        return skipped
    for record in _latest(ANCHOR_PATH).values():
        if not record.get("error") and not record.get("anchor_submission_id"):
            skipped.add(record["problem_id"])
    return skipped


def _already_exported(out_dir, submission_id):
    path = os.path.join(out_dir, "task.toml")
    if not os.path.isfile(path):
        return False
    with open(path, encoding="utf-8") as handle:
        return f'faulty_submission_id = "{submission_id}"' in handle.read()


def _latest(path):
    latest = {}
    for record in _read_jsonl(path):
        latest[record["problem_id"]] = record
    return latest


def _read_jsonl(path):
    if not os.path.isfile(path):
        return []
    rows = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _append(path, record):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _sort_key(problem_id):
    contest, index = problem_id.split("/", 1)
    return (int(contest), index)


def _load_script(name):
    return _load_script_path(os.path.join(PROJECT_ROOT, "scripts", name + ".py"), name)


def _load_script_path(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


if __name__ == "__main__":
    main()
