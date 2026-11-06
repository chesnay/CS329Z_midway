"""Decide whether a stored Codeforces instruction can be solved.

A competent coding model sees the statement and the samples. It does not see
hidden tests, an accepted program, or a failed starter. Qwen3-Coder 30B-A3B
Instruct tries first. Qwen2.5-Coder 7B Instruct is no longer on the OpenRouter
catalog. If that program fails the official tests, Qwen2.5-Coder 32B Instruct
tries the same instruction. The stored statement is kept when either program
passes. A failure of the first model is not, by itself, a bad statement.

The third call happens only when the 32B model also fails. That reply proposes
one small correction of the statement fields and a Python 3 program written
from the corrected text. The correction is accepted only when the program
passes every official test and the new text does not contain a hidden
expected output.

OpenRouter ids, checked against the catalog before a live call:

- ``qwen/qwen3-coder-30b-a3b-instruct``
- ``qwen/qwen-2.5-coder-32b-instruct``
"""

import json
import os
import re
import subprocess
import sys
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import pyarrow as pa
import pyarrow.parquet as pq

from tagent.data.statement_math import STATEMENT_FIELDS

CODER_FIRST = "qwen/qwen3-coder-30b-a3b-instruct"
CODER_32B = "qwen/qwen-2.5-coder-32b-instruct"
# Larger coder used only to test a statement that was corrected with a working solution.
CODER_SMART = "qwen/qwen3-coder"
MODELS = (CODER_FIRST, CODER_32B)
# Same per-test limit as export_harbor_task.program_passes.
GRADE_TIMEOUT = 30
ROW_GROUP_PROBLEMS = 4

_FENCE = re.compile(r"```(?:python|json)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)


def require_models(present, needed=None):
    """Raise SystemExit when a required coder id is absent from the catalog."""
    missing = [model for model in (needed or MODELS) if model not in present]
    if missing:
        raise SystemExit("OpenRouter catalog is missing " + ", ".join(missing))


def ensure_train(problem_id, split):
    """This check reads train problems only. Dev and test stay unread."""
    if split != "train":
        label = split or "not in a split"
        raise SystemExit(f"{problem_id} is {label}; this check reads train problems only")


def build_solve_prompt(problem):
    """The instruction a solver sees. Official tests are not included."""
    return (
        "Write a Python 3 program that solves this Codeforces problem. "
        "It reads standard input and writes the answer to standard output. "
        "Reply with one fenced python block and no other text.\n\n"
        + _public_statement(problem)
    )


def build_correction_prompt(problem):
    """Ask for one small statement correction and a program from that text."""
    fields = ", ".join(f'"{name}"' for name in STATEMENT_FIELDS)
    return (
        "The stored statement may be missing a superscript or a subscript. "
        "Correct only a flattened formula. Copy every statement field you are not changing. "
        "Then write a Python 3 program from your corrected statement. "
        "Reply with one JSON object and no other text, using these keys: "
        f"{fields}, and \"program\". "
        "Do not include samples, hidden tests, or expected outputs.\n\n"
        + _public_statement(problem)
    )


def build_solution_correction_prompt(problem, source):
    """Ask for a statement fix, using a working solution as private evidence.

    The solution is not part of the instruction a later solver sees.
    """
    fields = ", ".join(f'"{name}"' for name in STATEMENT_FIELDS)
    return (
        "The Python 3 program below is a working solution. "
        "The statement may have a flattened superscript or subscript. "
        "Correct only those formulas so a programmer can solve the problem from the statement alone. "
        "Copy every statement field you are not changing. Leave an empty field empty. "
        "Do not shorten the statement. Do not copy the solution into the statement. "
        "Reply with one JSON object and no other text, using these keys: "
        f"{fields}.\n\n"
        + _public_statement(problem)
        + "\n## Working solution\n\n```python\n"
        + source.strip()
        + "\n```\n"
    )


def parse_statement_fields(reply):
    """Return statement fields from a JSON object, or None when the reply is not one."""
    data = _load_json_object(reply)
    if data is None:
        return None
    fields = {}
    for name in STATEMENT_FIELDS:
        value = data.get(name)
        if value is None:
            continue
        if not isinstance(value, str):
            return None
        fields[name] = value
    return fields


def correction_only_adds_math(problem, corrected):
    """True when each changed field only inserts math markup into the stored text.

    A space-only edit, a deleted brace, or a rewritten sentence is not a
    formula repair. Markup is ``^ _ { } $ \\``.
    """
    for name, new in corrected.items():
        old = problem.get(name) or ""
        if new != old and not _adds_only_math_markup(old, new):
            return False
    return True


def _adds_only_math_markup(old, new):
    """True when ``new`` is ``old`` plus only math markup, spaces included.

    Deleting a space, a brace, or a letter fails. Markup is ``^ _ { } $ \\``.
    """
    if new == old:
        return False
    markup = set("^_{}$\\")
    return "".join(char for char in new if char not in markup) == old


def correction_keeps_substance(problem, corrected):
    """Reject a rewrite that invents an empty field or drops most of a long field."""
    for name, new in corrected.items():
        old = problem.get(name) or ""
        if not old.strip() and new.strip():
            return False
        if len(old) >= 100 and len(new) < int(0.8 * len(old)):
            return False
    return True


def statement_contains_solution(text, source):
    """True when the statement quotes the working solution."""
    program = (source or "").strip()
    if len(program) >= 40 and program in text:
        return True
    for line in program.splitlines():
        line = line.strip()
        if len(line) >= 40 and line in text:
            return True
    return False


def extract_python(reply):
    """Return the first fenced block, or None when the reply has no fence."""
    if not reply:
        return None
    match = _FENCE.search(reply)
    if not match:
        return None
    code = match.group(1).strip()
    return code or None


def parse_revision(reply):
    """Return (statement fields, program) from the third reply.

    Statement keys other than the stored statement fields are ignored. The
    result is None when the reply is not a JSON object with a program string.
    """
    data = _load_json_object(reply)
    if data is None:
        return None
    program = data.get("program")
    if not isinstance(program, str) or not program.strip():
        return None
    fields = {}
    for name in STATEMENT_FIELDS:
        value = data.get(name)
        if value is None:
            continue
        if not isinstance(value, str):
            return None
        fields[name] = value
    code = extract_python(program) or program.strip()
    return fields, code


def statement_diff(problem, corrected):
    """Fields whose new text differs from the stored statement."""
    diff = {}
    for name, new in corrected.items():
        old = problem.get(name) or ""
        if new != old:
            diff[name] = {"from": old, "to": new}
    return diff


def correction_is_allowed(problem, corrected):
    """A correction may change statement fields only, and must not leak a hidden output."""
    if not corrected:
        return False
    for name in corrected:
        if name not in STATEMENT_FIELDS:
            return False
    if not statement_diff(problem, corrected):
        return False
    merged = dict(problem)
    merged.update(corrected)
    leaked = _hidden_output_leaks(_public_statement(merged), merged)
    return not leaked


def program_passes_source(source, tests, timeout=GRADE_TIMEOUT):
    """True when the source matches every official test.

    Whitespace is ignored, same as ``tokens_match``. Each test gets ``timeout``
    seconds. The process is the local interpreter, same as
    ``export_harbor_task.program_passes``.
    """
    if not source or not tests:
        return False
    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, "solution.py")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(source)
        for test in tests:
            try:
                result = subprocess.run(
                    [sys.executable, path],
                    input=test.get("input") or "",
                    text=True,
                    capture_output=True,
                    timeout=timeout,
                )
            except subprocess.TimeoutExpired:
                return False
            if result.returncode != 0 or not _tokens_match(result.stdout, test.get("output") or ""):
                return False
    return True


def find_anchor(programs, tests, timeout=8):
    """Return the first source that passes every stored test, or None.

    Identical sources are tried once. The caller chooses the order.
    """
    seen = set()
    for source in programs:
        if not source or source in seen:
            continue
        seen.add(source)
        if program_passes_source(source, tests, timeout=timeout):
            return source
    return None


def first_passing_submission(programs, tests, timeout=8):
    """Return ``(submission_id, n_tried)`` for the first source that passes.

    ``programs`` is an iterable of ``{"submission_id", "source"}`` in the order
    to try. Identical sources count once. ``(None, n_tried)`` means none passed.
    A passing program is the anchor: the stored tests accept it.
    """
    seen = set()
    tried = 0
    for program in programs:
        source = program.get("source") or ""
        if not source or source in seen:
            continue
        seen.add(source)
        tried += 1
        if program_passes_source(source, tests, timeout=timeout):
            return program.get("submission_id"), tried
    return None, tried


def propose_from_solution(problem, complete, write=False, timeout=GRADE_TIMEOUT):
    """Ask the 32B coder for a formula repair and keep the candidate text.

    The smart coder is not called. ``write`` and ``timeout`` match the
    ``run_checks`` callback and are ignored: this never grades a program and
    never writes the problems table. A formula repair has status ``candidate``.
    Any other reply stays ``unresolved`` and still keeps the diff when there is one.
    """
    del write, timeout
    source = problem.get("_working_source") or ""
    steps = [{"model": CODER_32B, "role": "propose", "passed": False}]
    if not source.strip():
        steps[0]["reason"] = "no_solution"
        return _solution_record(problem, "unresolved", None, steps, None, False)
    fields = parse_statement_fields(complete(CODER_32B, build_solution_correction_prompt(problem, source)))
    if not fields or not statement_diff(problem, fields):
        steps[0]["reason"] = "no_change"
        return _solution_record(problem, "unresolved", None, steps, None, False)
    if not correction_is_allowed(problem, fields):
        steps[0]["reason"] = "hidden_output"
        return _solution_record(problem, "unresolved", None, steps, statement_diff(problem, fields), False)
    if not correction_keeps_substance(problem, fields):
        steps[0]["reason"] = "shortened"
        return _solution_record(problem, "unresolved", None, steps, statement_diff(problem, fields), False)
    merged = {key: value for key, value in problem.items() if key != "_working_source"}
    merged.update(fields)
    if statement_contains_solution(_public_statement(merged), source):
        steps[0]["reason"] = "solution_leak"
        return _solution_record(problem, "unresolved", None, steps, statement_diff(problem, fields), False)
    if not correction_only_adds_math(problem, fields):
        steps[0]["reason"] = "not_a_formula"
        return _solution_record(problem, "unresolved", None, steps, statement_diff(problem, fields), False)
    steps[0]["passed"] = True
    steps[0]["reason"] = "formula"
    return _solution_record(problem, "candidate", None, steps, statement_diff(problem, fields), False)


def correct_from_solution(problem, complete, write=False, timeout=GRADE_TIMEOUT):
    """Repair a statement with a working solution, then test it without that solution.

    ``problem["_working_source"]`` is the shortest accepted program. The solver
    sees only the corrected statement and the samples. The parquet is updated
    by the caller, and only when ``write`` is true and the solver passes.
    """
    source = problem.get("_working_source") or ""
    steps = []
    if not source.strip():
        steps.append({"model": CODER_32B, "role": "correct", "passed": False, "reason": "no_solution"})
        return _solution_record(problem, "unresolved", None, steps, None, False)

    reply = complete(CODER_32B, build_solution_correction_prompt(problem, source))
    fields = parse_statement_fields(reply)
    if not fields or not statement_diff(problem, fields):
        steps.append({"model": CODER_32B, "role": "correct", "passed": False, "reason": "no_change"})
        return _solution_record(problem, "unresolved", None, steps, None, False)
    if not correction_is_allowed(problem, fields):
        steps.append({"model": CODER_32B, "role": "correct", "passed": False, "reason": "hidden_output"})
        return _solution_record(problem, "unresolved", None, steps, None, False)
    if not correction_keeps_substance(problem, fields):
        steps.append({"model": CODER_32B, "role": "correct", "passed": False, "reason": "shortened"})
        return _solution_record(problem, "unresolved", None, steps, None, False)
    merged = {key: value for key, value in problem.items() if key != "_working_source"}
    merged.update(fields)
    public = _public_statement(merged)
    if statement_contains_solution(public, source):
        steps.append({"model": CODER_32B, "role": "correct", "passed": False, "reason": "solution_leak"})
        return _solution_record(problem, "unresolved", None, steps, None, False)
    if not correction_only_adds_math(problem, fields):
        steps.append({"model": CODER_32B, "role": "correct", "passed": False, "reason": "not_a_formula"})
        return _solution_record(problem, "unresolved", None, steps, None, False)

    steps.append({"model": CODER_32B, "role": "correct", "passed": True})
    solve_prompt = build_solve_prompt(merged)
    solved = _solved(complete(CODER_SMART, solve_prompt), problem.get("official_tests") or [], timeout)
    steps.append({"model": CODER_SMART, "role": "solve", "passed": solved})
    if not solved:
        return _solution_record(problem, "unresolved", None, steps, None, False)
    status = "amended" if write else "would_amend"
    return _solution_record(problem, status, "smart", steps, statement_diff(problem, fields), bool(write))


def retry_unsolved(problem, complete, write=False, timeout=GRADE_TIMEOUT):
    """Try one formula repair, then ask the smart coder to solve.

    ``problem["_working_source"]`` is an accepted program that already passes
    the stored tests. The solver does not see it. A markup-only repair is the
    statement the solver sees. Any other reply leaves the stored statement in
    place, and the solver tries that text again. The caller writes the parquet
    only when ``write`` is true and the solver passes the repaired statement.
    """
    source = problem.get("_working_source") or ""
    public = {key: value for key, value in problem.items() if key != "_working_source"}
    fields, reason = _formula_fields(problem, public, source, complete)
    steps = [{"model": CODER_32B, "role": "correct", "passed": reason == "formula", "reason": reason}]
    merged = dict(public)
    if fields:
        merged.update(fields)
    solved = _solved(complete(CODER_SMART, build_solve_prompt(merged)), problem.get("official_tests") or [], timeout)
    steps.append({"model": CODER_SMART, "role": "solve", "passed": solved})
    diff = statement_diff(problem, fields) if fields else None
    if fields and solved:
        status = "amended" if write else "would_amend"
        return _solution_record(problem, status, "smart", steps, diff, bool(write))
    if solved:
        return _solution_record(problem, "usable", "smart", steps, None, False)
    return _solution_record(problem, "unresolved", None, steps, None, False)


def _formula_fields(problem, public, source, complete, model=CODER_32B):
    """Return ``(fields, reason)``. ``fields`` is set only for a markup-only repair."""
    if not source.strip():
        return None, "no_solution"
    parsed = parse_statement_fields(complete(model, build_solution_correction_prompt(problem, source)))
    if not parsed or not statement_diff(problem, parsed):
        return None, "no_change"
    if not correction_is_allowed(problem, parsed):
        return None, "hidden_output"
    if not correction_keeps_substance(problem, parsed):
        return None, "shortened"
    merged = dict(public)
    merged.update(parsed)
    if statement_contains_solution(_public_statement(merged), source):
        return None, "solution_leak"
    if not correction_only_adds_math(problem, parsed):
        return None, "not_a_formula"
    return parsed, "formula"


def smart_trials(problem, complete, trials=3, write=False, timeout=GRADE_TIMEOUT):
    """Ask the smart coder up to ``trials`` times and stop at the first pass.

    The problems table is never updated. A working solution attached to the
    row is not shown. ``n_passed`` is 1 or 0 because the calls stop early.
    """
    del write
    if trials < 1:
        raise ValueError("trials must be at least 1")
    public = {key: value for key, value in problem.items() if key != "_working_source"}
    prompt = build_solve_prompt(public)
    steps = []
    for index in range(trials):
        reply = complete(CODER_SMART, prompt)
        code = extract_python(reply)
        passed = bool(code) and program_passes_source(code, problem.get("official_tests") or [], timeout)
        step = {"model": CODER_SMART, "role": "solve", "passed": passed, "trial": index + 1}
        if not code:
            step["reason"] = "unreadable"
        elif not passed:
            step["reason"] = "solver_failed"
        steps.append(step)
        if passed:
            break
    solved = bool(steps) and steps[-1]["passed"]
    return {
        "problem_id": problem["id"],
        "status": "usable" if solved else "unresolved",
        "passed_at": "smart" if solved else None,
        "models": {"solver": CODER_SMART},
        "steps": steps,
        "n_tried": len(steps),
        "n_passed": 1 if solved else 0,
        "diff": None,
        "written": False,
    }


def clean_unsolved(problem, complete, write=False, timeout=GRADE_TIMEOUT, trials=3):
    """Repair a missed statement with math markup, then retest it.

    ``problem["_working_source"]`` is an accepted program that already passes
    the stored tests. The smart coder proposes the repair and then solves the
    repaired statement, up to ``trials`` times, without seeing that program.
    The stored statement is not tried again. The caller writes the parquet
    only when ``write`` is true and one repaired attempt passes.
    """
    source = problem.get("_working_source") or ""
    public = {key: value for key, value in problem.items() if key != "_working_source"}
    fields, reason = _formula_fields(problem, public, source, complete, model=CODER_SMART)
    steps = [{"model": CODER_SMART, "role": "correct", "passed": reason == "formula", "reason": reason}]
    if not fields:
        return _solution_record(problem, "unresolved", None, steps, None, False)
    merged = dict(public)
    merged.update(fields)
    passed_step = None
    for index in range(trials):
        solved = _solved(
            complete(CODER_SMART, build_solve_prompt(merged)),
            problem.get("official_tests") or [],
            timeout,
        )
        passed_step = {"model": CODER_SMART, "role": "solve", "passed": solved, "trial": index + 1}
        steps.append(passed_step)
        if solved:
            break
    if not passed_step or not passed_step["passed"]:
        return _solution_record(problem, "unresolved", None, steps, None, False)
    status = "amended" if write else "would_amend"
    return _solution_record(problem, status, "smart", steps, statement_diff(problem, fields), bool(write))


def rest_train_problem_ids(train_problem_ids, already_solved, skipped=()):
    """Usable train problems a strong coder has not passed.

    ``train_problem_ids`` must already be train problems. Dev and test ids
    do not belong in either argument. ``skipped`` are problems whose stored
    tests reject every accepted program.
    """
    chosen = set(train_problem_ids) - set(already_solved) - set(skipped)
    return sorted(chosen, key=_problem_sort_key)


def _problem_sort_key(problem_id):
    contest, index = str(problem_id).split("/", 1)
    return (int(contest), index)


def solve_with_smart(problem, complete, write=False, timeout=GRADE_TIMEOUT):
    """Ask the smart coder to solve the stored statement.

    ``write`` is accepted so this can be the ``run_checks`` callback. The
    problems table is never updated. A working solution attached to the row
    is not shown.
    """
    del write
    public = {key: value for key, value in problem.items() if key != "_working_source"}
    reply = complete(CODER_SMART, build_solve_prompt(public))
    code = extract_python(reply)
    solved = bool(code) and program_passes_source(code, problem.get("official_tests") or [], timeout)
    step = {"model": CODER_SMART, "role": "solve", "passed": solved}
    if not code:
        step["reason"] = "unreadable"
    elif not solved:
        step["reason"] = "solver_failed"
    return {
        "problem_id": problem["id"],
        "status": "usable" if solved else "unresolved",
        "passed_at": "smart" if solved else None,
        "models": {"solver": CODER_SMART},
        "steps": [step],
        "diff": None,
        "written": False,
    }


def check_problem(problem, complete, write=False, timeout=GRADE_TIMEOUT):
    """Run at most three model calls. ``complete(model, prompt)`` returns the reply text.

    ``write`` only changes the recorded status. The caller writes the parquet.
    """
    tests = problem.get("official_tests") or []
    prompt = build_solve_prompt(problem)
    steps = []

    first = complete(CODER_FIRST, prompt)
    first_pass = _solved(first, tests, timeout)
    steps.append({"model": CODER_FIRST, "role": "solve", "passed": first_pass})
    if first_pass:
        return _record(problem, "usable", "first", steps, None, False)

    second = complete(CODER_32B, prompt)
    second_pass = _solved(second, tests, timeout)
    steps.append({"model": CODER_32B, "role": "solve", "passed": second_pass})
    if second_pass:
        return _record(problem, "usable", "32b", steps, None, False)

    third = complete(CODER_32B, build_correction_prompt(problem))
    parsed = parse_revision(third)
    if parsed is None:
        steps.append({"model": CODER_32B, "role": "correct", "passed": False, "reason": "unreadable"})
        return _record(problem, "unresolved", None, steps, None, False)

    fields, code = parsed
    passed = program_passes_source(code, tests, timeout)
    diff = statement_diff(problem, fields)
    if not passed:
        steps.append({"model": CODER_32B, "role": "correct", "passed": False})
        return _record(problem, "unresolved", None, steps, None, False)
    if not diff:
        steps.append({"model": CODER_32B, "role": "correct", "passed": True})
        return _record(problem, "usable", "32b", steps, None, False)
    if not correction_is_allowed(problem, fields):
        steps.append({
            "model": CODER_32B,
            "role": "correct",
            "passed": True,
            "accepted": False,
            "reason": "hidden_output",
        })
        return _record(problem, "unresolved", None, steps, None, False)

    status = "amended" if write else "would_amend"
    steps.append({"model": CODER_32B, "role": "correct", "passed": True, "accepted": bool(write)})
    return _record(problem, status, "32b_correction", steps, diff, bool(write))


def apply_statement_updates(table, updates):
    """Return a copy of the table with statement fields replaced for the given ids.

    ``updates`` maps a problem id to ``{field: new text}``. Every other column,
    including samples and official tests, is the same array content.
    """
    if not updates:
        return table
    ids = table.column("id").to_pylist()
    columns = []
    for name in table.column_names:
        column = table.column(name)
        if name not in STATEMENT_FIELDS:
            columns.append(column)
            continue
        values = column.to_pylist()
        for index, problem_id in enumerate(ids):
            change = updates.get(problem_id)
            if change and name in change:
                values[index] = change[name]
        columns.append(pa.array(values, type=column.type))
    return pa.Table.from_arrays(columns, schema=table.schema)


class StopChecks(Exception):
    """A failure that should stop every worker, such as a rejected key."""


def run_checks(problems, complete, workers, on_record, write=False, timeout=GRADE_TIMEOUT, check=None):
    """Check problems with ``workers`` calls in flight.

    ``on_record`` runs one at a time, so the caller can append the audit and
    write the parquet without two workers replacing the same file.
    """
    if workers < 1:
        raise SystemExit("workers must be at least 1")
    lock = threading.Lock()
    stop = threading.Event()
    records = []
    check = check or check_problem

    def work(problem):
        if stop.is_set():
            return None
        try:
            record = check(problem, complete, write=write, timeout=timeout)
        except SystemExit as error:
            message = str(error)
            if message.startswith("OpenRouter returned 401") or message.startswith("OpenRouter catalog"):
                stop.set()
                raise StopChecks(message) from None
            record = _error_record(problem, message)
        except Exception as error:
            record = _error_record(problem, f"{type(error).__name__}: {error}")
        with lock:
            on_record(record)
            records.append(record)
        return record

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(work, problem) for problem in problems]
        try:
            for future in as_completed(futures):
                future.result()
        except StopChecks as error:
            pool.shutdown(wait=False, cancel_futures=True)
            raise SystemExit(str(error)) from None
    return records


def _solution_record(problem, status, passed_at, steps, diff, written):
    return {
        "problem_id": problem["id"],
        "status": status,
        "passed_at": passed_at,
        "models": {"corrector": CODER_32B, "solver": CODER_SMART},
        "steps": steps,
        "diff": diff,
        "written": written,
    }


def _error_record(problem, message):
    return {
        "problem_id": problem["id"],
        "status": "error",
        "passed_at": None,
        "models": {"first": CODER_FIRST, "32b": CODER_32B},
        "steps": [],
        "diff": None,
        "written": False,
        "error": message,
    }


def ids_already_checked(records):
    """Problem ids whose latest audit row is finished.

    An ``error`` row is not finished, so a later run tries that problem again.
    """
    latest = {}
    for record in records:
        latest[record.get("problem_id")] = record.get("status")
    finished = {"usable", "unresolved", "amended", "would_amend"}
    return {problem_id for problem_id, status in latest.items() if status in finished}


def updates_from_records(records):
    """Statement text to save. Only an accepted amendment contributes."""
    updates = {}
    for record in records:
        if record["status"] != "amended" or not record["diff"]:
            continue
        updates[record["problem_id"]] = {
            field: change["to"] for field, change in record["diff"].items()
        }
    return updates


def save_statement_updates(path, updates):
    """Write accepted statement edits into the problems parquet.

    The file is replaced only after samples, official tests, and the other
    columns round-trip. A second call with the same edits changes those fields
    again only when the text still differs; identical text is not a new edit.
    """
    if not updates:
        return 0
    original = pq.read_table(path)
    group_sizes = _row_group_sizes(path)
    revised = apply_statement_updates(original, updates)
    temporary = path + ".tmp"
    try:
        pq.write_table(revised, temporary, compression="zstd", row_group_size=ROW_GROUP_PROBLEMS)
        written = pq.read_table(temporary)
        _check_round_trip(original, revised, written)
        if _row_group_sizes(temporary) != group_sizes:
            raise SystemExit("Row groups changed; the original file was left in place.")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)
    return len(updates)


def _public_statement(problem):
    samples = []
    for index, pair in enumerate(problem.get("examples") or [], start=1):
        samples.append(
            f"Sample {index} input:\n{(pair.get('input') or '').rstrip()}\n\n"
            f"Sample {index} output:\n{(pair.get('output') or '').rstrip()}"
        )
    sample_text = "\n\n".join(samples) if samples else "(no samples)"
    note = (problem.get("note") or "").strip()
    note_block = f"\n\n## Note\n\n{note}" if note else ""
    interaction = (problem.get("interaction_format") or "").strip()
    interaction_block = f"\n\n## Interaction\n\n{interaction}" if interaction else ""
    return f"""# {problem['id']}: {problem.get('title') or ''}

Time limit: {problem.get('time_limit')} seconds. Memory limit: {problem.get('memory_limit')} MB.

## Statement

{(problem.get('description') or '').strip()}

## Input

{(problem.get('input_format') or '').strip() or '(none)'}

## Output

{(problem.get('output_format') or '').strip() or '(none)'}{note_block}{interaction_block}

## Samples

{sample_text}
"""


def _load_json_object(reply):
    if not reply:
        return None
    text = reply.strip()
    match = _FENCE.search(text)
    if match:
        text = match.group(1).strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    return data


def _solved(reply, tests, timeout):
    code = extract_python(reply)
    return bool(code) and program_passes_source(code, tests, timeout)


def _record(problem, status, passed_at, steps, diff, written):
    return {
        "problem_id": problem["id"],
        "status": status,
        "passed_at": passed_at,
        "models": {"first": CODER_FIRST, "32b": CODER_32B},
        "steps": steps,
        "diff": diff,
        "written": written,
    }


def _tokens_match(actual, expected):
    """Codeforces ignores extra whitespace. Same rule as the project grader."""
    return (actual or "").split() == (expected or "").split()


def _hidden_output_leaks(instruction, problem):
    """Hidden expected outputs that appear as their own line or block.

    Same rule as ``export_harbor_task.hidden_output_leaks``. A short output
    such as ``1`` is not a leak merely because the statement contains that
    digit inside another number. Sample outputs and sample inputs are public.
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


def _row_group_sizes(path):
    parquet = pq.ParquetFile(path)
    return [parquet.metadata.row_group(index).num_rows for index in range(parquet.metadata.num_row_groups)]


def _same_schema(left, right):
    return left.names == right.names and [left.field(name).type for name in left.names] == [
        right.field(name).type for name in right.names
    ]


def _check_round_trip(original, revised, written):
    if not _same_schema(written.schema, original.schema):
        raise SystemExit("Schema changed; the original file was left in place.")
    if written.num_rows != original.num_rows:
        raise SystemExit("Row count changed; the original file was left in place.")
    for name in original.column_names:
        expected = revised.column(name) if name in STATEMENT_FIELDS else original.column(name)
        if written.column(name).to_pylist() != expected.to_pylist():
            raise SystemExit(f"Column {name} did not round-trip; the original file was left in place.")
