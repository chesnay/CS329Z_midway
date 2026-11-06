"""Which models have already been run on the clean Codeforces subset.

Every problem in data/clean was solved by Qwen3-Coder. The three small
student models were run afterwards, five attempts each at temperature 0.7,
and only on the problems that were exported to Harbor. A clean problem can
therefore have no small-model run. The stopped condition-01 job is not read.
"""

import json
import os

from tagent.data.clean_codeforces import PROJECT_ROOT

JOBS_DIR = os.path.join(PROJECT_ROOT, "data", "example_code", "harbor_jobs")
ATTEMPTS = 5
TEMPERATURE = 0.7

# slug, label shown to a reader, OpenRouter model id.
SMALL_MODELS = (
    ("llama-3.2-1b", "Llama 3.2 1B", "openrouter/meta-llama/llama-3.2-1b-instruct"),
    ("ministral-3b", "Ministral 3B", "openrouter/mistralai/ministral-3b-2512"),
    ("llama-3.2-3b", "Llama 3.2 3B", "openrouter/meta-llama/llama-3.2-3b-instruct"),
)
LARGE_CODER = ("qwen3-coder", "Qwen3-Coder", "qwen/qwen3-coder")

# Train has two Harbor jobs that do not share problems.
JOB_PREFIXES = {
    "train": ("solved158", "rest_train"),
    "dev": ("heldout_dev",),
    "test": ("heldout_test",),
}


def problem_id_from_trial(trial_name):
    """Return the Codeforces id for a Harbor trial directory name.

    ``304_B__38422755__3xoSN6G`` is problem ``304/B``. The last segment is the
    attempt, and the segment before that is the starter submission.
    """
    task = str(trial_name).rsplit("__", 1)[0]
    left = task.split("__", 1)[0]
    contest, index = left.split("_", 1)
    return f"{contest}/{index}"


def tally_reward_stats(reward_stats):
    """Return {problem id: {"passes", "trials"}} from one job's reward map.

    ``reward_stats`` is Harbor's ``reward_stats.reward`` object: score, then
    the trial names with that score. A score of 1.0 is a pass.
    """
    counts = {}
    for score, names in (reward_stats or {}).items():
        passed = float(score) == 1.0
        for name in names:
            row = counts.setdefault(problem_id_from_trial(name), {"passes": 0, "trials": 0})
            row["trials"] += 1
            if passed:
                row["passes"] += 1
    return counts


def job_name(prefix, slug):
    """Harbor job directory for one split prefix and one small model."""
    return f"{prefix}__{slug}__k{ATTEMPTS}__t0{int(TEMPERATURE * 10)}"


def small_model_runs(split, jobs_dir=JOBS_DIR):
    """Return {problem id: {model slug: run}} for one clean split.

    A run is ``passes``, ``trials``, and ``job`` (the Harbor prefix). Problems
    that were never exported are absent. The two train jobs may not both
    contain the same problem.
    """
    if split not in JOB_PREFIXES:
        raise ValueError(f"split must be one of {tuple(JOB_PREFIXES)}, not {split!r}")
    found = {}
    missing_jobs = []
    for prefix in JOB_PREFIXES[split]:
        for slug, _label, _model in SMALL_MODELS:
            path = os.path.join(jobs_dir, job_name(prefix, slug), "result.json")
            if not os.path.isfile(path):
                missing_jobs.append(os.path.basename(os.path.dirname(path)))
                continue
            with open(path, encoding="utf-8") as handle:
                result = json.load(handle)
            evals = (result.get("stats") or {}).get("evals") or {}
            if not evals:
                missing_jobs.append(os.path.basename(os.path.dirname(path)))
                continue
            reward_stats = (next(iter(evals.values())).get("reward_stats") or {}).get("reward") or {}
            for problem_id, counts in tally_reward_stats(reward_stats).items():
                slot = found.setdefault(problem_id, {})
                if slug in slot:
                    raise RuntimeError(f"{problem_id} is in two {slug} jobs")
                slot[slug] = {"passes": counts["passes"], "trials": counts["trials"], "job": prefix}
    return found, missing_jobs


def format_run(run):
    """Return ``2/5`` for a finished run, or ``not run`` when there is none."""
    if not run:
        return "not run"
    return f"{run['passes']}/{run['trials']}"
