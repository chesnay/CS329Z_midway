# Using `data/clean/`

The tables hold the Codeforces problems Qwen3-Coder solved from the public statement, and every Python 3 submission to those problems. [CLEANING.md](CLEANING.md) says why the cut was made. [DATA_DICTIONARY.md](DATA_DICTIONARY.md) lists the columns.

Run the commands from the repository root, the directory that contains `scripts/` and `data/`.

## See what is in the subset

`scripts/summarize_clean_codeforces.py` prints a short explanation, then the counts. For each split it reports the rating bands, the problem indexes, the official-test sizes, the submissions per problem, the verdicts, and how far the three small models got.

```bash
uv run python scripts/summarize_clean_codeforces.py
```

Qwen3-Coder (`qwen/qwen3-coder`) has already passed every problem in the subset. That pass is the reason the problem is here. The three small models are Llama 3.2 1B, Ministral 3B, and Llama 3.2 3B. Each had five separate attempts at temperature 0.7. A problem counts as solved for a small model when any one of those attempts passes the official tests.

On 5 November 2026 the summary was:

| Split | Problems | Submissions | Small-model runs | Not run |
| --- | ---: | ---: | ---: | ---: |
| train | 229 | 75,049 | 224 | 5 |
| dev | 123 | 52,726 | 121 | 2 |
| test | 107 | 40,940 | 106 | 1 |

"Not run" means the problem is in the clean tables, and the small models were not given it. The eight problems are:

| Problem | Split | Why it was not exported |
| --- | --- | --- |
| `195/B` | train | The student instruction would have contained a hidden expected output. |
| `177/A1`, `236/B`, `544/A`, `1298/C` | train | The chosen starter already passes the official tests, so there was nothing to repair. |
| `112/A`, `887/A` | dev | The chosen starter already passes. |
| `208/A` | test | The chosen starter already passes. |

The stopped condition-01 job is not included. Some of its trials used statements from before the math repair.

Among the problems that were run, the passing attempts were:

| Model | Train | Dev | Test |
| --- | --- | --- | --- |
| Llama 3.2 1B | 12 of 224 problems, 18 of 1120 trials | 7 of 121, 11 of 605 | 5 of 106, 5 of 530 |
| Ministral 3B | 86 of 224, 218 of 1120 | 52 of 121, 142 of 604 | 48 of 106, 130 of 530 |
| Llama 3.2 3B | 32 of 224, 61 of 1120 | 20 of 121, 30 of 605 | 15 of 106, 25 of 530 |

Ministral's dev count is 142 of 604 because one trial errored and has no score. The other denominators are five attempts times the number of problems that were run.

These train numbers are not the earlier 328-problem table. That table also included 104 train problems solved only by the 30B or 32B coder. Those 104 are not in `data/clean/`.

## List a split

`scripts/list_clean_codeforces.py` prints one line per problem. The line has the id, rating, number of official tests, number of submissions, number accepted, and one cell per small model. A cell is `2/5` when two of five attempts passed, or `not run`.

```bash
uv run python scripts/list_clean_codeforces.py --split train
uv run python scripts/list_clean_codeforces.py --split dev
uv run python scripts/list_clean_codeforces.py --split test
```

## List one problem's submissions

`--problem` checks that the id is in that split, names the models that have run, and prints each submission's id, verdict, and how many tests Codeforces counted as passed. It does not print the program, the statement, or the official tests.

```bash
uv run python scripts/list_clean_codeforces.py --split dev --problem 26/A
```

`--source` prints one program. The id has to belong to that problem.

```bash
uv run python scripts/list_clean_codeforces.py --split dev --problem 26/A --source 101817015
```

## See only the model grid

`scripts/show_clean_runs.py` is the same small-model grid without the submission counts. The last column is the Harbor job the row came from: `solved158` or `rest_train` on train, `heldout_dev` on dev, and `heldout_test` on test.

```bash
uv run python scripts/show_clean_runs.py --split train
uv run python scripts/show_clean_runs.py --split test --problem 84/A
```

A pipe that closes early, such as `head`, stops the listing without an error.

## Load the tables from Python

The scripts add `src` to the path themselves. Another program run from the repository root needs the same path.

```bash
PYTHONPATH=src uv run python
```

```python
from tagent.data.clean_codeforces import (
    load_problem,
    load_submissions,
    load_submission,
    split_of,
    problem_ids,
    problems_table,
    submissions_table,
)

split_of("26/A")            # "dev", or None if the problem is not in the subset
problem_ids("train")        # ids in table order
problem = load_problem("26/A")
submissions = load_submissions("26/A")
one = load_submission("101817015")

train_problems = problems_table("train")
train_submissions = submissions_table("train")
```

`load_problem` returns one dict, or `None`. `load_submissions` returns a list, empty when the problem is not in the subset. `problems_table` and `submissions_table` return a PyArrow table for one of `train`, `dev`, or `test`.

To read a run without the viewer:

```python
from tagent.data.clean_runs import small_model_runs, format_run

runs, missing_jobs = small_model_runs("dev")
print(format_run(runs["26/A"]["ministral-3b"]))  # "2/5"
```

The slug is `llama-3.2-1b`, `ministral-3b`, or `llama-3.2-3b`. A missing key means that model was not run on that problem.

## What a student prompt may contain

A repair prompt may contain the statement fields, the public samples (`examples`), one rejected `source`, and the teacher line. It must not contain `official_tests`, the `editorial`, or an accepted program. The official tests are in the problem row so the subset can be graded. Holding them out of the prompt is the caller's job. The viewers do not print the tests.

`input_mode` is `stdio` for every problem here, and no row has a custom checker, so a program can be graded by running it and comparing tokens. That is the comparison the coder screen and the Harbor jobs used.
