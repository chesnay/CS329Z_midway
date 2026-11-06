# Data dictionary for `data/clean/`

This dictionary describes the columns in the clean Codeforces tables. The reason a problem is included, and the problems that were left out, are in [CLEANING.md](CLEANING.md). How to load and list the tables is in [USAGE.md](USAGE.md).

The tables are a subset of `data/interim/codeforces_official_subset/`. Column names and types are unchanged. Counts below were taken from the clean tables on 5 November 2026: 459 problems and 168,715 submissions.

## Files

| Path | Rows | What a row is |
| --- | ---: | --- |
| `problems/train.parquet` | 229 | one train problem |
| `problems/dev.parquet` | 123 | one dev problem |
| `problems/test.parquet` | 107 | one test problem |
| `submissions/train.parquet` | 75,049 | one Python 3 submission to a train problem |
| `submissions/dev.parquet` | 52,726 | one Python 3 submission to a dev problem |
| `submissions/test.parquet` | 40,940 | one Python 3 submission to a test problem |
| `manifest.json` | | the model, the inclusion rule, the audit counts, and the problem ids |

Problem rows are sorted by contest number, then index, in row groups of four. Submission rows are sorted by problem id, then submission id. A submission is stored in the same split as its problem. There is no separate split column. The directory name is the split.

Model runs are not columns in these tables. Qwen3-Coder passed every problem here. The three small models were run later, and those scores live in the Harbor job files. [USAGE.md](USAGE.md) says how to read them.

## Problem columns

| Column | Type | Meaning |
| --- | --- | --- |
| `id` | string | Problem id, `contest/index`, such as `26/A`. This is the key used everywhere else. |
| `aliases` | list of string | Other ids that point at this problem. 31 of the 229 train problems have at least one. Empty when there is none. |
| `contest_id` | string | Contest number, the part of `id` before the slash. |
| `contest_name` | string | Contest name as published. |
| `contest_type` | string | `CF` (370), `ICPC` (88), or `IOI` (1). |
| `contest_start` | int64 | Contest start, Unix seconds. |
| `contest_start_year` | int64 | Year of `contest_start`. In this subset the years run from 2010 to 2020. |
| `index` | string | Problem index inside the contest, such as `A` or `A1`. |
| `time_limit` | double | Time limit in seconds. Values in this subset are 0.5, 1, 2, and 3. |
| `memory_limit` | double | Memory limit in megabytes. Values in this subset are 64, 256, and 512. |
| `title` | string | Problem title. |
| `description` | string | Statement body. Flattened math was restored in this field. See below. |
| `input_format` | string | How the input is arranged. Same restoration as `description`. |
| `output_format` | string | What the program must print. Same restoration. |
| `note` | string | Statement note. Empty or null on 131 of 459 problems. Same restoration. |
| `interaction_format` | string | Interactive-problem instructions. Empty or null on every problem in this subset. |
| `examples` | list of struct | Public samples. Each element has `input` and `output`. Not restored. A sample line can be the integer `1018`. |
| `editorial` | string | Published editorial, when one was in the source. Empty or null on 286 of 459 problems. Not restored. Not part of the student prompt. |
| `rating` | int64 or null | Codeforces problem rating. Null on 5 problems. |
| `tags` | list of string | Codeforces tags, such as `number theory`. |
| `testset_size` | int64 | Number of official tests. Equal to the length of `official_tests`. Median 44 on train, 45 on dev, 43 on test. |
| `official_tests` | list of struct | Hidden grader tests. Each element has `input` and `output`. Line endings are `\r\n`. Not part of the student prompt. |
| `official_tests_complete` | bool | `true` on every row. The problem was kept only when the official tests were complete on disk. |
| `input_mode` | string | `stdio` on every row. The program reads standard input and writes standard output. |
| `generated_checker` | string or null | Custom checker source. Null on every row in this subset, so answers are compared as tokens. |
| `executable` | bool | `true` on every row. |
| `generated_tests` | int64 | `0` on every row. Grading uses `official_tests` only. |

`description`, `input_format`, `output_format`, `note`, and `interaction_format` are the statement fields. `restore_statement_math` rewrote flattened powers and subscripts in those fields when the reading was unambiguous. A power such as `10^{18}` that had been stored as `1018` is written back with `^`, `_`, and braces. Text that was already inside `$$...$$` was left as it was. Newer statements can still contain the site's `$$$...$$$` markup. Samples, official tests, and the editorial were copied as published.

Two further edits are in the stored statement and nowhere else. `697/B` has `10^{B}` and `10^{100}`. `114/A` has `k^{2}`, `k^{3}`, and `2^{31} - 1`. Both still match the old text once the math markup is removed.

## Submission columns

| Column | Type | Meaning |
| --- | --- | --- |
| `submission_id` | string | Codeforces submission id. Unique in these tables. |
| `problem_id` | string | Problem id. Matches `problems.id` in the same split. |
| `contestId` | string | Contest number. Same value as the problem's `contest_id`, with a different column name because it comes from the submission release. |
| `problem_index` | string | Problem index. Same value as the problem's `index`. |
| `programmingLanguage` | string | `Python 3` on every row. PyPy was not downloaded. |
| `source` | string | The submitted program. |
| `verdict` | string | Codeforces' verdict for this submission. Values are listed below. |
| `testset` | string | Which test set Codeforces used. `TESTS` (163,749), `PRETESTS` (3,712), `CHALLENGES` (685), `TESTS1` (421), or `TESTS2` (148). |
| `passedTestCount` | double | How many tests passed before the first failure, as counted by Codeforces. Stored as a whole number in a floating-point column. `0` means it failed the first test, or did not run. |
| `timeConsumedMillis` | double | Time Codeforces recorded, in milliseconds. Whole number in a floating-point column. From 0 to 3000 in this subset. |
| `memoryConsumedBytes` | double | Memory Codeforces recorded, in bytes. Whole number in a floating-point column. |
| `creationTimeSeconds` | double | When the submission was created, Unix seconds. Whole number in a floating-point column. |

Accepted means the verdict is `OK`. Every other verdict is a rejected submission. The rejected programs are the starters a repair task can use. An accepted program is a candidate solution. It passed on Codeforces' machines. It has not been re-run here as a golden solution.

| Verdict | Submissions | Meaning |
| --- | ---: | --- |
| `OK` | 88,589 | Passed the test set Codeforces used. |
| `WRONG_ANSWER` | 55,783 | Ran, and printed a wrong answer on some test. |
| `RUNTIME_ERROR` | 11,389 | Crashed. |
| `TIME_LIMIT_EXCEEDED` | 7,208 | Too slow on Codeforces' judge. |
| `COMPILATION_ERROR` | 4,830 | For Python, usually a syntax error. |
| `CHALLENGED` | 545 | Broken by another contestant's test. |
| `MEMORY_LIMIT_EXCEEDED` | 280 | Used too much memory. |
| `SKIPPED` | 87 | Not judged. |
| `CRASHED` | 2 | Judge-side failure, not a program verdict to imitate. |
| `FAILED` | 1 | Judge-side failure. |
| `IDLENESS_LIMIT_EXCEEDED` | 1 | The program stopped reading or writing. |

`SKIPPED`, `REJECTED`, `FAILED`, and `CRASHED` are not treated as faulty student programs when a starter is chosen. This subset has no `REJECTED` rows.

## What is not a column

The small-model scores are not stored on the problem or the submission. They are in `data/example_code/harbor_jobs/`, one directory per model and split group. A trial name such as `26_A__101817015__ab12` is problem `26/A`, starter submission `101817015`, and one attempt. The score is `1.0` or `0.0` under `stats.evals.*.reward_stats.reward`. The viewers join that file to these tables. They do not write it back.

These Harbor jobs are the ones that count:

| Split | Job prefixes | Small models |
| --- | --- | --- |
| train | `solved158`, `rest_train` | Llama 3.2 1B, Ministral 3B, Llama 3.2 3B |
| dev | `heldout_dev` | the same three |
| test | `heldout_test` | the same three |

Each small-model job is five attempts at temperature 0.7. The OpenRouter ids are `openrouter/meta-llama/llama-3.2-1b-instruct`, `openrouter/mistralai/ministral-3b-2512`, and `openrouter/meta-llama/llama-3.2-3b-instruct`. Qwen3-Coder is `qwen/qwen3-coder`. The stopped `cond01__llama-3.2-1b__k5__t07` job is not part of this record.
