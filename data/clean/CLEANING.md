# Cleaning the Codeforces statements

This note is the methods record for `data/clean/`. The first section is written so it can be shortened into a working paper. The later sections give the rationale and the exact rule that produced the tables.

## Summary

We study whether small language models can repair a rejected Python 3 program when the problem statement is one a strong coder can solve. A wrong statement, or a stored test that no accepted program can pass, would look like a model failure. The clean subset is the set of problems for which that confound is unlikely.

The source is the open-r1 release of Codeforces. Its statements often lost superscripts and subscripts when formulas were read from page images, so a bound such as \(10^{18}\) is stored as `1018` and an index such as \(x_{n-1}\) as `xn - 1`. We repaired only the patterns that have one sensible reading, and we left the raw download unchanged. Samples, official tests, editorials, and submissions were not rewritten.

We then asked coding models to solve the public statement, with no accepted program and no hidden test in the prompt, and graded the program on the stored official tests. A statement was edited further only when the edit inserted math markup and left the words in place, and only when Qwen3-Coder (`qwen/qwen3-coder`) then passed the tests. Two statements were changed by hand under that same rule: `697/B` and `114/A`. Proposals that deleted words, spaces, or braces were discarded. One problem, `44/H`, has no accepted program that passes its stored tests, so it was not graded and was not edited.

`data/clean/` keeps a problem only when Qwen3-Coder itself passed. A pass by the smaller coders, Qwen3-Coder 30B-A3B or Qwen2.5-Coder 32B, is not enough. On 5 November 2026 that rule kept 459 of the 901 problems in the official subset: 229 train, 123 dev, and 107 test, together with the 168,715 Python 3 submissions to those problems. The other problems stay in the official subset. They are omitted here because we do not have this evidence that the published statement is solvable.

## What the experiment needs from a statement

The student sees the statement, the samples, one rejected submission, and the teacher line "try again". It must return a Python 3 program. The program is graded on the official tests after the model stops. The student does not see those tests or an accepted solution.

Under that protocol, three different failures look the same:

- the model cannot repair the program;
- the statement no longer states the problem, because a formula was flattened;
- the stored tests do not match any accepted program, so nothing can pass.

The cleaning is aimed at the second and third. It is not a claim that the remaining statements are easy, and it is not a filter on the small models' scores.

## Where the text came from

Codeforces problems and Python 3 submissions were taken from `open-r1/codeforces` and `open-r1/codeforces-submissions`. The problem text is pinned to revision `fbe3f6e903ee854eec2e69e9d96d0306cde59baf`. In that release, older formulas had been page images. A vision model read the images into plain text and often glued a superscript to the base or pulled an operator out of a subscript. Later problems still contain the site's `$$$...$$$` MathJax, because that markup was in the HTML. We do not translate general LaTeX. We only restore flattenings we can recognize.

The raw files stay as published. Every repair is applied to the working copy, `data/interim/codeforces_official_subset/`.

## The official subset, before statement repair

Of 6,038 problems with a Python 3 submission, 901 were kept on 14 October 2026. A problem had to have complete official tests on disk, at least one accepted Python 3 submission, at least one rejected Python 3 submission, no official test whose expected output exceeds 1 MB, and a non-empty statement. Empty statements in this collection are April Fools puzzles whose hidden rule is the task. Tests with huge outputs mostly measure printing speed.

Those 901 problems were split by contest, stratified by rating, with seed `20261014`: 538 train, 185 dev, and 178 test. A problem's submissions stay in that problem's split. Not every split member is usable for repair. A usable problem has the fields the grader needs and at least one rejected submission that can serve as a starter. The usable counts are 421 train, 150 dev, and 143 test.

This subset is the population. The clean tables are a subset of it.

## Restoring flattened math

`restore_statement_math` in `src/tagent/data/statement_math.py` rewrites five fields: the description, the input format, the output format, the note, and the interaction format. On 4 November 2026 it changed 470 fields across 382 of the 901 problems. Running it again changes nothing.

The function is deliberate about what it will not guess.

A power of ten is rewritten only for `10` glued to an exponent in {18, 15, 12, 10, 9, 8, 7, 6, 5, 4, 3, 2}, and only when the surrounding words force that reading: a bound, a product, a modulus, a parenthesis whose integer equals the power, or a magnitude word such as "at most". `≤ 1018` becomes `≤ 10^{18}`. A comma-separated list of ordinary integers, a binary code, and a count such as `108 days` stay as written. In particular, `75/A` keeps `105 + 106 = 211`, `611/B` keeps the binary equality `510 = 1012`, and `213/B` keeps `103` inside a sample list.

A subscript is rewritten only when a plus or a minus was flattened into the index and the sentence has one reading. `(xn - 1 - xn)` becomes `(x_{n-1} - x_{n})`. A comparison of a name with itself, such as `ai ≥ ai + 1`, becomes `a_{i} ≥ a_{i+1}`, because a value is never constrained to be at least itself plus one. A bare index such as `a1` stays `a1`. The token `n2` can be \(n_2\) or \(n^2\); `177/F1` and `255/D` were left as published.

Text already inside `$$...$$` is set aside and copied back. Sample inputs and sample outputs are not passed through the restorer. A sample line can be the integer 1018, and changing it would change the example rather than the formula. Official tests and editorials are copied as published for the same reason.

The same function still runs when a statement is rendered for a student, so a fresh copy of the published text is repaired on the way to the prompt. The stored table is already repaired, and the function does not change repaired text.

## Checking that a statement can be solved

Before a problem is treated as a fair repair task, a coder has to solve it from the public statement alone. The prompt contains the statement fields and the samples. It does not contain an accepted program, a failed starter, or a hidden test. Grading matches the student grader: the program is run locally on each official test, with a 30-second limit per test, and outputs are compared without regard to whitespace. A pass means every stored test passed.

On a fixed list of 200 train problems we first tried smaller coders. Qwen3-Coder 30B-A3B Instruct tried once. If it failed, Qwen2.5-Coder 32B Instruct tried the same text. The planned 7B coder is no longer on the OpenRouter catalog, so it was not used. Either pass kept the stored statement. That pass is evidence a model of that size can solve the text. It is not the evidence used for `data/clean/`. One hundred and four of the 200 were solved this way. They remain in the official subset and are absent from the clean tables.

Where both smaller coders failed, we did not treat the statement as broken. The shortest accepted program often fails the stored tests, while a longer accepted program passes. Using the shortest program as an oracle would have marked good tests as bad. For each unsolved problem we searched accepted programs, shortest first, until one passed. That program is the anchor. It is shown only to a researcher-side editor, never to the solver and never to the student. Eighty-eight of the eighty-nine remaining problems had an anchor. `44/H` did not: all six of its accepted programs raise `ValueError` on the first stored test, and there is no custom checker. Its statement was not edited, it was not sent to the solver, and it is not in the clean tables.

## Why statement edits were refused unless the large coder passed

An early automatic cleaner rewrote prose. It deleted words and, in some cases, existing math such as `p_{i}`. Those edits were restored from the audit, including `13/A`, `53/C`, `57/A`, `79/A`, `90/A`, and `192/A`. Six further train statements that had been rewritten (`221/B`, `257/B`, `402/C`, `560/B`, `723/B`, `813/B`) were restored as well. A cleaner that can change the words can change the problem. A later pass would then "solve" a different problem from the one Codeforces asked.

The rule after that is narrow. A proposed edit is kept only when both of these hold.

1. The new text is the old text plus math markup. Deleting `^`, `_`, `{`, `}`, `$`, and backslash from the new text must reproduce the old text exactly. Deleting a space, a letter, or a brace that was already there fails the check. Turning `10^{6}` into `10^6` fails, because a character other than markup would have to disappear. A field may not be emptied or invented.
2. Qwen3-Coder then solves the repaired statement from the public text alone. The anchor is not in that prompt. If the solver fails, the edit is not written, even when it looks like a formula.

Under that rule the model-written repairs saved one statement, `697/B` (`10^{B}` and `10^{100}`, with the surrounding spaces kept). The model's own glued version was not the version stored. One further edit was applied by hand, not by the model, and then checked the same way: in `114/A`, `k2` and `k3` became `k^{2}` and `k^{3}`, and the input bound `231 - 1` became `2^{31} - 1`. The anchor still passes. The 32B draft of `114/A` was not stored, because it also deleted blank lines.

Later repair passes on the problems the large coder missed proposed edits that deleted words, shortened a field, or changed nothing. None of those edits were saved. On the rest of the usable train problems, and on dev and test, the same repair stage saved nothing.

## What Qwen3-Coder solved

The clean tables use one definition. A problem is in if and only if a program written by `qwen/qwen3-coder` passed every stored official test. The audit row must not be an error. The calls that contribute are these.

| Setting | Problems the large coder passed |
| --- | ---: |
| Stored statement, temperature 0, one try, on 88 anchored train problems the smaller coders missed | 41 |
| Second try at temperature 0, after rejecting non-formula edits | 8 |
| Further try at temperature 0.4 | 4 |
| `697/B`, after the markup repair | 1 |
| Remaining usable train problems, up to 3 tries at temperature 0.7 | 175 |
| Usable dev problems, up to 3 tries at temperature 0.7 | 123 |
| Usable test problems, up to 3 tries at temperature 0.7 | 107 |

The first four rows are 54 train problems and do not overlap the 175. The review file's 53 passes are the first three rows; `697/B` is recorded separately. The total is 229 train, 123 dev, and 107 test.

Temperature 0.7 with up to three tries is the screen used once we moved past the first 200 train problems. The calls stop at the first pass. Three tries, rather than one, avoid dropping a problem because a strong model had a single unlucky sample. They are not a search over prompts, and the statement is not changed between them.

The 104 problems solved only by the 30B or 32B coder were not sent to Qwen3-Coder again. They may be fine statements. We do not claim they are unsolvable. They are absent because this file records a pass by the large coder, and that pass was not collected for them. Anyone who wants the broader set, "solved by any of the coding models we tried," has to add those 104 back from `data/example_code/instruction_audit.jsonl`. That broader train count would be 333, not 229.

## What was left out, and why

**No large-coder pass.** This is the main cut. It includes the 104 smaller-coder passes, the train problems Qwen3-Coder tried and missed, the 27 dev misses, and the 20 test misses.

**Ungradeable tests.** `44/H` has no anchor. A failure there cannot be blamed on the model or repaired as a formula. It is out.

**API errors are not passes.** Sixteen test problems returned an OpenRouter rate limit during the solve stage (`27/E`, `18/A`, `40/C`, `45/A`, `40/A`, `40/B`, `3/C`, `18/B`, `48/A`, `82/A`, `73/A`, `93/A`, `94/A`, `94/B`, `979/A`, `711/E`). The cleanup calls for the other 20 test misses failed with a rate limit or a gateway timeout. Dev had no such failures. An error is not evidence the statement is bad, and it is not evidence the model can solve it. Those 36 test problems are out. `3/C` is one of them. It was also seen before the split was frozen, as was `690/A1`, which is not in the clean test table.

**Harbor exclusions are not clean-set exclusions.** The student runs skipped a few problems the large coder did solve. `195/B` would have shown a hidden expected output in the student instruction. `177/A1`, `236/B`, `544/A`, and `1298/C` already had a starter that passed the tests, so there was nothing to repair. Those problems are in `data/clean/` when the large coder passed. The clean tables are a statement set, not a copy of the Harbor task list.

## What the files contain

Column definitions are in [DATA_DICTIONARY.md](DATA_DICTIONARY.md). Commands and the loader are in [USAGE.md](USAGE.md).

| File | Rows |
| --- | ---: |
| `problems/train.parquet` | 229 |
| `problems/dev.parquet` | 123 |
| `problems/test.parquet` | 107 |
| `submissions/train.parquet` | 75,049 |
| `submissions/dev.parquet` | 52,726 |
| `submissions/test.parquet` | 40,940 |

Problem rows have the same columns as the official subset, including the repaired statement fields, the samples, the official tests, and the editorial when one exists. They are sorted by contest and index, with four problems per row group. Submission rows are every Python 3 submission to those problems, accepted or not, sorted by problem and submission id. `manifest.json` records the model, the rule, and the problem ids.

`tagent.data.clean_codeforces` reads the tables in the same way `tagent.data.codeforces` reads the official subset: `load_problem`, `load_submissions`, `load_submission`, `split_of`, and `problem_ids`.

## Limits of the claim

The clean subset supports a narrower claim than "these statements are correct Codeforces statements."

The math restorer misses any formula that was never written into the published text, and it leaves ambiguous tokens alone. Many newer statements still contain `$$$...$$$` markup. A reader has to treat that as part of the text the models saw.

The solvability screen shows that some program written by one model passed the stored tests. It does not show that every accepted program passes, and it does not show that a different reading of a flattened formula would be wrong. Where we refused an edit, the stored statement is the one the large coder solved, flattened math included.

The test split is missing 16 problems that were not finished, plus 20 that were missed and whose repair call did not finish. Dev is missing only the 27 the model failed. Train is missing the 104 smaller-coder solves that were never rechecked with the large model. Rates computed on `data/clean/` should not be described as rates on all usable Codeforces problems, or on all problems any coder in this project solved.

The official tests in these tables are the grader's tests. They are not part of the student prompt. Putting them in the released problem row is what makes the subset gradable. A training pipeline that shows the row to a model has to hold the tests, the editorial, and the accepted submissions out of the prompt, as the student runs did.
