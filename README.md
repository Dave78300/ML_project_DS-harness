# DS-Harness: a rule-first agent with an LLM fallback (challenge 194, ML-Arena)

**Team:** _to be filled in_ · **Freeze:** 2026-11-03 23:59 · **Oral:** 2026-11-04

## TL;DR

- **What it is:** an agent around Qwen2.5-Coder-1.5B. The system reads and cleans the files, solves the task families
  it recognises with deterministic rules, and sends everything else to the LLM, which writes Python that the system
  runs, repairs and votes on.
- **Best private leaderboard result (reported):** **92.5 %** (111/120): no files 95.6 %, tables 88.0 %, fits 96.0 %.
- **Progress on the private set:** 51.7 → 74.2 → 89.2 → 90.8 → 92.5 (versions v1 to v5, section 2).
- **How to read the numbers:** the public dev set is only a diagnostic. The deterministic rules were written from the
  dev objectives, so a perfect dev score is expected by construction. The private score is the measure that counts.
  Nine private tasks are still missed (6 tables, 2 no-files, 1 fit); task-level private errors are not available.

## 1. Challenge and grading

Each task gives an English objective, sometimes with CSV files and a data dictionary. The required answer is one
number. The private evaluation contains 120 tasks:

| Family | Content | Tasks | Share |
|---|---|---:|---:|
| No files | Arithmetic, percentages, growth, dates, probability, word problems | 45 | 37.5 % |
| Tables | Statistics, filters, group rankings, joins, derived columns | 50 | 41.7 % |
| Fits | Linear or logistic regression on `train.csv` | 25 | 20.8 % |

The score is the percentage of tasks answered within tolerance (`1.5 × 10⁻ⁿ` for n requested decimals). According to the
course module, the project grade is split between the leaderboard (25 %) and the oral for both members (75 %). The
leaderboard mark is the private-set score of the submission chosen by the team at the freeze. If the conversion is
linear, 92.5 % contributes 4.625 of the leaderboard component's 5 points on the final /20 grade (to be confirmed with
the course); one task is worth about 0.04 points of the final grade.

Runtime constraints: a model of about 1.5B parameters, 40 seconds per call of 8 tasks, 15 calls, 3 CPUs, 3 GiB of RAM,
no network. A call that raises or exceeds its time limit fails the deployment.

## 2. Results

Version labels: **v1** first deployment, **v2** deployment with the fit rules and execution fixes (called "stage 3" in
earlier notes), **v3** to **v5** later deployments with more deterministic rules. The file `stage3.py` is the current
candidate; its name is unrelated to the earlier "stage 3" label.

### 2.1 Private leaderboard (reported by the team; aggregate scores only)

| Version | Score | No files | Tables | Fits | Tasks |
|---|---:|---:|---:|---:|---:|
| v1, initial agent | 51.7 | 75.6 | 40.0 | 32.0 | 62/120 |
| v2, fit rules and execution fixes | 74.2 | 77.8 | 60.0 | 96.0 | 89/120 |
| v3 | 89.2 | 91.1 | 84.0 | 96.0 | 107/120 |
| v4 | 90.8 | 95.6 | 84.0 | 96.0 | 109/120 |
| **v5, best reported** | **92.5** | **95.6** | **88.0** | **96.0** | **111/120** |

v5 corresponds to 43/45 no-file, 44/50 table and 24/25 fit tasks. One task is about 0.83 points of the total score, so
the step from v4 to v5 is two tasks, both in the tables family (42 → 44); the step from v3 to v4 was two tasks in the
no-files family.

For reference, the challenge's own reference systems score 46.1 (Qwen2.5-1.5B), 53.3 (Qwen2.5-Coder-1.5B) and 62.2
(Qwen3-1.7B) on the public dev set (no files / tables / fits: 86.2/28.8/11.4, 81.5/47.5/14.3, 80.0/62.5/28.6). Those
are dev scores; their private scores are not known.

### 2.2 Local runs on the public dev set (diagnostic only)

| Run | Score | No files | Tables | Fits |
|---|---:|---:|---:|---:|
| Initial agent, Coder | 61.1 | 75.4 | 58.8 | 40.0 |
| Initial agent, Qwen3-1.7B | 55.0 | 81.5 | 48.8 | 20.0 |
| Coder + fit rules | 65.6 | 72.3 | 46.2 | 97.1 |
| Coder + fit rules + execution fixes | 69.4 | 72.3 | 55.0 | 97.1 |
| Full local run, later candidate | 97.2 | 92.3 | 100.0 | 100.0 |
| Latest local result (candidate behind v4) | 100.0 | 100.0 | 100.0 | 100.0 |

The 97.2 % run covers 180 tasks with 0 format errors. For the 100.0 % result the task count was not stated and the run
file is not attached to this report. A 100 % dev score says that the system reproduces dev, not how it behaves on
other task draws.

A replay of the rules alone, without the model, on the 180 dev tasks gave 129 direct rule answers, all correct; the
other 51 tasks go to the LLM. This replay is not an end-to-end run.

## 3. Architecture

```text
task ──► route ──► read files ──► rules ──► answer
         │         (cleaning)       │
         │                          └─ no rule applies ─► hints ─► generate ─► execute ─► repair/vote ─► answer
         └─ round the answer as the objective asks
```

1. **Router:** no files → `nofile`; `train.csv` present → `fit`; otherwise → `table`.
2. **Robust reading:** separator detection, decimal commas, missing-value tokens (fixed list plus tokens quoted in the
   objective), glued units (`12 kg`), class codes (`5A`, `5B`) kept as text, day-first dates, requested deduplication.
3. **Joins:** finds shared key columns and normalises case, prefixes and leading zeros.
4. **Rules (deterministic, tried first for every family):** supported no-file, table and fit templates are solved by
   code; a rule that is not sure returns nothing and the task goes to the LLM.
5. **LLM fallback:** Qwen2.5-Coder-1.5B, loaded through `dsh.load_llm`, writes Python from a short instruction, one
   worked example, the table schema, the dictionary and hints (join keys, named entities, formulas).
6. **Execution and recovery:** generated code runs in a watched thread with pandas, NumPy, SciPy, SymPy and
   scikit-learn. `pd.read_csv` returns the cleaned tables (and the data dictionary), `SmartFrame` recovers common
   column-name mismatches, failed code is repaired using its error message, truncated code is continued.
7. **Vote and safety:** greedy result weight 1.5, each sample weight 1; errors are caught per task; internal time
   budget of 29 s out of 40 s, and every `generate` call is bounded by `max_time`.

### What the rules cover

- **Fits:** target = the `train.csv` column missing from `test.csv`; features matched to the dictionary by word overlap;
  year filter; unregularised linear or logistic regression; prediction, intercept, R², coefficient, count and mean
  questions. 34/35 correct on dev with the first version of the rules and 24/25 on the private set.
- **Tables:** whole-table statistics, common filters, group aggregates and counts, joins, derived ratios.
- **No files:** summing listed amounts, periodic departures (LCM), ordered arrangements, linear systems, Bayesian
  positive-test questions, compound-interest thresholds, and similar recurring templates.

The rules exploit the regularity of the task generator. They are validated on dev, where they were written, and their
private-set behaviour is only known through the aggregate scores in section 2.1.

## 4. Defects found and fixes

| Defect | Effect | Fix |
|---|---|---|
| Generated code re-imported pandas and replaced the patched `read_csv` | Table tasks failed | Patch `pd.read_csv` temporarily so it survives imports |
| scikit-learn rejects NaN | Fit tasks failed | NaN-tolerant linear and logistic regression |
| Join keys written differently in the two files | `table.join` at 0 % on dev | Key detection and normalisation |
| Logistic regression regularised by default | Probabilities outside tolerance | Unregularised fit when the objective asks for it |
| `dsh.load_llm` returns an `LLM` object, not a tuple | Double load, attempted download | Use `llm.model` and `llm.tok` |
| Unbounded generation, signal-based timeout | `deadline exceeded` | `max_time`, preloaded imports, watched thread |
| Sampling skipped when generation was slow | No valid program for some tasks | Opportunistic sampling bounded by remaining time |
| Wrong column names, dictionary read through `read_csv`, code cut by the token limit | Many generated programs failed | `SmartFrame`, dictionary served by `read_csv`, continuation of truncated code |
| Recurring formula and template errors in no-file and table tasks | Wrong answers | Deterministic rules |

Known issues in the current candidate (they only affect tasks sent to the LLM):
- `LAST_ERR` is not reset between candidates, so a repair prompt can quote another task's error.
- `.round()` on a Python float raises; the prompt only asks the model not to round.

A patched copy (`stage3_plus.py`) fixes both and leaves the rule outputs unchanged (129/129 on the replay); its effect on
the LLM path has not been measured.

## 5. Limits, risks and next steps

- **Tables are the largest remaining loss:** 88 % privately (6 tasks missed). Keep unseen variants (for example
  `dev_ext.json`) as a holdout and compare versions task by task, not by family totals.
- **No-file rules are pattern-based:** two private tasks are missed and the private errors cannot be inspected.
- **Fits are near saturation** (96 % privately); differently worded objectives would fall back to the LLM.
- **Dev is exhausted as a test set:** it was used to write the rules. Evaluate changes on data and phrasings that were not
  used to write them.
- **Runtime and packaging:** quality depends on the time left for repair and voting. Check the 3 GiB memory limit, the
  upload scan and the exact entry-point filenames before deploying.
- **Noise:** one private task is 0.83 points; do not read a one- or two-task change as a general improvement.

## 6. Files and local usage

| File | Role |
|---|---|
| `stage3.py` | Current candidate (`Agent` entry point); submit it as `agent.py` |
| `dsh.py` | Challenge kit imported by the candidate; submit it with `agent.py` |
| `localtest.py`, `scoring.py` | Local evaluator and reference scoring code (keep `scoring.py` unchanged) |
| `env.py`, `schema.md` | Platform environment and agent input/output contract |
| `requirements.txt`, `hf_models.json`, `test_dsh.py`, `UPSTREAM.md` | Kit dependencies, mounted model IDs, kit tests, upstream revision |
| `dev.json` | Public dev set; download locally, excluded from Git |
| `dev_ext.json`, `compare_runs.py` | Optional: table tasks of unseen types; task-by-task comparison of two runs |
| `README.md` | This report |

```bash
python -m pip install -r requirements.txt
python -m pytest -q test_dsh.py
python localtest.py stage3.py --test-run -v          # the platform's 16-task test run
python localtest.py stage3.py --out run.json         # all 180 dev tasks at the real 40 s limit
python localtest.py stage3.py --type table.join -v   # a single task type
python compare_runs.py run_a.json run_b.json         # which tasks flipped between two runs
```

Environment variables, for local experiments only:

| Variable | Effect |
|---|---|
| `DSH_DEBUG=1` | Prints generated code, errors and the schema seen by the model; never enable on the platform |
| `DSH_BUDGET=100` | Internal time budget (default 29 s) |
| `DSH_SAMPLES=8` | Samples per task (default 4) |
| `DSH_MODEL=Qwen/Qwen3-1.7B` | Another model from `ALLOWED_MODELS` |
| `DSH_NO_REPAIR=1`, `DSH_OLD_PROMPT=1` | Ablation switches |

In Colab, a variable set with `os.environ[...]` is inherited by every later `!python` command.

## 7. Deployment

Before submitting: create the team, unset every `DSH_*` variable, and check memory.

```bash
/usr/bin/time -v python localtest.py stage3.py --test-run     # "Maximum resident set size" must stay below 3,000,000 KB
```

Submit `agent.py` (a copy of `stage3.py`) and `dsh.py` with the PyTorch runtime. Quota: five deployments per person per
rolling 24 hours, failed ones included. Keep an exact copy of every submitted file, so that the version behind each
score can be reproduced and the chosen submission can be explained at the oral.

## 8. Privacy

The challenge rules forbid logging objectives or files from platform runs. In production the agent prints only
durations and the loaded model; debug output exists only behind `DSH_DEBUG`.
