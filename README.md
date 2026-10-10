# DS-Harness: an agent built around a small LLM (challenge 194, ML-Arena)

**Team:** _to be filled in_ · **Freeze:** 2026-11-03 23:59 · **Oral:** 2026-11-04

## 1. Challenge and grading

Each task gives an English objective, sometimes with CSV files and a data dictionary. The required answer is one
number. The private evaluation contains 120 tasks:

| Family | Content | Tasks | Share of private tasks |
|---|---|---:|---:|
| No files | Arithmetic, percentages, growth, dates, probability, word problems | 45 | 37.5% |
| Tables | Statistics, filters, group rankings, joins, derived columns | 50 | 41.7% |
| Fits | Linear or logistic regression on `train.csv` | 25 | 20.8% |

The score is the percentage of tasks answered within tolerance (`1.5 × 10⁻ⁿ` for n requested decimals). The project
grade is split between the leaderboard (25%) and the oral for both members (75%). The leaderboard mark is the best
chosen submission's private-set score at the freeze. On a linear conversion, a 90.8% score contributes 4.54 of the
leaderboard component's 5 points on the final /20 grade.

Runtime constraints: a model of about 1.5B parameters, 40 seconds per call of 8 tasks, 15 calls, 3 CPUs, 3 GiB RAM,
and no network. A call that raises or exceeds its time limit fails the deployment.

## 2. Results

| Evaluation | Score | No files | Tables | Fits | Notes |
|---|---:|---:|---:|---:|---|
| Task reference, Qwen2.5-1.5B (dev) | 46.1 | 86.2 | 28.8 | 11.4 | |
| Task reference, Qwen2.5-Coder-1.5B (dev) | 53.3 | 81.5 | 47.5 | 14.3 | |
| Task reference, Qwen3-1.7B (dev) | 62.2 | 80.0 | 62.5 | 28.6 | |
| Agent, local dev, initial version | 61.1 | 75.4 | 58.8 | 40.0 | Before fit rules |
| Agent, local dev, Qwen3-1.7B | 55.0 | 81.5 | 48.8 | 20.0 | |
| Agent, local dev, Coder + fit rules | 65.6 | 72.3 | 46.2 | 97.1 | |
| Agent, local dev, earlier current version | 69.4 | 72.3 | 55.0 | 97.1 | |
| Agent, platform, first version | 51.7 | 75.6 | 40.0 | 32.0 | Before fit rules |
| Agent, platform, earlier stage 3 | 74.2 | 77.8 | 60.0 | 96.0 | From the previous README |
| Agent, platform, penultimate submission | 89.2 | 91.1 | 84.0 | 96.0 | 107/120 tasks |
| **Agent, best platform result reported** | **90.8** | **95.6** | **84.0** | **96.0** | **109/120 tasks** |
| Agent, full local dev run | 97.2 | 92.3 | 100.0 | 100.0 | 180 tasks; 0 format errors |

The latest private-set category scores correspond to 43/45 no-file tasks, 42/50 table tasks, and 24/25 fit tasks. The
increase from 89.2% to 90.8% is two additional tasks, reflected in the no-files score moving from 91.1% to 95.6%;
tables and fits remained at 84% and 96%. These are aggregate results only; private objectives and files are not logged.

One task is about 0.83 percentage points of the total score. The remaining errors are concentrated in tables (8 of
50 tasks at 84%), followed by no-files (2 of 45) and fits (1 of 25). The local 97.2% result is useful evidence on the
public dev set, but it does not predict private-set performance by itself.

## 3. Recent changes and checks

- Deterministic no-file rules now cover summing explicitly listed amounts, periodic departures using an LCM,
  Bayesian positive-test questions, and whole-year compound-interest thresholds.
- Table rules cover whole-table statistics, common filters, group aggregates and counts, joins, and derived ratios.
  Recent group-query handling recognizes additional phrasings such as “per”, “by”, “within each”, and group counts.
- Six targeted synthetic group-aggregation variants passed.
- A deterministic replay of the public 180-task dev produced 129 direct rule answers, with no incorrect direct
  outputs; 51 tasks were delegated to the LLM. This replay is not a full end-to-end model run.
- The last full end-to-end local run reported 97.2% on dev. It preceded the latest group-phrase extension.

Synthetic cases are a separate test set; they do not change the public dev data or train the model. A synthetic pass
shows that a specific code path behaves as expected, not that the same phrasing or failure occurs in the private set.

## 4. Architecture

```text
task ──► route ──► read files ──► hints ──► generate ──► execute ──► vote ──► rounded number
         │         (cleaning)     (prompt)    (LLM)       (sandbox)
         └─ no-file and table rules; fit solver falls back to the LLM when uncertain
```

1. **Router:** no files → `nofile`; `train.csv` present → `fit`; otherwise → `table`.
2. **Robust reading:** separator detection, decimal commas, missing-value tokens (fixed list plus tokens quoted in the
   objective), glued units (`12 kg`), class codes (`5A`, `5B`) kept as text, day-first dates, and requested deduplication.
3. **Joins:** finds shared key columns and normalizes case, prefixes, and leading zeros; the detected join key is passed
   to the model as a hint.
4. **Named entities:** links quoted names in the objective (for example, “Grenoble lab”) to dictionary codes.
5. **Rules:** deterministic handlers solve supported no-file, table, and fit templates. Uncertain fit tasks fall back
   to generated code.
6. **Prompt and generation:** Qwen2.5-Coder-1.5B writes Python; the system gives it a short instruction, a worked
   example, table schema, dictionary, hints, and objective. It runs a greedy pass, then samples while time remains.
7. **Execution and recovery:** generated code runs with pandas, NumPy, SciPy, SymPy, and scikit-learn. `pd.read_csv`
   returns cleaned tables; `SmartFrame` recovers common column-name mismatches; failed code can be repaired and
   truncated code continued.
8. **Vote and safety:** the greedy result has weight 1.5, each sample weight 1. Errors are caught per task, and the
   internal time budget is 29 seconds out of 40.

## 5. Files and local usage

| File | Role |
|---|---|
| `agent.py` | Agent class and `solve` entry point |
| `dsh.py` | Challenge starter kit, submitted alongside `agent.py` |
| `outputs/stage3.py` | Current working candidate with the latest deterministic-rule changes; the latest small group-phrase extension has not yet had a full end-to-end run |
| `dev.json` | Public development tasks |
| `dev_ext.json` | 30 extra table tasks (standard deviation, correlation, second-largest value, and others) |
| `README.md` | This report |

Run from the repository containing `localtest.py`, `scoring.py`, and `dev.json`:

```bash
python localtest.py agent.py --test-run -v
python localtest.py agent.py --out run.json
python localtest.py agent.py --tasks dev_ext.json
python localtest.py agent.py --type table.join -v
```

Environment variables for local experiments only:

| Variable | Effect |
|---|---|
| `DSH_DEBUG=1` | Prints generated code, errors, and the schema seen by the model; never enable on the platform |
| `DSH_BUDGET=100` | Internal time budget (default 29 seconds) |
| `DSH_SAMPLES=8` | Samples per task (default 4) |
| `DSH_MODEL=Qwen/Qwen3-1.7B` | Selects another model from `ALLOWED_MODELS` |
| `DSH_NO_REPAIR=1`, `DSH_OLD_PROMPT=1` | Ablation switches |

In Colab, an environment variable set with `os.environ[...]` is inherited by subsequent shell commands.

## 6. Defects found and fixes

| Defect | Effect | Fix |
|---|---|---|
| Generated code re-imported pandas and replaced the patched `read_csv` | Table tasks failed | Patch `pd.read_csv` temporarily so the wrapper survives imports |
| Scikit-learn rejected NaNs | Fit tasks failed | NaN-tolerant linear and logistic regression |
| Join keys differed between files | `table.join` initially scored 0% | Normalize case, prefixes, and leading zeros |
| Logistic regression used regularization by default | Predictions missed tolerance | Use unregularized fitting when requested |
| Incorrect `dsh.load_llm` handling | Duplicate model load and attempted download | Use `llm.model` and `llm.tok` |
| Unbounded generation and signal-based timeout | Deadline failures | Bound `generate()` with `max_time`; preload imports and use a watched thread |
| Sampling skipped when generation was slow | No valid program for some tasks | Opportunistic sampling bounded by remaining time |
| Wrong columns, missing dictionary reads, and truncated code | Many generated programs failed | Serve the dictionary via `read_csv`, add `SmartFrame`, and continue truncated code |
| Several recurring no-file formula patterns | A few local calculation errors | Add deterministic formula handlers and check them with synthetic variants |

## 7. Limits, risks, and next steps

- **Tables are the largest current weakness:** 84% on the latest reported private result. Improve broad handling of
  statistics, filtered aggregates, grouping, and joins; keep unseen synthetic variants as a holdout.
- **No-file rules are pattern-based:** the private score rose from 91.1% to 95.6%, but there are still two missed
  tasks and no private task-level diagnostics to identify them.
- **Fits are near saturation:** 96% privately and 100% in the last full local dev run. The rules rely on regular
  objective templates; differently worded fits remain a risk.
- **Synthetic coverage is not a private score estimate:** vary both data and language, and keep separate examples
  that are not used when writing rules.
- **Runtime and packaging:** quality depends on time available for repair and voting. Check the 3 GiB memory limit,
  the upload scanner, and the exact entry-point filenames before deployment.
- **Noise:** one private task changes the total by about 0.83 points. Compare versions on the same fixed evaluation
  and avoid interpreting one small score movement as proof of a general improvement.

## 8. Deployment

Before submission, create the team, unset all `DSH_*` variables, and check memory:

```bash
/usr/bin/time -v python localtest.py agent.py --test-run
```

The reported maximum resident set size must stay below 3,000,000 KB. Submit both `agent.py` and `dsh.py` using the
challenge's required runtime. The quota is five deployments per person per rolling 24 hours, including failed ones.
The best chosen submission at the freeze determines the leaderboard mark.

## 9. Privacy

The challenge rules forbid logging objectives or files from platform runs. In production, print only durations and
model-loading status; do not save private task content or attempt to recover hidden task-level errors.
