# DS-Harness: an agent built around a small LLM (challenge 194, ML-Arena)

**Team:** _to be filled in_ · **Freeze:** 2026-11-03 23:59 · **Oral:** 2026-11-04

## 1. The problem

Each task is an English objective, sometimes with CSV files (and a `data_dictionary.csv`), whose answer is **a single
number**. Three families (120 private tasks):

| Family | Content | Tasks |
|---|---|---|
| No files | arithmetic, percentages, growth, dates, probability, word problems | 45 |
| Tables | a statistic, a filter, a group ranking, a join, a derived column (1 or 2 CSV files) | 50 |
| Fits | linear or logistic regression on `train.csv` (prediction, coefficient, R², count) | 25 |

Constraints: a ~1.5B-parameter model (5 models mounted), **40 s per call of 8 tasks**, 15 calls, 3 CPUs, 3 GiB of RAM,
no network. A call that raises or runs out of time fails the whole deployment. Score = % of tasks within tolerance
(`1.5 × 10⁻ⁿ` for n decimals).

**Guiding idea:** the model computes nothing itself. It writes Python code, and the system reads the files, cleans the
data, runs the code, checks the result and votes. The system matters more than the model.

## 2. Results (measured)

| Evaluation | Score | No files | Tables | Fits |
|---|---|---|---|---|
| Task reference, Qwen2.5-1.5B (dev) | 46.1 | 86.2 | 28.8 | 11.4 |
| Task reference, Qwen2.5-Coder-1.5B (dev) | 53.3 | 81.5 | 47.5 | 14.3 |
| Task reference, Qwen3-1.7B (dev) | 62.2 | 80.0 | 62.5 | 28.6 |
| **Our agent, local, dev, Coder** (before the fit rules) | 61.1 | 75.4 | 58.8 | 40.0 |
| Our agent, local, dev, Qwen3-1.7B | 55.0 | 81.5 | 48.8 | 20.0 |
| **Our agent, local, dev, Coder + fit rules** | 65.6 | 72.3 | 46.2 | 97.1 |
| **Our agent, platform (private set), version before the fit rules** | **51.7** | 75.6 | 40.0 | 32.0 |

Reading the table, and caveats:
- The local dev score does not transfer: **tables lose about 19 points** between dev (58.8) and the private set (40.0),
  which matches the statement that the private set has table types dev does not. The no-files family transfers without
  loss.
- The last local run (65.6) had slow generation (Colab T4), so the voting round was skipped: its table score is an
  underestimate. The changes made afterwards (section 6) are **not yet measured**.
- The fit rules were written and validated on the dev objectives (see limits).
- Noise: about ±4.5 points on 120 tasks, about ±6 points per 16-task type (one task = 6 points).

## 3. Architecture

```
task ──► route ──► read files ──► hints ──► generate ──► execute ──► vote ──► rounded number
         │         (cleaning)     (prompt)    (LLM)       (sandbox)
         └─ fits: rule-based solver ──► number   (falls back to the LLM when it is not sure)
```

1. **Router** (`route`): no file → `nofile`; `train.csv` present → `fit`; otherwise → `table`.
2. **Robust reading** (`read_table`, `load_task`): separator detection, decimal commas, missing-value tokens (a fixed
   list plus tokens quoted in the objective), glued units (`12 kg`), class codes (`5A`, `5B`) kept as text, day-first
   dates (keyword in the objective, otherwise values above 12 decide), duplicates dropped when the objective mentions them.
3. **Joins** (`align_keys`): finds the key column shared by two tables (value inclusion, up to case, prefix and leading
   zeros) and normalises it; the model is told "Join key: …".
4. **Named entities** (`entity_hints`): a name quoted in the objective ("Grenoble lab") is linked to its dictionary code
   (`S02`).
5. **Fits** (`fit_rules`): target = the `train.csv` column missing from `test.csv`; features chosen by word overlap with
   the dictionary; year filter; **unregularised** linear or logistic regression (as the objective asks); the question
   is read with regular expressions. **When in doubt the solver returns nothing** and the LLM takes over.
6. **Prompt**: short system instruction, one worked example per family (invented, not copied from dev), table schema,
   dictionary, hints, objective.
7. **Generation** (`Agent.generate`): Qwen2.5-Coder-1.5B through `dsh.load_llm` (weights from the platform cache, no
   download). One greedy pass, then a sampling round (temperature 0.5, `DSH_SAMPLES` per task) as long as time remains.
   Every `generate` call is bounded by `max_time`.
8. **Execution** (`run_code`): the program runs in a watched thread (1.5 s limit, stops infinite loops). Namespace:
   pandas, numpy, scipy.stats, sympy, sklearn. In addition:
   - `pd.read_csv('name.csv')` returns the **already cleaned** table (including `data_dictionary.csv`);
   - `SmartFrame` recovers a wrongly named column (case, dictionary description, near-equality);
   - `LinearRegression` / `LogisticRegression` tolerate NaN; the logistic one is unregularised when the objective says so.
9. **Repair and continuation**: a program that crashes is sent back to the model with its error and the real columns;
   a program cut off by the token limit is continued (`run_continue`).
10. **Vote** (`vote`): majority value (the greedy pass weighs 1.5, each sample 1), rounded as the objective asks.
11. **Safety net**: every error is caught per task, `solve` always returns a number, internal budget of 29 s out of 40 s.

## 4. Files

| File | Role |
|---|---|
| `agent.py` | the agent (`Agent` class, `solve`); imports `dsh` |
| `dsh.py` | the challenge starter kit (model loading, tools), to be uploaded **together with** `agent.py` |
| `dev_ext.json` | 30 table tasks of a kind absent from dev (standard deviation, correlation, second-largest value…), gold values computed with pandas |
| `README.md` | this document |

## 5. Usage

### Local tests (in the `ds-harness` repository, next to `localtest.py`, `scoring.py`, `dev.json`)

```bash
python localtest.py agent.py --test-run -v          # the 16 tasks of the platform's test run
python localtest.py agent.py --out run.json         # the 180 dev tasks, at the real 40 s limit
python localtest.py agent.py --tasks dev_ext.json   # new table types
python localtest.py agent.py --type table.join -v   # a single task type
```

### Environment variables (local only, absent on the platform)

| Variable | Effect |
|---|---|
| `DSH_DEBUG=1` | prints generated code, errors and the schema the model sees (**never on the platform**: task content) |
| `DSH_BUDGET=100` | internal time budget (default 29 s), for slow GPUs |
| `DSH_SAMPLES=8` | samples per task (default 4) |
| `DSH_MODEL=Qwen/Qwen3-1.7B` | another model from `ALLOWED_MODELS` (default: the Coder) |
| `DSH_NO_REPAIR=1`, `DSH_OLD_PROMPT=1` | ablation switches |

In Colab, a variable set with `os.environ[...]` is inherited by every later `!python` command.

### Deployment

```python
import mlarena
client = mlarena.connect(api_key="mlk_user_...")      # never commit the key
client.submit(challenge_id=194, files=["agent.py", "dsh.py"],
              runtime={"language": "python", "framework": "torch"},
              wait=True, timeout_sec=1800)
```

Before submitting: create the team, set no `DSH_*` variable, check memory
(`/usr/bin/time -v python localtest.py agent.py --test-run` → "Maximum resident set size" below 3,000,000 KB).
Quota: 5 deployments per person per rolling 24 h, failed ones included.

## 6. What we learned (defects found, in order)

| Defect | Effect | Fix |
|---|---|---|
| Generated code re-ran `import pandas as pd`, overwriting my version of `read_csv` | tables near 0 % | `pd.read_csv` patched temporarily (survives imports) |
| sklearn raises on NaN | fits failing | NaN-tolerant regressions |
| Join keys written differently in the two files | `table.join` at **0 %** | `align_keys` → **87.5 %** |
| `LogisticRegression()` regularised by default (C=1) while the objective says "without regularisation" | probabilities outside tolerance | unregularised, solver forced |
| Wrong use of `dsh.load_llm` (it returns an `LLM` object, not a tuple) | double loading, attempted download | use `llm.model` / `llm.tok` |
| Unbounded `generate()`, heavy imports on the first call, signal-based timeout | `deadline exceeded` | `max_time`, preloaded imports, watched thread |
| Sampling round skipped when generation is slow | zeros (no valid program) | opportunistic round bounded by `max_time` |
| Recurring generated-code errors: `read_csv('data_dictionary.csv')`, wrong column names, program cut at 300 tokens | ~33 tasks at `0.0` out of 62 failures | dictionary served by `read_csv`, `SmartFrame`, `run_continue` (**not measured**) |

## 7. Limits and risks (to state at the oral)

- **Table generalisation**: 40 % on the private set. New table types remain the weak point; `dev_ext.json` approximates
  them but is not a measurement.
- **The fit rules exploit the regularity of the objectives** (five scenarios, fixed phrasings). 34/35 on dev, but that
  is a validation on the very objectives used to write them. If the private set words things differently, they fall
  back to the LLM (no loss) or, more rarely, answer wrongly with confidence (not observed on dev).
- **Time dependence**: quality depends on available time (voting, repair). Local measurements were made on a T4; the
  platform uses an RTX 4090 (faster).
- **Not verified here**: memory (3 GiB), upload scan rules (`exec` and `ctypes` are present in `agent.py`).
- **Overfitting**: several fixes come from failures observed on dev. Keep `dev_ext.json` and an unseen half of dev as
  guardrails.
- **Noise**: one task weighs 0.8 points out of 120; do not read a change of 1 to 3 tasks as an effect.

## 8. Next steps

1. Measure the effect of the latest fixes (`SmartFrame`, dictionary, continuation, opportunistic voting) on dev and `dev_ext.json`.
2. Adaptive voting: more samples for tasks where answers disagree.
3. An example library (2 or 3 picked by keywords in the objective) instead of one example per family.
4. Retry Qwen3-1.7B with a prompt tuned for it (it beats the Coder in the task reference, not in our agent).
5. Compare two versions task by task (right/wrong flips) to separate noise from real effect.

## 9. Privacy

The challenge rules forbid logging the objectives or files of platform runs. In production the agent prints only
durations (`[solve] prep … | … | total …`, labels in French) and the loaded model (`[init] …`).