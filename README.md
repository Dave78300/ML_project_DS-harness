# DS-Harness: a rule-first agent with an LLM fallback (challenge 194, ML-Arena)

**Team:** Merge_conflict · **End of project:** Mon 2027-01-04, end of day (scores frozen, slides sent) · **Oral:**
Wed 2027-01-06, 9:00–12:00 (one slot per team)

## TL;DR

- **What it is:** an agent around Qwen2.5-Coder-1.5B. The system reads and cleans the files, solves the task families
  it recognises with deterministic rules, and sends everything else to the LLM, which writes Python that the system
  runs, repairs and votes on.
- **Best private leaderboard result (reported):** **97.5 %** (117/120): no files 97.8 % (44/45), tables 96.0 % (48/50), fits 100 % (25/25),
  with `agent/agent_v10.py`.
- **Progress on the private set:** 51.7 → 74.2 → 89.2 → 90.8 → 92.5 (v1 to v5) → 93.3 → 94.2 → 95.8 →
  97.5 (v6 to v10, section 2.2).
- **Dates:** the end of the project is Monday 4 January 2027, end of day: scores are frozen at the last deployment that
  passed, and the slides (PDF) are sent the same day. The orals take place on Wednesday 6 January 2027, 9:00–12:00, one
  slot per team. v6 to v10 (5–6 November 2026) are therefore inside the project period. **Make `agent/agent_v10.py`
  the last deployment before the freeze:** any later deployment that passes, even an experiment, would set the frozen
  score.
- **How to read the numbers:** the public dev set is only a diagnostic. The deterministic rules were written from the
  dev objectives, so a perfect dev score is expected by construction. The private score is the measure that counts.
  Three private tasks are still missed (2 tables, 1 no-file); task-level private errors are not available.
- **Known weakness:** the rules are not fully strict. A blind audit run after v10 found wrong answers on 24 % of the
  no-file variants the rules answer; a guard built against this defect left the private score unchanged (section 2.4).

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
leaderboard mark is the private-set score frozen at the end of the project (Monday 4 January 2027, end of day), that is
the score of the last deployment that passed. If the conversion is
linear, one task is worth about 0.04 points of the final /20 grade (to be confirmed with the course).

Key dates:

| Date | Event |
|---|---|
| Mon 2027-01-04, end of day | End of project: scores frozen (last deployment that passed), slides (PDF) sent |
| Wed 2027-01-06, 9:00–12:00 | Orals, one slot per team |

Runtime constraints: a model of about 1.5B parameters, 40 seconds per call of 8 tasks, 15 calls, 3 CPUs, 3 GiB of RAM,
no network. A call that raises or exceeds its time limit fails the deployment.

## 2. Results

The table below shows ten milestones, in three phases. Each milestone is the best of several variants tested locally or
on the leaderboard, not the only system tried. Milestones are numbered in the order of the design steps, which is not always the order
in which the scores appeared on ML-Arena. Variants that did not help are listed in section 2.2.

| Phase | Versions | Approach | Private score | File |
|---|---|---|---:|---|
| 1 | v1 | The LLM writes and runs the code for every task | 51.7 | |
| 2 | v2 to v5 | Deterministic rules first, LLM as a fallback (v2: fit rules and execution fixes, called "stage 3" in earlier notes; v3: first no-file and table rules; v4: more no-file rules; v5: refined table rules) | 74.2 → 92.5 | `stage3.py` (v5) |
| 3 | v5b, v6 to v10 (5–6 November 2026) | Wider rule coverage and cross-checked rules (v5b: independent rewrite of all rules, not kept) | 93.3 → 97.5 | `agent_v10.py` (v10) |

"The v5 rules" below are the rules written in phase 2 and kept unchanged in `agent_v10.py`; "the corrected rules" are
the independent rule set written in v5b and corrected from v7 to v10.

### 2.1 Milestones of phases 1 and 2, v1 to v5 (reported by the team; aggregate scores only)

| Version | Score | No files | Tables | Fits | Tasks |
|---|---:|---:|---:|---:|---:|
| v1, initial agent | 51.7 | 75.6 | 40.0 | 32.0 | 62/120 |
| v2, fit rules and execution fixes | 74.2 | 77.8 | 60.0 | 96.0 | 89/120 |
| v3 | 89.2 | 91.1 | 84.0 | 96.0 | 107/120 |
| v4 | 90.8 | 95.6 | 84.0 | 96.0 | 109/120 |
| v5 (`stage3.py`) | 92.5 | 95.6 | 88.0 | 96.0 | 111/120 |

For reference, the challenge's own reference systems score 46.1 (Qwen2.5-1.5B), 53.3 (Qwen2.5-Coder-1.5B) and 62.2
(Qwen3-1.7B) on the public dev set. Those are dev scores; their private scores are not known.

### 2.2 Milestones of phase 3, v6 to v10 (5–6 November 2026)

Each milestone adds one main change; the private score is the only measurement of the LLM path, because no GPU was available
to run the model offline.

| Version | Change | Score | No files | Tables | Fits | Tasks |
|---|---|---:|---:|---:|---:|---:|
| v5b | All families re-solved by extended rules (not kept) | 92.5 | 97.8 | 84.0 | 100 | 111/120 |
| v6 | v5 path for tables, v5b rules for no files and fits | 93.3 | 97.8 | 86.0 | 100 | 112/120 |
| v7 | + 21 narrow table templates, veto on constructs the v5 rules misread | 94.2 | 97.8 | 88.0 | 100 | 113/120 |
| v8 | + two waves of no-file templates, thousands separators, answer checks on LLM output | 94.2 | 97.8 | 88.0 | 100 | 113/120 |
| v9 | + parser hints in the table prompts, 8 samples when ≤ 4 tasks reach the LLM | 95.8 | 97.8 | 92.0 | 100 | 115/120 |
| **v10** | **+ on a disagreement between the v5 and the corrected table rules, the corrected answer wins** | **97.5** | **97.8** | **96.0** | **100** | **117/120** |

Examples of variants that did not help (each submitted once on top of v9 or v10):

| Change | Score | Reading |
|---|---:|---|
| Vote weighted by use of the columns found by the parser | 95.8 | No effect |
| 12 samples, extended to ≤ 6 live tasks, second round on split votes | 95.8 | No effect |
| Hints for no-file tasks sent to the LLM | 95.8 | No effect |
| LLM arbitrates between the two table rule sets | 96.7 | One table instead of two |
| Corrected table rules also where the v5 rules abstain | 93.3 | Tables 86 %: generic rules misfire on private phrasings |
| Corrected table rules first everywhere (earlier version of those rules) | tables 84.0 | Tables 42/50 in two submissions |
| Guard after v10: a rule that did not read a modifier of the question votes, the LLM is consulted (`agent/experiments/agent_v10_guard.py`) | 97.5 | Unchanged in every family |

What these results show:
1. **Scores identical across five LLM-side variants** (hints, samples, weighting, second round, no-file hints) mean the
   remaining errors were not decided by the LLM. Most were wrong answers given by deterministic rules. Comparing two
   independent rule implementations and keeping the corrected one on disagreement (v10) recovered two of them.
2. **A wrong rule is worse than an abstention.** On generated variants, the v5 table rules answer 11 of 500 dev-like
   tasks wrongly, and 50 of 420 tasks of types absent from dev (for example an interquartile range computed as a range,
   "number of distinct carriers per depot" counted as rows, "not of the Lyon store" read as "of the Lyon store").
3. **The LLM with hints beats generic rules on unseen phrasings.** Extending the corrected rules to tasks the v5 rules
   leave to the LLM cost five table tasks, although it looked better on generated data. The generator does not cover
   every private phrasing; private results decide.
4. **One to two tasks of noise.** The LLM path varies between submissions (the composition of a batch changes the sampled
   tokens). Only v7, v9 and v10 are clearly above that noise.

### 2.3 Offline checks (dev and generated tasks, without the model)

| Check | Result |
|---|---|
| Dev, rules only | 180/180, no LLM call |
| Dev, full `solve()` with a stub LLM | 180/180, at most 1.5 s per batch of 8 |
| Generated tables, fresh draws (dev-like and new types) | 1,548/1,550 correct, 0 wrong (corrected rules) |
| 21 new table types, two draws | 420/420 and 420/420 |
| Generated no-file tasks (three template waves, thousands separators) | 100 % on every set, 0 wrong |
| Paraphrased dev objectives (synonyms, filler sentences, sentence order, rounding phrases) | tables 765/775, no files 542/542, fits 244/244 |
| No-file rules versus v7, 6,505 tasks | no task where v7 was right and the new version differs |
| Tables and fits versus the previous version | identical answers and identical LLM prompts on 1,285–1,685 tasks |
| Answer checks on the true answers of 9,811 tasks | 0 true answers rejected |

Generated tasks come from generators that rebuild the seven table domains (energy, hospital, loans, shipments, pupils,
flats, orders), the five fit domains and the nine no-file categories, with the dev formatting traps (`;` separator,
decimal comma, missing-value tokens, units in cells, `-1` sentinel, duplicates, day-first dates, key prefixes). The
truth is computed on the clean data, independently of the agent. A Good-Turing estimate on the dev sub-templates
suggests that about one third of the private no-file tasks come from sub-templates that dev does not contain, which is
why the template waves cover many forms instead of chasing one task.

### 2.4 Blind audit after v10 and the guard experiment

The generated tasks share the blind spots of the rules they were written with. After v10, a blind audit was therefore
run on variants written without reading the rule code (`audit/`):

| Audit | Result with `agent_v10.py` |
|---|---|
| 232 hand-written no-file variants (sibling sub-templates of dev) | 85 right, 120 left to the LLM, **27 wrong** (24 % of the 112 answers) |
| A modifier appended to generated table questions ("for the three largest values", ", doubled"…) | The rules keep the same answer for about 78 % of the modified questions |

The rules can answer without reading a modifier that changes the computation ("in minutes", "of the three longest",
"once a 30 % tax on the interest is paid"), and the two table rule sets then agree on the wrong answer, so the v10
cross-check cannot see it.

`agent/experiments/agent_v10_guard.py` adds a guard: when a rule answers although a risk word of the question sentence
was matched by none of its regular expressions, its answer becomes a vote (weight 2 for no-file tasks, 3 for tables) and
the LLM is consulted. It also falls back on the second table rule set when the LLM returns nothing.

| Check of the guard (no model) | Result |
|---|---|
| Dev (180), generated tables (3,040), known-good no-file tasks (3,538), fits (250) | 0 answer changed |
| False alarms on correct rule answers | 1 of 126 (no-file audits), 1 of 380 (paraphrased dev tables) |
| Wrong answers flagged, no-file audit 1 (blind) | 17 of 27 |
| Wrong answers flagged, no-file audit 2 (80 further variants) | 19 of 29 before the word list was extended, 26 of 29 after |
| Wrong answers flagged, modified table questions | about 70 % |
| **Private leaderboard** | **97.5 %, identical in every family** |

Reading: the defect is real, but it is not what decides the three missed private tasks, or the LLM does no better on
them. `agent_v10.py` therefore remains the final solution; the guard is kept as a documented negative result.

```bash
python audit/run_nofile_audit.py agent/agent_v10.py                       # 27 wrong answers on audit 1
python audit/run_nofile_audit.py agent/experiments/agent_v10_guard.py     # 17 of them flagged by the guard
```

## 3. Architecture

```text
task ──► route ──► normalise thousands separators ──► read files ──► rules ──► answer
         │                                                            │
         │                                                            └─ no rule ─► hints ─► generate ─► execute
         │                                                                                    ─► repair ─► checks ─► vote
         └─ round the answer as the objective asks
```

1. **Router:** no files → `nofile`; `train.csv` present → `fit`; otherwise → `table`.
2. **Robust reading:** separator detection, decimal commas, missing-value tokens (fixed list plus tokens quoted in the
   objective), glued units (`12 kg`, `12 half-days`, `12 m2`), class codes (`5A`, `5B`) kept as text, day-first dates,
   requested deduplication, `24,000`-style thousands separators in the objective.
3. **Tables, in order:**
   1. narrow templates for types absent from dev (share of a total, days between two dates, gap between two groups,
      percentage of missing values, top-k, mode, interquartile range, weekday and quarter filters, date ranges,
      negations, `A or B`, per-group standard deviation and distinct counts, weighted means, year-over-year change…);
   2. a veto: constructs the v5 rules are known to misread go straight to the LLM;
   3. the v5 rules (`stage3.py`, kept verbatim inside `agent_v10.py` under a `_v1_` prefix);
   4. a cross-check: when the corrected rules also answer and disagree, their answer is kept;
   5. the LLM, with the v5 prompts plus parser hints (resolved columns, population and counted rows of a proportion,
      filters), 8 samples when few tasks of the batch reach it.
4. **No files:** the v5 templates, then about 130 additional templates in three waves, with explicit exclusions (statistics
   of a sub-list, games with an entry fee and "interest earned" questions are not answered as plain cases). The rules are
   not fully strict: a blind audit after v10 showed that a rule can ignore a modifier it does not know (section 2.4).
5. **Fits:** target = the `train.csv` column missing from `test.csv`; features matched to the dictionary; year filter;
   unregularised linear or logistic regression; prediction, intercept, R², coefficient, odds ratio, count and mean
   questions; "least squares" and "regularization" spellings.
6. **LLM fallback:** Qwen2.5-Coder-1.5B, loaded through `dsh.load_llm`, writes Python from a short instruction, one
   worked example, the table schema, the dictionary and hints.
7. **Execution and recovery:** generated code runs in a watched thread with pandas, NumPy, SciPy, SymPy and
   scikit-learn. `pd.read_csv` returns the cleaned tables, `SmartFrame` recovers column-name mismatches, failed code is
   repaired from its error message, truncated code is continued.
8. **Answer checks and vote:** LLM candidates that break the form the question asks for (a fraction or probability
   outside [0, 1], a non-integer count, an invalid YYYYMMDD date, a weekday outside 1–7, a rank below 1, a negative
   spread) are discarded before the vote; rule answers are never touched. Greedy result weight 1.5, each sample weight
   1; errors are caught per task; internal budget of 29 s out of 40 s, every `generate` call bounded by `max_time`.

## 4. Defects found and fixes

| Defect | Effect | Fix |
|---|---|---|
| Generated code re-imported pandas and replaced the patched `read_csv` | Table tasks failed | Patch `pd.read_csv` temporarily so it survives imports |
| scikit-learn rejects NaN | Fit tasks failed | NaN-tolerant linear and logistic regression |
| Join keys written differently in the two files | `table.join` at 0 % on dev | Key detection and normalisation |
| Logistic regression regularised by default | Probabilities outside tolerance | Unregularised fit when the objective asks for it |
| Unbounded generation, signal-based timeout | `deadline exceeded` | `max_time`, preloaded imports, watched thread |
| A field whose name contains "of" (`length of stay`) cut to "stay" | Filter silently dropped, wrong answer | Several candidate phrases, nearest first, within the sentence |
| Apostrophe in `company's sites` read as an opening quote | Quoted values shifted, filter lost | Quotes must not be inside a word |
| `A discount of -1 means…` matched across sentences | Sentinel not applied | Sentence-bounded pattern |
| `no less than 5` read as `< 5` | Wrong filter | Explicit operator table, more synonyms |
| `(between 0 and 1)` read as a filter | Wrong answer | Ignored after "fraction" |
| `How many … ?` only recognised at the start of the objective | Filtered counts left to the LLM | Any sentence start |
| `shipped in 2022`, `listed in 2021` ignored | Year filter dropped | Participle → date column |
| `24,000`, `1,404`, `1 tonne = 1,000 kg` | Numbers truncated at the comma | Thousands separators removed before parsing |
| "How much interest has been earned" answered with the balance | Wrong answer | Interest-only questions handled |
| Hints that described a proportion's numerator as a filter | Misleading prompt | Population and counted rows labelled separately |

Known issues: the v5 code used for the table LLM path still keeps `LAST_ERR` between candidates (a repair prompt can quote
another task's error), and `.round()` on a Python float raises (the prompt only asks the model not to round).

## 5. Limits, risks and next steps

- **Three private tasks are still missed** (2 tables, 1 no-file). Their scores did not move with any LLM-side change, so
  they are most likely decided by rules; they cannot be inspected.
- **The rules are not strict enough:** the blind audit (section 2.4) found 24 % wrong answers among the no-file variants the
  rules answer, and table rules that ignore most appended modifiers. The guard built against this defect did not change
  the private score.
- **Selection on the leaderboard:** the milestones were chosen among many variants tested on the same private set. Each
  change was a single, explained hypothesis, but the private score now also reflects that selection.
- **The LLM path was never measured offline:** no GPU was available. A self-contained Kaggle notebook that compares LLM
  settings with the rules switched off (`DSH_RULES_OFF`) is the next step when a GPU is available.
- **Generated data is not the private set:** v6's drop shows that generated phrasings miss some private ones. Keep the
  private score as the arbiter and the generators as a regression net.
- **Noise:** one private task is 0.83 points; do not read a one-task change as a general improvement.

## 6. Files and local usage

| File | Role |
|---|---|
| `agent/agent_v10.py` | Best submission (97.5 %); submit it as `agent.py` with `dsh.py` |
| `agent/stage3.py` | v5 (92.5 %), kept for reference; its code is also embedded in `agent_v10.py` |
| `agent/experiments/agent_v10_guard.py` | v10 plus the guard of section 2.4 (97.5 %, same scores); negative result, not the final solution |
| `audit/` | Blind audit sets for the no-file rules and the script that runs them without the model |
| `tests/test_agent_v10.py` | Smoke tests of the deterministic layers of `agent_v10.py` (no model needed) |
| `tests/test_stage3.py` | Earlier tests; they import a `ml_project_ds_harness` package that is not in the repository |
| `dsh.py`, `localtest.py`, `scoring.py` | Challenge kit (submodule `ds-harness`): model loader, local evaluator, scoring |
| `dev.json` | Public dev set; download locally, excluded from Git |
| `report/` | Project report (PDF) |
| `slides oral presentation/` | Oral slides (PDF) and their HTML source |
| `README.md` | This report |

```bash
python -m pytest -q tests/test_agent_v10.py                # 11 tests, a few seconds, no GPU
python localtest.py agent/agent_v10.py --test-run -v       # the platform's 16-task test run
python localtest.py agent/agent_v10.py --out run.json      # all 180 dev tasks at the real 40 s limit
```

Environment variables, for local experiments only (the defaults are the submitted configuration):

| Variable | Default | Effect |
|---|---|---|
| `DSH_TABLE_MODE` | `v1_v2wins` | `v1` (v5 rules only), `v1_arbitrate` (LLM breaks ties), `v1_then_v2` and `v1_then_v2_v2wins` (corrected rules also where v5 abstains) |
| `DSH_HINTS` | `1` | Parser hints in the table prompts |
| `DSH_TABLE_SAMPLES` | `8` | Samples when ≤ `DSH_LIVE_MAX` tasks reach the LLM |
| `DSH_LIVE_MAX` | `4` | Threshold for the extra samples |
| `DSH_SANITY` | `1` | Answer checks on LLM candidates |
| `DSH_HINT_WEIGHT`, `DSH_EXTRA_ROUND`, `DSH_NOFILE_HINTS` | `0` | Variants that did not help (section 2.2) |
| `DSH_RULES_OFF` | empty | `table`, `nofile`, `fit` or `all`: switch rules off to measure the LLM alone; never in a submission |
| `DSH_DEBUG=1` | off | Prints generated code, errors and the schema seen by the model; never enable on the platform |
| `DSH_BUDGET`, `DSH_SAMPLES`, `DSH_MODEL` | 29, 4, Coder | Time budget, samples per task, model from `ALLOWED_MODELS` |

## 7. Deployment

Before submitting: unset every `DSH_*` variable and check memory.

```bash
/usr/bin/time -v python localtest.py agent/agent_v10.py --test-run   # "Maximum resident set size" below 3,000,000 KB
```

Submit `agent.py` (an exact copy of `agent/agent_v10.py`) and `dsh.py` with the PyTorch runtime. The file defines a
single `class Agent` and keeps `stage3.py`'s single `exec` call in the generated-code runner; every deployment passed
the platform's upload scan. Quota: five deployments per person per rolling 24 hours, failed ones included. Keep an
exact copy of every submitted file, so that the version behind each score can be reproduced.

Before the freeze (Monday 4 January 2027, end of day), finish with a deployment of `agent/agent_v10.py` that passes and
check its private score (97.5 %) on the leaderboard: the frozen score is that of the last deployment that passed, so an
experiment deployed after it would replace it. With the quota above, do not leave this to the last day.

## 8. Privacy

The challenge rules forbid logging objectives or files from platform runs. In production the agent prints only
durations and the loaded model; debug output exists only behind `DSH_DEBUG`.
