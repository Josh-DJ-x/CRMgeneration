# Viewing and writing up experiment results

The Streamlit dashboard reads the existing MLflow store. No MLflow web server is
needed for the default local database. Continue running `system.py` to record
experiments as before.

For comparison testing, `python system.py 1` prompts for a task once and runs it
with long and short specifications, using the `ITERATIVE` setting in `system.py`
for both (currently `False`). `python system.py 2` runs the same entered task in
this order: long iterative, short iterative, and long one-shot. Each run records
its own results for the dashboard. A failed run does not stop the remaining
comparisons; Ctrl+C stops the batch. `python system.py` keeps the configured
single-run behaviour.

The dashboard omits MLflow's mandatory empty `Default` experiment and the old
`MLflow Demo` entry, leaving the project experiment as the visible dataset.

## Start the dashboard

From `C:\University\HonoursProject\Code` in PowerShell:

```powershell
& .venv311\Scripts\python.exe -m pip install -r requirements-results.txt
& .venv311\Scripts\python.exe -m streamlit run dashboard.py --server.address 127.0.0.1
```

Open http://127.0.0.1:8501. The default database is `mlflow.db` beside the source
files. Set `MLFLOW_TRACKING_URI` to use another store; this also supports a remote
MLflow tracking server. Keep local SQLite paths absolute when working outside
the project directory. The existing artifact files must remain accessible.

- **Overview:** filter by exact task ID, experiment, model, iterative/one-shot mode,
  outcome and difficulty;
  inspect summary metrics, individual records, and runtime versus difficulty.
- **Experiment detail:** select a run, then choose an iteration to see its
  compact grid, agent results, traces, and judge feedback together. Expand
  the environment specification, full evaluation data, or CRM when needed.
  The final replay is in a separate expander. Earlier runs have only their
  final artifacts because iteration history was not recorded at the time.
- **Comparison & export:** compare task/model groups, choose iterations or another
  metric for the individual-run chart, inspect individual points,
  and download the currently filtered records and summaries as CSV.

Click **Refresh results** after recording new runs. Results are cached for up to
60 seconds. Empty filters for model/outcome mean all values. Missing outcomes
are shown as Unknown and excluded from success rates. Missing numeric values
are excluded from means. Reward efficiency is shown as the recorded ratio;
CRM failure is shown as a percentage: generations whose CRM cannot parse or
accept the desired trace, divided by all returned generator generations, times
100. Each generation is counted at most once, including corrections and
refinements. Provider calls with no response are excluded. Each record is weighted
equally when averaging percentages. Historical metrics retain their old definition.
The success flag is the system's recorded outcome, not the MLflow run status.
The sidebar Task ID field searches only `task_id`: entering `1` returns task 1,
not task 10 or records with 1 elsewhere. The table's built-in search is still a
general table search. Historical runs are classified as iterative. Runs missing
a difficulty score remain visible without a separate toggle.

Enable **Hide erroneous errors** in the sidebar to exclude rate-limited runs,
provider failures due to high demand, and invalid task specification rejections (including "Please specify a valid
task"). This applies to tables, metrics, comparisons, and filtered CSV exports.
Other failures remain visible. The toggle is off by default.
Filtering uses the recorded `failure_category` parameter (`rate_limit`,
`provider_capacity`, or `invalid_task`), independent of the displayed error text.
Historical runs without a category remain visible until explicitly categorized.

Unsuccessful tasks record a `failure_reason`. Completed evaluations describe
trace mismatches, step-limit exhaustion, and reward efficiency below 0.94.
Generation, provider, initialization, training, call-limit, and interruption
errors preserve their error message and execution stage. Successful runs do
not record a failure reason. Existing missing reasons can be filled from saved
artifacts with `experiment_results.backfill_failure_reasons(experiment_id)`;
existing nonempty reasons are preserved.

Rate limits at the judge or during CRM corrections/refinements also record the
task's difficulty score and available specifications. If a correction is rate
limited before the grid is built, the fixed environment is generated locally
for scoring. A rate limit on the first generation has no generated task to score.
In iterative mode, a CRM that still fails parsing or desired-trace validation
after three correction attempts is passed to the judge with its validation
diagnostic. Training is skipped for that CRM; judge feedback is used for the next
refinement. The existing iteration and provider-call limits still apply.

**Download database (.db)** in the sidebar creates a consistent SQLite snapshot
of the whole results database, independent of dashboard filters. Keep the
`mlruns` artifact directory with the database when archiving an experiment:
animations and JSON files are separate artifacts, not embedded in the database.

In **Experiment detail**, choose an individual run and click **Delete this task
run** to remove that record from the active results list. Other runs, including
other runs for the same task, remain available. This uses MLflow's soft deletion
and can be restored through MLflow. The dashboard refreshes after deletion.

The comparison chart keeps all filtered generator models on the x axis for
every metric. A missing metric has no point but does not remove the model label.

Set `JUDGE_LLM` in `system.py` to `"openrouter/free"` for random free routing,
or to an exact OpenRouter model ID to pin the judge. New runs log the requested
model separately from the actual model IDs reported for each response. The
dashboard displays `OpenRouter (model IDs)` and each iteration identifies its
judge. Legacy records without actual IDs are labelled as not recorded; runs
with no judge calls are labelled as not called. Pinning the model alone does
not guarantee identical outputs or provider execution on a later run.

Set `ITERATIVE` near the top of `system.py` for A/B tests. `True` allows up to
three evaluation attempts with judge feedback and CRM refinement. `False` runs
one generation/training/evaluation pass and makes no judge or refinement call
after evaluation. Existing runs are numbered by their recorded start time, and
new task labels continue from the largest recorded number.

Set `long` in `system.py` to select the prompt specification condition. `True`
uses the original specifications; `False` uses every corresponding `_SHORT`
constant. The selected condition is logged and can be filtered as Long or Short
in the dashboard. Existing runs are labelled Long because they used the original
specifications. Training hyperparameters are also logged on new runs.

Short prompts use states without counters for single-event ordering, explain
positional counter conditions and additive updates, and require the final GOAL
to enter the terminal state directly. A compact counting example consumes the
next stage's event after the counter reaches zero, avoiding an extra counted
event. Short prompt specifications stay within half the regular token budget;
prompt tokens exclude model responses and growing conversation history.
Gemini measured the final three-key initial prompt at 692 tokens versus 1,479
for the regular prompt (46.8%); the complete short specification set measured
913 versus 2,147 tokens (42.5%).

CRM validation replays the fixed trace and checks for premature GOAL termination.
Correction requests include the failed CRM, parser/replay error, fixed trace,
and syntax example. Alignment failures are logged with their diagnostic reason.

The current tabular Q-learning settings are a reasonable controlled baseline for
the 10x10 environment: 5,000 episodes, learning rate 0.1, discount factor 0.99,
epsilon 1.0 decaying by 0.999 to a 0.05 floor, and seed 42. The exploration rate
reaches its floor after about 2,994 episodes, leaving roughly 2,000 episodes for
mostly greedy refinement. Keep these fixed while comparing prompt lengths.

These settings are not evidence of convergence for every generated CRM. Larger
CRM/counter state spaces may need more than 5,000 episodes. The current result is
also based on one training seed and one evaluation episode. For research claims,
repeat each task/prompt condition across several seeds and report variation, or
first confirm from learning curves that the last episodes have stabilised. A
fixed seed is useful for a tightly controlled prompt comparison, but it does not
measure robustness across training randomness.

Task difficulty uses the following workload formula:

`workload = required_events + 0.5 * ordered_stage_changes + minimum_steps / 50
            + 4 * hazard_count / (hazard_count + 5)`

`difficulty = 1 - exp(-workload / 24)`

Required events exclude the final GOAL but include every coin, key, and bell
interaction. A stage is a consecutive group of the same event. This gives the
three-key task with a 38-step route a score of 0.180; the 48-event reference task
with repeated coins, keys, and bell rings scores approximately 0.9. These are
calibration anchors, not measured probabilities of failure. CRM complexity is
still recorded as diagnostic information but does not change task difficulty.
The minimum-step search respects event order and obstacles. If its search cannot
find a route, the route contribution is zero and `route_length_known` is false;
such a score is incomplete.

The generator now outputs just one compact `DESIRED_TRACE`, for example
`GREEN_KEY -> COIN x 3 -> BELL x 23 -> GOAL`, followed by ENVIRONMENT and CRM.
Python expands the sequence into events. Syntax and event names are validated;
there is no duplicate signature comparison, trace verification retry loop, or
trace verification failure metric. CRM parsing and execution corrections remain.

Generation time averages every CRM generator request (initial, refinement, and
correction, including responses whose CRM fails parsing). Judge calls have their
own average. Provider timings exclude MLflow span bookkeeping. Iteration 1 starts
before the initial generator request; subsequent iterations start before their
refinement request. Each iteration includes corrections, grid generation,
training, evaluation, and any judge feedback. There is no unused refinement
after the final evaluation.

The **Runtime breakdown** in Experiment detail shows total generation, judge,
training, grid generation, and misc time, followed by total system time.
Misc time combines setup, evaluation and reward bounds, difficulty calculation
during execution, and processing and tracing overhead.
Total system time starts when task input is recorded and ends when the task passes
or its attempts stop. Final difficulty calculation, animation export, and the final
MLflow result write are excluded. New runs use timing version `task-outcome-v3`;
historical recorded timings are unchanged. Per-call averages
must be multiplied by their call counts before comparing them with this total.
For older runs, misc time is inferred from the total minus the recorded activity
times when those values are available.

Reward efficiency is `(episode reward - minimum reward) / (maximum reward -
minimum reward)`. The minimum reward assumes the agent takes the CRM's ordinary
step reward for `max_steps - 1` steps, then triggers the worst non-goal failure
transition on the final step. If there is no such failure transition, it uses
the lowest CRM reward as the fallback final reward. The value is not clamped,
so an episode outside the estimated bounds can produce a result below 0 or above 1.

## Produce a research report

```powershell
& .venv311\Scripts\python.exe export_report.py
```

This creates a new timestamped directory in `outputs/reports` containing:

- `preview.html`: immediately viewable results tables, with no Quarto installation.
- `report.qmd`: editable research write-up with methodology and case-study prompts.
- `results.md`: generated results included by the Quarto document.
- `runs.csv`, `comparison.csv`, `provenance.json`: fixed data and run IDs for the snapshot.

The exporter includes **all active runs in the selected experiment**, independent
of dashboard filters. Use `--experiment "Name"` to select another experiment and
`--output "outputs/reports/my-study"` for a new named directory. Existing output
directories are rejected to preserve earlier snapshots. Dashboard CSV downloads
respect the current filters.

Install the [Quarto CLI](https://quarto.org/docs/get-started/) to render the edited
write-up (it is separate from the Python dependencies). Then run:

```powershell
quarto render outputs/reports/<snapshot>/report.qmd --to html
quarto render outputs/reports/<snapshot>/report.qmd --to docx
```

PDF rendering uses `--to pdf` and requires a TeX installation, such as Quarto's
`quarto install tinytex`. The template uses static snapshot data and needs no
Jupyter kernel. Render the exported template, whose directory contains
`results.md`, rather than the source template in `reports`.

Group comparisons by task and difficulty when assessing models; overall means
may reflect different task mixes. Identical difficulty scores do not establish
identical environments. Historical records cannot supply iteration histories,
seeds, or other details that were never logged.
