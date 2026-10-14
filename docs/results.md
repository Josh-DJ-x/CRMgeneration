# Results and reporting

Run commands from the repository root after installing `requirements.txt`.

## Dashboard

```bash
python -m streamlit run dashboard.py
```

Open `http://localhost:8501`. No separate MLflow web server is needed for the
local SQLite store. Use **Refresh results** after a run; cached data expires
within 60 seconds.

The dashboard provides:

- Task, model, outcome, prompt-length, and iterative/one-shot filters.
- Per-iteration grids, CRM specifications, desired/actual traces, and judge feedback.
- Final animations and runtime breakdowns.
- Grouped comparisons, difficulty charts, and filtered CSV downloads.
- Database snapshot download (artifacts must be backed up separately).
- Manual outcome corrections and run deletion controls; these modify the store.

## Storage and configuration

`mlflow.db` stores run metadata and metrics. `mlruns/` contains JSON, text,
traces, and animations. Both are local, ignored by Git, and required to preserve
complete experiment history. Existing MLflow artifact locations can be absolute;
copying or moving only the database does not relocate those artifacts.

Environment variables:

| Variable | Default | Purpose |
| --- | --- | --- |
| `MLFLOW_TRACKING_URI` | `sqlite:///mlflow.db` | Tracking database or remote server |
| `MLFLOW_EXPERIMENT_NAME` | `CRM GridWorld` | Experiment used for new runs |

Use an absolute SQLite URI if running from another directory. The simplest
workflow is to run both the experiment and dashboard from the repository root.

## What the metrics mean

- **Success:** normally the exact ordered event trace plus step efficiency
  strictly greater than 0.80 and reward efficiency strictly greater than 0.99.
- **Step efficiency:** shortest valid task route / steps actually taken.
- **Reward efficiency:** `(episode_reward - minimum_reward) /
  (maximum_reward - minimum_reward)`. Bounds come from the generated CRM and
  route baseline; this is not a probability and is not clamped to [0, 1].
- **CRM failure percentage:** returned generations whose CRM cannot parse or
  accept the required trace / all returned generations, multiplied by 100.
  Corrections and refinements count; a failed provider call with no response does not.
- **Runtime:** separates generation, training, judging, grid generation, and
  remaining overhead. Checkpoints count episodes, not environment steps.

If route search exceeds its budget, or any single collectible type has 17 or
more objects, the exact baseline is unavailable. Training uses the full episode
budget and success depends on trace match alone. Efficiency fields remain blank.
Mixed types are not added together for that threshold. Inspect the baseline flag
before comparing outcomes across these cases.

Training currently uses up to 2,500 episodes, 200 steps per episode, and a check
every 100 episodes. See `TRAINING_*` in `system_loop.py` for the source of truth.
Every CRM refinement starts training a fresh Q-table on the fixed grid.

Missing historical values are excluded from numeric means. Unknown outcomes do
not count in success-rate denominators. Identical task text or difficulty scores
do not guarantee identical generated CRMs or layouts. Include failures in model
comparisons, and match task sets, modes, seeds, and training settings.

## Export a fixed report snapshot

```bash
python -m scripts.export_report
python -m scripts.export_report --experiment "CRM GridWorld" --output outputs/reports/my-study
```

The exporter includes all active runs in the chosen experiment, independently
of dashboard filters. It creates a new directory and refuses to overwrite one.
The default location is `outputs/reports/<timestamp>/`.

- `preview.html`: standalone results tables.
- `report.qmd`: editable Quarto document, copied from `templates/report.qmd`.
- `results.md`: generated tables included by the Quarto document.
- `runs.csv`, `comparison.csv`, `provenance.json`: snapshot data and run IDs.

With the Quarto CLI installed separately, render the exported document:

```bash
quarto render outputs/reports/my-study/report.qmd --to html
quarto render outputs/reports/my-study/report.qmd --to docx
```

PDF output additionally requires a TeX installation. Edit the exported document
to add methodology and interpretation; the source template has no results data.

## Historical research notes

The notes in [research/](research/) describe previous failures and investigations.
They may reference older settings or private run IDs. Current behavior is defined
by the code and the configuration recorded with each run.
