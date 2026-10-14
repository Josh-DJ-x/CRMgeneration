"""Snapshot recorded results into an editable Quarto report and an HTML preview."""

import argparse
from datetime import datetime, timezone
import html
import json
from pathlib import Path

from crm_support.experiment_results import ROOT, experiments, load_runs, summarize


def markdown_table(frame):
    def cell(value):
        return html.escape(str(value)).replace("|", "&#124;").replace("\n", " ")
    rows = [list(frame.columns), ["---"] * len(frame.columns)]
    rows.extend(frame.fillna("Unavailable").astype(str).values.tolist())
    return "\n".join("| " + " | ".join(map(cell, row)) + " |" for row in rows)


def export_report(experiment="CRM GridWorld", destination=None, uri=None):
    found = next((eid for eid, name in experiments(uri) if name == experiment), None)
    if found is None:
        raise ValueError(f"Experiment not found: {experiment}")
    runs = load_runs(found, uri)
    if runs.empty:
        raise ValueError(f"No runs in experiment: {experiment}")
    destination = Path(destination or ROOT / "outputs" / "reports" / datetime.now().strftime("%Y%m%d-%H%M%S-%f"))
    destination.mkdir(parents=True, exist_ok=False)
    summary = summarize(runs, ["iterative_mode", "specification_length", "generator_model", "judge_model", "task_description", "overall_difficulty_score"])
    runs.to_csv(destination / "runs.csv", index=False)
    summary.to_csv(destination / "comparison.csv", index=False)
    created = datetime.now(timezone.utc).isoformat()
    (destination / "provenance.json").write_text(json.dumps({
        "experiment": experiment, "experiment_id": found, "created_at_utc": created,
        "run_ids": runs.run_id.tolist(), "scope": "All active runs in the experiment at export time",
    }, indent=2), encoding="utf-8")
    known = int(runs.success.count())
    successes = int(runs.success.sum())
    outcome = f"{successes}/{known} ({successes / known:.1%})" if known else "Unavailable (no known outcomes)"
    description = (
        f"Snapshot created at {created}. Includes {len(runs)} recorded runs. "
        f"Success: {outcome}. Unknown outcomes: {len(runs) - known}. "
        "Missing values are excluded from means. Success rate in the comparison table is a fraction (0–1). "
        "CRM parsing or trace acceptance failure percentages are averaged per run, not pooled across generations. "
        "Comparisons group by run mode, exact task description, difficulty score, generator and judge. "
        "Matching descriptions and difficulty scores do not guarantee identical environments or random seeds."
    )
    columns = ["task_id", "task_description", "iterative_mode", "specification_length", "outcome", "generator_model", "judge_model",
               "total_system_time_seconds", "reward_efficiency", "environment_minimum_steps_to_complete_trace",
               "crm_parse_fail_percentage", "run_id"]
    sections = "## Recorded results\n\n" + description + "\n\n## Task and model comparisons\n\n"
    sections += markdown_table(summary.round(4)) + "\n\n## Individual runs\n\n" + markdown_table(runs[columns]) + "\n"
    (destination / "results.md").write_text(sections, encoding="utf-8")
    template = (ROOT / "templates" / "report.qmd").read_text(encoding="utf-8")
    (destination / "report.qmd").write_text(template, encoding="utf-8")
    preview = """<!doctype html><html lang="en"><meta charset="utf-8">
<title>CRM experiment results</title><style>
body{font:16px system-ui;margin:3rem;color:#183047}h1,h2{color:#12566b}
table{border-collapse:collapse;font-size:13px}td,th{padding:10px;border:1px solid #d7e0e5;text-align:left}
th{background:#eaf2f5}.table{overflow:auto}p{max-width:1000px;line-height:1.6}
</style><h1>CRM experiment results</h1>"""
    preview += f"<p>{html.escape(description)}</p><h2>Task and model comparisons</h2>"
    preview += '<div class="table">' + summary.round(4).to_html(index=False, escape=True) + "</div>"
    preview += '<h2>Individual runs</h2><div class="table">' + runs[columns].to_html(index=False, escape=True) + "</div>"
    preview += "<p>Edit report.qmd to add methodology, interpretation, and selected case studies.</p></html>"
    (destination / "preview.html").write_text(preview, encoding="utf-8")
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", default="CRM GridWorld")
    parser.add_argument("--output", type=Path, help="New output directory; existing directories are never overwritten")
    args = parser.parse_args()
    print(export_report(args.experiment, args.output))
