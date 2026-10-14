# GridWorld Counting Reward Machines

An honours research project exploring whether an LLM can translate a natural-language
task into a **Counting Reward Machine (CRM)** that a reinforcement-learning agent can
learn. A CRM represents task stages with states, counters, transitions, and rewards.

For example: **“Collect five coins, then a blue key, then five more coins.”**

The generator produces an ordered event trace, object counts, and a CRM. Python builds
a 10×10 grid, validates the specification, and trains a tabular Q-learning agent.
The system compares the agent's trace and efficiency with the target. In iterative
mode, a judge LLM suggests CRM revisions after unsuccessful evaluations. MLflow stores
the runs; a Streamlit dashboard shows results and animations.

## Local setup

Use **Python 3.11** and Git. Clone this repository, open a terminal in its root, and
create a virtual environment:

**Windows PowerShell**

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks activation, use `.\.venv\Scripts\python.exe` in place of
`python` in the commands below; activation is optional.

**macOS / Linux**

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Dependencies include [PyCRM (`pyrewardmachines`)](https://github.com/TristanBester/pycrm),
which also installs PyTorch and Stable Baselines3. This project's tabular training
runs on the CPU; a GPU is not required. Use `pygame-ce`, as listed in the requirements,
rather than installing a second package that also provides `pygame`.

## Choose your LLMs

The **generator** writes the CRM; the **judge** proposes revisions. They can use
different providers and models. Set credentials in your terminal environment:

| Provider flag | API key variable | Generator | Judge |
| --- | --- | --- | --- |
| `gemini` | `GEMINI_API_KEY` | Yes | No |
| `openai` | `OPENAI_API_KEY` | Yes | Yes |
| `openrouter` | `OPENROUTER_API_KEY` | Yes | Yes |
| `groq` | `GROQ_API_KEY` | Yes | No |
| `nvidia` | `NVIDIA_API_KEY` | Yes | No |
| `ollama` | None; local server | Yes | No |

For example, to use Gemini for generation and OpenRouter for judging:

```powershell
$env:GEMINI_API_KEY = "your-gemini-key"
$env:OPENROUTER_API_KEY = "your-openrouter-key"
python system.py --generator gemini --generator-model YOUR_GEMINI_MODEL_ID --judge-provider openrouter --judge-model YOUR_OPENROUTER_MODEL_ID --task "Collect two coins, then reach the goal."
```

On macOS/Linux, set variables with `export GEMINI_API_KEY="your-gemini-key"`
and `export OPENROUTER_API_KEY="your-openrouter-key"` instead. Replace the model
placeholders with exact IDs available through your provider account. Model availability,
quotas, and API charges depend on that account; the checked-in defaults are research
settings, not guaranteed available models. The application reads environment variables
directly and **does not automatically load `.env` files**. Never commit real keys.

To use another supported provider, change the flags and set its key. For example:

```bash
python system.py --generator openai --generator-model YOUR_OPENAI_MODEL_ID --judge-provider openai --judge-model YOUR_JUDGE_MODEL_ID
```

For a local generator, install and start [Ollama](https://ollama.com), download a model,
and pass its installed tag:

```bash
ollama pull qwen2.5:3b
python system.py --generator ollama --generator-model qwen2.5:3b
```

Ollama connects at `http://localhost:11434/v1/`. This command still uses the configured
hosted judge, so configure that judge's key too. For a generator-only experiment, set
`ITERATIVE = False` in `system.py`; unsuccessful evaluations then end without a judge
call. See [model configuration](docs/models.md) for custom providers and comparisons.

## Run and configure experiments

Run all commands from the repository root:

```bash
python system.py --help
python system.py                         # Prompts for a task; uses configured defaults
python system.py --task "Ring the bell three times, then reach the goal."
python system.py 1                       # Same task with long and short prompts
python system.py 2                       # Long iterative, short iterative, long one-shot
```

The comparison modes also accept the provider/model flags above. Each comparison
starts a fresh run. An unsuccessful run is recorded and the next comparison continues.

- **`system.py`:** default providers/models, `ITERATIVE`, and `long` prompt selection.
- **`system_loop.py`:** training budget, exploration settings, evaluation interval,
  efficiency thresholds, and LLM call limits.
- **`system_specifications.py`:** the actual long/short generator and judge prompts.
- **`generatorformat.md` / `judgeformat.md`:** readable response-format examples.

Supported events are coins, red/blue/green/yellow keys, bell interactions, hazards,
and the goal. Walls and locations are generated by Python. New grids require
independent access to task objects to prevent forced out-of-order pickups.

Training currently allows 2,500 episodes of up to 200 steps, evaluating every 100
**episodes**. Success normally requires the exact event trace, step efficiency above
0.80, and reward efficiency above 0.99. If the exact route baseline is unavailable,
the run uses its full episode budget and qualifies by trace match alone; missing
efficiency metrics remain blank. Details are in [results](docs/results.md).

## Interactive playground (`system_fancy.py`)

After completing the local setup, launch the one-page playground from the repository root:

```bash
python system_fancy.py
```

The interface opens at **http://localhost:8502**. In PowerShell, you can also launch
it without activating the existing virtual environment:

```powershell
.\.venv\Scripts\python.exe system_fancy.py
```

Set `GEMINI_API_KEY` and the configured judge's key in the same terminal before
launching. The playground uses **Gemini 3.5 Flash Lite**, **iterative refinement**,
and **long specifications**. The judge follows `JUDGE_PROVIDER` / `JUDGE_LLM` in
`system.py`; the default OpenRouter judge requires `OPENROUTER_API_KEY`.

1. Read the object guide and valid-task examples on the right.
2. Choose an example to edit, or enter a task such as
   "Collect 3 coins, then a blue key, then ring the bell twice."
3. Select **Run task**. The generated grid appears during training, followed by
   an animated replay and the task result.
4. Expand **Environment specification and generated CRM** to inspect the generated
   definitions. Scroll down to see previous tasks.

One web experiment runs at a time. History persists in `outputs/fancy/`, with
experiment records also saved in MLflow. Existing terminal runs are available
through the results dashboard rather than the playground's history.

Optional launch settings:

```bash
python system_fancy.py --port 8503
python system_fancy.py --no-browser
```

The server listens locally at `127.0.0.1`. Keep the terminal open while using the
page, and press **Ctrl+C** to stop it. An experiment worker already in progress
may finish independently. No additional dependencies beyond `requirements.txt`
are needed.

## View and export results

```bash
python -m streamlit run dashboard.py
python -m scripts.export_report
```

The dashboard opens at `http://localhost:8501`. It supports task/model filters,
iteration details, failure reasons, animations, and CSV downloads. The report command
creates a timestamped snapshot in `outputs/reports/`, including an HTML preview,
CSV data, and an editable Quarto document. See [results and reporting](docs/results.md).

Default storage is `mlflow.db` plus the `mlruns/` artifact directory. Set
`MLFLOW_TRACKING_URI` and `MLFLOW_EXPERIMENT_NAME` to override the store and experiment.
Keep the database **and** artifacts together when backing up results; moving artifact
directories may require updating MLflow's stored paths.

## Repository layout

```text
system.py                  Command-line entry point
system_fancy.py            Launch the local GridWorld Studio website
system_loop.py             Generation, training, evaluation, and refinement loop
system_functions.py        LLM adapters, grid generation, CRM parsing, and Q-learning
system_specifications.py   Long and short prompts
generatorformat.md         Generator response example
judgeformat.md             Judge response example
dashboard.py               Streamlit results interface
grid_visualizer.py         Grid animation rendering
crm_support/               Tracking, failure diagnostics, and result analysis
scripts/                   Report export utility
templates/                 Quarto report template
web/                       Studio page and styling
docs/                      Usage details and historical research notes
tests/                     Offline regression tests and CRM fixtures
requirements.txt           Direct Python dependencies
```

Local virtual environments, credentials, databases, `mlruns/`, generated `outputs/`,
and `.local_archive/` are excluded from Git. A fresh clone starts without experiment
history. Research notes in `docs/research/` describe historical runs; their private
run artifacts are not included.

## Verify the installation

```bash
python system.py --help
python system_fancy.py --help
python -m scripts.export_report --help
python -m unittest discover -s tests
```

Tests use local fixtures, temporary stores, and mocked model responses; no LLM API
keys or paid calls are required. Some route-search and dashboard tests take longer
than unit tests. LLM output and learning success can vary across tasks and runs;
passing validation establishes that the specification is executable, not that
Q-learning will converge within the configured budget.
