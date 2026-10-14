# Model configuration and extension

See the [README](../README.md#choose-your-llms) for keys and provider selection.
The CLI accepts exact model IDs; no source change is needed to try another model
on a supported provider. Use IDs supported by the selected adapter and your account.

## Provider behavior

| Provider | Adapter behavior |
| --- | --- |
| Gemini | Google GenAI chat sessions |
| OpenAI | Responses API; conversation output retained for subsequent corrections |
| OpenRouter | OpenAI-compatible chat completions; resolved model IDs recorded |
| Groq | OpenAI-compatible chat completions; one delayed retry for the recognized TPM-limit response |
| NVIDIA | OpenAI-compatible chat completions with reasoning and output-budget options |
| Ollama | Local OpenAI-compatible chat completions at `http://localhost:11434/v1/` |

Each task starts with fresh conversations. Corrections, refinements, and judge
fallback retain context within that task. Supported judge adapters are OpenRouter
and OpenAI. Certain judge capacity errors trigger fallback according to
`system_loop.py`; inspect recorded resolved models when comparing runs.

NVIDIA options:

```bash
python system.py --generator nvidia --generator-model YOUR_MODEL_ID --nvidia-reasoning low --nvidia-max-tokens 16384
```

Reasoning choices are `high`, `low`, and `none`; the output budget includes
reasoning tokens. The adapter is configured for NVIDIA's Nemotron-style reasoning
options, so another model may require adapting those request fields.

The CLI flags take precedence over defaults. Default model IDs and mode toggles
are near the top of `system.py`. The provider factory in `system_loop.py` also
supports provider-specific model environment variables when a model is not passed
explicitly; the CLI supplies explicit defaults for some providers and the judge.
Prefer `--generator-model` and `--judge-model` for reproducible experiments.

## Local models

Start Ollama, install a model with `ollama pull <tag>`, and pass the exact tag to
`--generator-model`. A local generator still uses the configured hosted judge.
For generator-only runs, set `ITERATIVE = False` in `system.py`. The judge adapter
is currently initialized even in one-shot mode, so configure its key as required
by that adapter, although no judging request is made.

For comparisons, hold the task set, judge, prompt length, iteration mode, training
seed, context window, output budget, and quantization constant. Repeat runs and
record failures. Model weights and runtime memory requirements differ; choose a
local model that fits the machine. Temperature zero does not guarantee identical
outputs across executions.

## Add a provider

Adapters live in `system_functions.py` and implement the `LLM` interface:

- `model`: a model identifier used in run metadata.
- `generate(prompt: str) -> str`: return plain answer text, preserving the context
  needed for corrections within the run. Raise an exception for unusable responses.
- `reset_history()`: clear task-specific state. The base class clears `messages`
  and optional response-model metadata; override it for provider-owned sessions.

To expose a new adapter in the CLI, add its name to `system.py`'s choices, then
update validation, the model/key configuration, and construction in
`system_loop.py`. Add mocked provider tests in `tests/test_model_providers.py`.

For a Python experiment, the loop also accepts injected adapters through its
historically named `gemini` (generator) and `openrouter` (judge) arguments:

```python
from system_loop import main

# generator and judge are instances of your adapters.
main(
    gemini=generator,
    openrouter=judge,
    task_description="Collect two coins, then reach the goal.",
    iterative=True,
    long=False,
)
```

Injection overrides construction, but the provider labels still come from the
backend arguments. For a newly supported CLI provider, update those labels too
so experiment metadata identifies the adapter correctly.

Response structure must follow [generatorformat.md](../generatorformat.md) and
[judgeformat.md](../judgeformat.md). The prompts actually sent are selected from
`system_specifications.py`. Initial responses define the desired trace and object
counts; refinements normally replace only the CRM. The parser also accepts the
compact `EVENT x count -> EVENT` trace format used by the short prompts.
