"""Run CRM experiments from the terminal:

  python system.py       One task using the ITERATIVE and long settings below.
  python system.py 1     Same task with long then short prompts; keeps ITERATIVE.
  python system.py 2     Same task: long iterative, short iterative, long one-shot.

Each comparison run starts fresh; failed runs are logged and the next continues.
All modes prompt for a task unless --task "your task" is supplied. Override the
generator with --generator gemini|ollama|groq|openai|openrouter and choose its
exact ID with --generator-model. Override the judge with --judge-provider
openrouter|openai and --judge-model. API keys come from environment variables;
paid model IDs incur API charges. Local Ollama models must already be installed.

Example: python system.py --generator openrouter --generator-model qwen/qwen3-32b
Use python system.py --help for all options. Loop logic lives in system_loop.
"""

import argparse

from semantic_specifications import select_specifications
from system_loop import main

# Developer A/B toggle. False runs one generation/training/evaluation pass and
# skips judge feedback and CRM refinement after an unsuccessful evaluation.
ITERATIVE = True
# Prompt-length A/B toggle. False selects every *_SHORT specification.
long = False
# Select "gemini", "ollama", "groq", "openai", or "openrouter".
GENERATOR_LLM = "gemini"
JUDGE_PROVIDER = "openrouter"

# Pin the judge to a free reasoning model for consistent comparison runs.
JUDGE_LLM = "nvidia/nemotron-3-ultra-550b-a55b:free"

# Judge Models
# nvidia/nemotron-3-ultra-550b-a55b:free
# google/gemma-4-31b-it:free
# stealth/space-bunny-alpha
# poolside/laguna-s-2.1:free
# "openai/gpt-oss-120b:free"

def cli(argv=None):
    parser = argparse.ArgumentParser(description="Run a CRM task or compare run settings.")
    parser.add_argument(
        "mode", nargs="?", choices=("1", "2"),
        help="1: long/short with the configured ITERATIVE setting; "
             "2: long iterative, short iterative, long one-shot",
    )
    parser.add_argument("--generator", choices=("gemini", "ollama", "groq", "openai", "openrouter"), default=GENERATOR_LLM)
    parser.add_argument("--generator-model", help="Exact generator model ID, e.g. qwen2.5:7b")
    parser.add_argument("--judge-provider", choices=("openrouter", "openai"), default=JUDGE_PROVIDER)
    parser.add_argument("--judge-model", help="Exact judge model ID")
    parser.add_argument("--task", help="Task text; skip the interactive prompt")
    args = parser.parse_args(argv)
    options = {"generator_backend": args.generator,
               "judge_model": args.judge_model or (JUDGE_LLM if args.judge_provider == "openrouter" else "gpt-4.1")}
    if args.generator_model is not None:
        options["generator_model"] = args.generator_model
    if args.judge_provider != "openrouter":
        options["judge_backend"] = args.judge_provider
    if args.task is not None:
        options["task_description"] = args.task
    if args.mode is None:
        try:
            return main(iterative=ITERATIVE, long=long, **options)
        except (Exception, SystemExit) as error:
            # main logs the failed run before propagating its diagnostic.
            print(f"Run stopped: {error}", flush=True)
            return 1

    settings = ([(True, ITERATIVE), (False, ITERATIVE)] if args.mode == "1"
                else [(True, True), (False, True), (True, False)])
    task_description = args.task if args.task is not None else input(select_specifications(settings[0][0])["TASK_PROMPT"])
    if not task_description.strip():
        parser.error("Please specify a valid task")
    failed = False
    options["task_description"] = task_description
    for index, (use_long, iterative) in enumerate(settings, start=1):
        print(f"\nComparison run {index}/{len(settings)}: "
              f"long={use_long}, iterative={iterative}", flush=True)
        try:
            main(iterative=iterative, long=use_long, **options)
        except (Exception, SystemExit) as error:
            # main records failed runs too. Continue so every setting is tested.
            failed = True
            print(f"Run stopped: {error}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(cli())
