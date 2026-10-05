"""Compatibility entry point for the CRM system.

The implementation lives in ``system_functions``, ``semantic_specifications``,
and ``system_loop``.
"""

import argparse

from semantic_specifications import select_specifications
from system_loop import main

# Developer A/B toggle. False runs one generation/training/evaluation pass and
# skips judge feedback and CRM refinement after an unsuccessful evaluation.
ITERATIVE = True
# Prompt-length A/B toggle. False selects every *_SHORT specification.
long = False

# Set to "gemini", "ollama", or "groq" to select the generator.
GENERATOR_LLM = "gemini"

# Pin the judge to a free reasoning model for consistent comparison runs.
JUDGE_LLM = "nvidia/nemotron-3-ultra-550b-a55b:free"

# Judge Models
# JUDGE_LLM = "nvidia/nemotron-3-ultra-550b-a55b:free"

def cli(argv=None):
    parser = argparse.ArgumentParser(description="Run a CRM task or compare run settings.")
    parser.add_argument(
        "mode", nargs="?", choices=("1", "2"),
        help="1: long/short with the configured ITERATIVE setting; "
             "2: long iterative, short iterative, long one-shot",
    )
    args = parser.parse_args(argv)
    if args.mode is None:
        return main(iterative=ITERATIVE, long=long, generator_backend=GENERATOR_LLM,
                    judge_model=JUDGE_LLM)

    settings = ([(True, ITERATIVE), (False, ITERATIVE)] if args.mode == "1"
                else [(True, True), (False, True), (True, False)])
    task_description = input(select_specifications(settings[0][0])["TASK_PROMPT"])
    if not task_description.strip():
        parser.error("Please specify a valid task")
    failed = False
    for index, (use_long, iterative) in enumerate(settings, start=1):
        print(f"\nComparison run {index}/{len(settings)}: "
              f"long={use_long}, iterative={iterative}", flush=True)
        try:
            main(iterative=iterative, long=use_long, task_description=task_description,
                 generator_backend=GENERATOR_LLM, judge_model=JUDGE_LLM)
        except (Exception, SystemExit) as error:
            # main records failed runs too. Continue so every setting is tested.
            failed = True
            print(f"Run stopped: {error}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(cli())
