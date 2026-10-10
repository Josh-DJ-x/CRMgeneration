"""Application entry point and trace-based CRM generation/evaluation loop."""

import json
import os
import time

from semantic_specifications import select_specifications
from failure_diagnostics import failure_reason
from system_functions import (
    Gemini,
    Groq,
    GridWorldCrossProduct,
    GridWorldLabellingFunction,
    Ollama,
    OpenRouter,
    OpenRouterResponseError,
    OpenAILLM,
    EnvironmentGenerationError,
    RouteSearchLimitError,
    calculate_trace_reward_bounds,
    collectible_count,
    MINIMUM_STEP_COLLECTIBLE_LIMIT,
    combined_difficulty,
    crm_accepts_trace,
    crm_trace_alignment_error,
    crm_specification,
    crm_parse,
    extract_judge_reflection,
    extract_generation_trace,
    extract_refinement_crm,
    environment_difficulty,
    format_trace,
    format_compact_trace,
    generate_gridworld,
    parse_environment,
    run_agent,
    step_efficiency,
    traces_match,
    train_agent,
)
from grid_visualizer import export_grid_animation
from mlflow_tracking import (
    configure_tracking, log_experiment_run, log_generation_failure, next_task_id,
)


INVALID_TASK_FLAG = "<<<INVALID_TASK>>>"
MAX_GENERATOR_CALLS = 6
MAX_JUDGE_CALLS = 3
TRAINING_EPISODES = 2500
TRAINING_EVALUATION_INTERVAL = 100
TRAINING_STEP_EFFICIENCY_THRESHOLD = 0.8
TRAINING_REWARD_EFFICIENCY_THRESHOLD = 0.99
TRAINING_LEARNING_RATE = 0.1
TRAINING_DISCOUNT_FACTOR = 0.99
TRAINING_INITIAL_EPSILON = 1.0 
TRAINING_MIN_EPSILON = 0.02
TRAINING_EPSILON_DECAY = 0.997
TRAINING_SEED = 42
TRAINING_REWARD_SHAPING = False


def _performance_passed(run_information):
    """Require both route and reward efficiency to exceed their thresholds."""
    return (
        run_information["step_efficiency"] > TRAINING_STEP_EFFICIENCY_THRESHOLD
        and run_information["reward_efficiency"] > TRAINING_REWARD_EFFICIENCY_THRESHOLD
    )


def _parse_crm_with_error(crm_text):
    try:
        return crm_parse(crm_text), None
    except ValueError as error:
        return None, str(error)


def _serialise_run_information(run_information):
    """Remove display-only frames before supplying raw run data to an LLM."""
    raw_data = {
        key: value
        for key, value in run_information.items()
        if key != "grid_progression"
    }
    return json.dumps(raw_data, indent=2)


def _judge_followup_prompt(iteration, generation, raw_run_information, agent_trace):
    """Supply changed data while retaining instructions in judge chat history."""
    return (
        f"Iteration {iteration}: evaluate this revised CRM and latest run using "
        "the same desired trace, semantics, and qualification thresholds as before.\n\n"
        + "Generated CRM:\n\n" + generation + "\n\n"
        + "<<<RUN_INFORMATION>>>\n" + raw_run_information + "\n\n"
        + "Copy this latest agent-run trace exactly:\n" + format_trace(agent_trace)
        + "\n\nReturn <<<AGENT_TRACE>>>, <<<RUN_INFORMATION>>>, and <<<REFLECTION>>>. "
        "Copy the supplied trace and run JSON exactly. Diagnose trace differences "
        "and performance; suggest precise CRM edits when needed. "
        "Compare this revision with the previous iteration and your previous feedback "
        "in the conversation history. Explain which requested changes were applied "
        "and whether trace correctness, step_efficiency, and reward_efficiency improved, "
        "regressed, or stayed unchanged. Both efficiency thresholds still apply. "
        "When the trace is correct, distinguish inefficient learned behaviour from "
        "incorrect CRM logic before recommending further revisions."
    )


def is_invalid_task_response(response):
    """Return whether an LLM has rejected the task through the agreed sentinel."""
    return INVALID_TASK_FLAG in response.upper()


def _llm_capacity_message(error):
    """Return a user-facing reason for provider capacity failures, if applicable."""
    status = getattr(error, "status_code", None)
    if status is None:
        status = getattr(error, "code", None)
    if status is None:
        status = getattr(getattr(error, "response", None), "status_code", None)
    if status == 429 or str(status) == "429":
        return "rate limited"
    if status == 503 or str(status) == "503":
        return "temporarily unavailable due to high demand"
    detail = str(error).lower()
    if any(term in detail for term in (
        "rate limit", "rate_limit", "resource exhausted", "resource_exhausted",
        "too many requests", "quota exceeded",
    )):
        return "rate limited"
    if any(term in detail for term in (
        "high demand", "overloaded", "over capacity", "model is busy",
    )):
        return "temporarily unavailable due to high demand"
    return None


def main(gemini=None, openrouter=None, iterative=True, long=False, generator_backend="gemini",
         judge_model=None, task_description=None, generator_model=None, judge_backend="openrouter"):
    """Run a task and record every unsuccessful exit with a diagnostic reason."""
    prompt = select_specifications(long)
    if task_description is None:
        task_description = input(prompt["TASK_PROMPT"])
    started = time.perf_counter()
    model_setting, default_model = {
        "gemini": ("GEMINI_MODEL", "gemini-3.5-flash"),
        "ollama": ("OLLAMA_MODEL", "llama3.2"),
        "groq": ("GROQ_MODEL", "openai/gpt-oss-120b"),
        "openai": ("OPENAI_MODEL", "gpt-4.1-mini"),
        "openrouter": ("OPENROUTER_GENERATOR_MODEL", "qwen/qwen3-32b"),
    }.get(generator_backend, ("GEMINI_MODEL", generator_backend))
    context = {"stage": "initialization", "logged": False, "record": {
        "task_id": "unassigned", "task_description": task_description,
        "generator_model": getattr(gemini, "model", generator_model or os.getenv(model_setting, default_model)),
        "generator_backend": generator_backend, "judge_backend": judge_backend,
        "judge_model": getattr(openrouter, "model", judge_model or (
            os.getenv("OPENAI_JUDGE_MODEL", "gpt-4.1") if judge_backend == "openai"
            else os.getenv("OPENROUTER_MODEL", "openrouter/free"))),
        "iterative": iterative, "long": long, "generator_call_count": 0,
        "judge_call_count": 0, "iterations": 0, "average_generation_time": 0.0,
    }}
    try:
        return _run_task(task_description, prompt, started, context, gemini, openrouter,
                         iterative, long, generator_backend, judge_model, generator_model, judge_backend)
    except (Exception, SystemExit, KeyboardInterrupt) as error:
        if not context["logged"]:
            if isinstance(error, KeyboardInterrupt):
                reason = f"Task interrupted during {context['stage']}."
            elif isinstance(error, SystemExit):
                reason = str(error) or f"Task exited during {context['stage']}."
            else:
                reason = f"{context['stage']}: {type(error).__name__}: {error}"
            record = context["snapshot"]() if "snapshot" in context else dict(context["record"])
            record.update(failure_reason=reason,
                          failure_category=context.get("failure_category", "other"),
                          total_system_time=time.perf_counter() - started)
            try:
                log_generation_failure(record)
            except Exception as logging_error:
                print(f"MLflow failure logging failed: {logging_error}. Task failure: {reason}")
        raise


def _run_task(task_description, prompt, system_start, failure_context, gemini, openrouter,
              iterative, long, generator_backend, judge_model, generator_model=None, judge_backend="openrouter"):
    """Generate a CRM, train it, and verify one agent trace against the fixed trace."""
    if not task_description.strip():
        failure_context["failure_category"] = "invalid_task"
        raise SystemExit("Please specify a valid task")
    mlflow = configure_tracking()
    task_id = next_task_id(mlflow) if mlflow is not None else "1"
    failure_context["record"]["task_id"] = task_id
    if generator_backend not in ("gemini", "ollama", "groq", "openai", "openrouter"):
        raise ValueError("Unknown generator backend")
    if judge_backend not in ("openrouter", "openai"):
        raise ValueError("judge_backend must be 'openrouter' or 'openai'")
    generator = gemini or (
        Gemini(os.getenv("GEMINI_API_KEY"), model=generator_model or os.getenv("GEMINI_MODEL", "gemini-3.5-flash")) if generator_backend == "gemini" else
        Ollama(model=generator_model or os.getenv("OLLAMA_MODEL", "llama3.2")) if generator_backend == "ollama" else
        Groq(os.getenv("GROQ_API_KEY"), model=generator_model or os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")) if generator_backend == "groq" else
        OpenAILLM(os.getenv("OPENAI_API_KEY"), model=generator_model or os.getenv("OPENAI_MODEL", "gpt-4.1-mini")) if generator_backend == "openai" else
        OpenRouter(os.getenv("OPENROUTER_API_KEY"), model=generator_model or os.getenv("OPENROUTER_GENERATOR_MODEL", "qwen/qwen3-32b"))
    )
    openrouter = openrouter or (OpenAILLM(
        os.getenv("OPENAI_API_KEY"), model=judge_model or os.getenv("OPENAI_JUDGE_MODEL", "gpt-4.1"),
    ) if judge_backend == "openai" else OpenRouter(
        os.getenv("OPENROUTER_API_KEY"),
        model=judge_model or os.getenv("OPENROUTER_MODEL", "openrouter/free"),
    ))
    # Injectable providers may be reused by callers across independent tasks.
    # Reset once here, before any prompts; preserve history within the task.
    for provider in (generator, openrouter):
        reset_history = getattr(provider, "reset_history", None)
        if callable(reset_history):
            reset_history()
    requested_judge_model = getattr(openrouter, "model", type(openrouter).__name__)
    judge_fallback_used = False
    judge_chat_history = []

    generator_call_count = 0
    judge_call_count = 0
    judge_evaluation_count = 0
    crm_validation_failures = 0
    generation_times = []
    generation_history = []
    judge_times = []
    training_times = []
    grid_generation_times = []
    iteration_times = []
    iteration_history = []
    latest_training_metrics = None
    ground_env = None
    environment_spec = None
    desired_trace = ()
    crm = None
    judge_reflection = ""
    baseline_error = None

    def timed(callback, durations):
        start = time.perf_counter()
        try:
            return callback()
        finally:
            durations.append(time.perf_counter() - start)

    def finish_iteration(record, started_at):
        duration = time.perf_counter() - started_at
        record["time_seconds"] = duration
        iteration_times.append(duration)

    def traced_call(name, span_type, attributes, inputs, callback, output_summary=None,
                    propagate_capacity_error=False):
        """Run a unit of work and record an MLflow span when tracking is enabled."""
        failure_context["stage"] = name
        stage_messages = {
            "generator": "Generating...",
            "judge": "Judging...",
            "environment_generation": "Generating environment...",
            "training": "Training...",
            "evaluation": "Evaluating...",
        }
        if name in stage_messages:
            print(stage_messages[name], flush=True)
        try:
            if mlflow is None:
                return callback()
            with mlflow.start_span(name, span_type=span_type, attributes=attributes) as span:
                span.set_inputs(inputs)
                result = callback()
                span.set_outputs(output_summary(result) if output_summary else result)
                return result
        except Exception as error:
            reason = _llm_capacity_message(error) if span_type == "LLM" else None
            if propagate_capacity_error and reason is not None:
                raise
            if reason is None:
                if name == "judge" and span_type == "LLM":
                    failure_context["failure_category"] = (
                        "judge_output_budget" if isinstance(error, OpenRouterResponseError)
                        and error.finish_reason == "length" else "judge_provider_error"
                    )
                    raise SystemExit(f"Judge LLM failed: {error}") from None
                raise
            failure_context["failure_category"] = (
                "rate_limit" if reason == "rate limited" else "provider_capacity"
            )
            raise SystemExit(
                f"{name.capitalize()} LLM is {reason}. Please try again later. Details: {error}"
            ) from None

    def call_generator(prompt):
        nonlocal generator_call_count
        if generator_call_count >= MAX_GENERATOR_CALLS:
            stop_generation(
                f"Generator call limit reached ({MAX_GENERATOR_CALLS}) for this task.",
                completed_iterations=len(iteration_history),
            )
        generator_call_count += 1
        response = traced_call(
            "generator", "LLM",
            {"model": getattr(generator, "model", type(generator).__name__)},
            {"prompt": prompt},
            lambda: timed(lambda: generator.generate(prompt), generation_times),
        )
        generation_history.append({
            "call": generator_call_count,
            "kind": "initial" if generator_call_count == 1 else "correction or refinement",
            "response": response,
        })
        return response

    def call_judge(prompt):
        nonlocal judge_call_count, judge_evaluation_count, openrouter, judge_fallback_used
        if judge_evaluation_count >= MAX_JUDGE_CALLS:
            raise SystemExit(
                f"Judge call limit reached ({MAX_JUDGE_CALLS}) for this task."
            )
        judge_evaluation_count += 1

        def attempt(allow_fallback):
            nonlocal judge_call_count
            judge_call_count += 1
            return traced_call(
                "judge", "LLM",
                {"model": getattr(openrouter, "model", type(openrouter).__name__),
                 "max_tokens": getattr(openrouter, "max_tokens", None)},
                {"prompt": prompt},
                lambda: timed(lambda: openrouter.generate(prompt), judge_times),
                propagate_capacity_error=allow_fallback,
            )

        can_fallback = not judge_fallback_used and getattr(openrouter, "model", None) != "openrouter/free"
        try:
            judgement = attempt(can_fallback)
        except Exception as error:
            reason = _llm_capacity_message(error)
            if not can_fallback or reason is None:
                raise
            print(f"Judge is {reason}; switching to openrouter/free for the rest of this run.", flush=True)
            try:
                fallback = OpenRouter(os.getenv("OPENROUTER_API_KEY"), model="openrouter/free")
            except Exception as setup_error:
                failure_context["failure_category"] = "judge_provider_error"
                raise SystemExit(f"Cannot initialize fallback judge: {setup_error}") from None
            fallback.messages = list(judge_chat_history)
            fallback.resolved_models = list(getattr(openrouter, "resolved_models", []))
            openrouter = fallback
            judge_fallback_used = True
            judgement = attempt(False)
        judge_chat_history.extend([
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": judgement},
        ])
        return judgement

    initial_prompt = (
        prompt["CRM_GENERATION_PROMPT"]
        + "\n\n" + prompt["GENERATION_FORMAT"] + "\n\n"
        + prompt["ENVIRONMENT_CONTEXT"] + "\n\n" + prompt["ENVIRONMENT_SEMANTICS"] + "\n\n"
        + prompt["CRM_FORMAT"] + "\n\nThis is the FIRST generation, not a refinement. "
        "For a valid task, return all three sections in order: "
        "<<<DESIRED_TRACE>>>, <<<ENVIRONMENT>>>, <<<CRM>>>.\n\nTask:\n\n"
        + task_description
    )
    def crm_failure_percentage():
        # Count returned generations, excluding provider calls with no response.
        return (100.0 * crm_validation_failures / len(generation_history)
                if generation_history else 0.0)

    def validate_crm(crm_text):
        """Count each generated CRM once if parsing or trace acceptance fails."""
        nonlocal crm_validation_failures
        crm, parse_error = _parse_crm_with_error(crm_text)
        accepts_trace = crm is not None and crm_accepts_trace(crm, desired_trace)
        if not accepts_trace:
            crm_validation_failures += 1
        return crm, parse_error, accepts_trace

    def validate_refinement(response):
        nonlocal crm_validation_failures
        try:
            crm_text = extract_refinement_crm(response)
        except ValueError as error:
            crm_validation_failures += 1
            return response, None, str(error), False
        return crm_text, *validate_crm(crm_text)

    def validate_generation(response):
        nonlocal desired_trace, environment_spec, crm_validation_failures
        nonlocal baseline_error
        if environment_spec is not None:
            return validate_refinement(response)
        try:
            trace, crm_text, environment_text = extract_generation_trace(response, initial=True)
            specification = parse_environment(environment_text)
        except ValueError as error:
            crm_validation_failures += 1
            return response, None, str(error), False
        desired_trace, environment_spec = trace, specification
        items = collectible_count(specification)
        if items >= MINIMUM_STEP_COLLECTIBLE_LIMIT:
            baseline_error = (
                f"Skipped minimum-step calculation: {items} collectible items "
                f"meet the {MINIMUM_STEP_COLLECTIBLE_LIMIT}-item threshold."
            )
            print(baseline_error + " Training will use the full episode budget and trace-only qualification.", flush=True)
        elif baseline_error and baseline_error.startswith("Skipped minimum-step calculation:"):
            baseline_error = None
        return crm_text, *validate_crm(crm_text)

    def initial_repair_prompt(issue):
        return (
            initial_prompt + "\n\nRepair the previous response for this same task. "
            "Return all three sections: <<<DESIRED_TRACE>>>, <<<ENVIRONMENT>>>, <<<CRM>>>. "
            "The trace must contain only supported EVENT or EVENT x positive_count stages "
            "joined by ->, ending with GOAL. State numbers and terminal annotations belong "
            "in the CRM, never in DESIRED_TRACE. Preserve the user's task intent.\n"
            "ENVIRONMENT must have exactly coins, keys, bells and hazards. Coins, bells "
            "and each red/blue/green/yellow key count must be nonnegative integers; "
            "hazards must be boolean. At most 84 object cells fit, including 7 hazard "
            "cells when enabled. Bells counts physical bell tiles, not the number of "
            "rings: one bell can be rung repeatedly.\n"
            "Validation failure:\n" + str(issue) + "\n\nPrevious response:\n" + generation
        )

    def failure_record(message, completed_iterations=0):
        record = {
            "task_id": task_id,
            "task_description": task_description,
            "generator_model": getattr(generator, "model", type(generator).__name__),
            "generator_backend": generator_backend,
            "judge_backend": "openai+openrouter" if judge_fallback_used and judge_backend == "openai" else judge_backend,
            "judge_model": requested_judge_model,
            "judge_fallback_used": judge_fallback_used,
            "judge_models": list(getattr(openrouter, "resolved_models", [])),
            "iterative": iterative,
            "long": long,
            "failure_reason": message,
            "failure_stage": failure_context.get("stage", "unknown"),
            "generation_history": list(generation_history),
            "iteration_history": list(iteration_history),
            "failure_category": failure_context.get("failure_category", "other"),
            "generator_call_count": generator_call_count,
            "crm_parse_fail_percentage": crm_failure_percentage(),
            "judge_call_count": judge_call_count,
            "iterations": completed_iterations,
            "total_system_time": time.perf_counter() - system_start,
            "average_generation_time": (
                sum(generation_times) / len(generation_times) if generation_times else 0.0
            ),
        }
        difficulty_env = ground_env
        if (difficulty_env is not None or
                (environment_spec is not None and failure_context.get("failure_category")
                 in {"rate_limit", "provider_capacity"})):
            try:
                # A provider may stop a correction before the first grid is built.
                # Reconstruct the fixed task locally so its score can still be logged.
                if difficulty_env is None:
                    difficulty_env = generate_gridworld(
                        seed=42, environment_json=json.dumps(environment_spec),
                        task_trace=desired_trace,
                    )
                difficulty = combined_difficulty(
                    difficulty_env.grid.tolist(), environment_spec, crm, desired_trace,
                    **({"calculate_route": False} if baseline_error else {})
                )
                record.update(
                    overall_difficulty_score=difficulty["difficulty_score"],
                    specifications={
                        "difficulty": difficulty,
                        "environment_spec": environment_spec,
                        "crm_specification": crm_specification(crm) if crm is not None else None,
                    },
                )
            except Exception as error:
                # Preserve the original task error if result preparation fails.
                print(f"Failure difficulty calculation skipped: {error}")
        record.update(
            desired_trace=list(desired_trace),
            final_trace=list(iteration_history[-1]["agent_trace"]) if iteration_history else [],
            judge_reflection=judge_reflection,
        )
        if latest_training_metrics is not None:
            record.update(latest_training_metrics)
        return record
    failure_context["snapshot"] = lambda: failure_record("", len(iteration_history))

    def stop_generation(message, completed_iterations=0):
        try:
            record = failure_record(message, completed_iterations)
            log_generation_failure(record)
            failure_context["logged"] = True
        except Exception as error:
            print(f"MLflow logging skipped: {error}")
        raise SystemExit(message)

    def judge_iteration(iteration_record, generation, raw_run_information, agent_trace):
        judge_prompt = (
            "Generated CRM:\n\n" + generation + "\n\n"
            + "Desired trace:\n" + format_trace(desired_trace) + "\n\n"
            + prompt["TRAIN_RESULTS"] + "\n\n"
            + f"Qualification requires an exact trace match and minimum_steps / steps "
              f"> {TRAINING_STEP_EFFICIENCY_THRESHOLD} and reward_efficiency "
              f"> {TRAINING_REWARD_EFFICIENCY_THRESHOLD}. Both ratios must qualify.\n\n"
            + "<<<RUN_INFORMATION>>>\n" + raw_run_information + "\n\n"
            + prompt["REFLECTION_FORMAT"] + "\n\n" + prompt["FEEDBACK_FORMAT"] + "\n\n"
            + prompt["ENVIRONMENT_SEMANTICS"] + "\n\n" + prompt["CRM_FORMAT"] + "\n\n"
            + "Copy this agent-run trace into your AGENT_TRACE section exactly:\n"
            + format_trace(agent_trace)
        )
        if iteration_record["iteration"] > 1:
            judge_prompt = _judge_followup_prompt(
                iteration_record["iteration"], generation, raw_run_information, agent_trace,
            )
        if baseline_error:
            judge_prompt += (
                "\n\nShortest-route baseline is unavailable. Efficiency values are null, "
                "not zero; qualification for this run requires only an exact desired-trace "
                "match. Diagnose trace failures without requiring efficiency thresholds."
            )
        if iteration_record.get("validation_error"):
            judge_prompt += (
                "\n\nTask:\n" + task_description
                + "\n\nTraining and evaluation were skipped because generation validation failed "
                "after three correction attempts. Diagnose the supplied validation_error "
                "and suggest precise repairs to the trace, environment or CRM as needed. "
                "No agent run or efficiency measurements are available for this iteration. "
                "An empty desired trace means initial parsing failed; it is not a valid target. "
                "Once validated and buildable, the desired trace and environment are fixed.\nEnvironment specification:\n"
                + json.dumps(environment_spec, indent=2)
            )
        judgement = call_judge(judge_prompt)
        iteration_record["judge_feedback"] = extract_judge_reflection(judgement)
        iteration_record["judge_model"] = getattr(openrouter, "last_response_model", None)
        return judgement

    def prepare_generation(build_for_diagnostics=False):
        nonlocal ground_env, environment_spec, desired_trace, crm, parse_error, accepts_trace
        nonlocal env, reward_bounds, crm_validation_failures
        nonlocal baseline_error
        if environment_spec is None or (not accepts_trace and not build_for_diagnostics):
            return
        stage = "environment_generation"
        failure_context["stage"] = stage
        try:
            if ground_env is None:
                started = time.perf_counter()
                try:
                    ground_env = traced_call(
                        stage, "TOOL", {}, {"environment_spec": environment_spec},
                        lambda: generate_gridworld(seed=42, environment_json=json.dumps(environment_spec),
                                                   task_trace=desired_trace),
                        lambda result: environment_difficulty(result.grid.tolist(), environment_spec, desired_trace,
                                                              calculate_route=False),
                    )
                finally:
                    grid_generation_times.append(time.perf_counter() - started)
            if not accepts_trace:
                return
            stage = "evaluation reward bounds"
            failure_context["stage"] = stage
            env = GridWorldCrossProduct(ground_env=ground_env, crm=crm,
                                       lf=GridWorldLabellingFunction(ground_env), max_steps=200)
            if baseline_error:
                reward_bounds = {"shortest_valid_steps": None, "minimum_reward": None, "maximum_reward": None}
            else:
                reward_bounds = calculate_trace_reward_bounds(crm, desired_trace, env.max_steps, ground_env.grid)
        except RouteSearchLimitError as error:
            if stage == "environment_generation" or env is None:
                failure_context["failure_category"] = "route_search_limit"
                raise SystemExit(f"Task stopped during shortest-route calculation: {error}") from None
            baseline_error = str(error)
            reward_bounds = {"shortest_valid_steps": None, "minimum_reward": None, "maximum_reward": None}
            print("Shortest-route baseline unavailable; training for the full episode budget "
                  "and qualifying by trace match only. Efficiency metrics will be blank.", flush=True)
        except (EnvironmentGenerationError, ValueError, LookupError, TypeError, NameError, ArithmeticError) as error:
            # These diagnostics describe unusable generated content, not an agent
            # evaluation. Feed them back through the same bounded repair loop.
            if accepts_trace:
                crm_validation_failures += 1
            parse_error = f"{stage}: {type(error).__name__}: {error}"
            crm, accepts_trace = None, False
            if stage == "environment_generation" or str(error) == "No ordered route exists for the desired trace.":
                environment_spec, ground_env, desired_trace = None, None, ()
            failure_context["stage"] = "generation validation"

    env = reward_bounds = None
    first_iteration_start = time.perf_counter()
    print("\nIteration 1", flush=True)
    generation = call_generator(initial_prompt)
    print("Validating generation...", flush=True)
    failure_context["stage"] = "initial generation validation"
    if is_invalid_task_response(generation):
        failure_context["failure_category"] = "invalid_task"
        stop_generation("Please specify a valid task")
    crm_text, crm, parse_error, accepts_trace = validate_generation(generation)

    performance = False
    trace_correct = False
    status = False
    run_information = None
    iterations = 1
    completed_iterations = 0

    while iterations <= 3:
        iteration_start = first_iteration_start if iterations == 1 else time.perf_counter()
        completed_iterations += 1
        if iterations > 1:
            print(f"\nIteration {iterations}", flush=True)
            refinement_prompt = (
                prompt["REFINEMENT_CONTEXT"] + "\n\nDesired trace:\n"
                + format_compact_trace(desired_trace) + "\n\nEnvironment specification (fixed):\n"
                + json.dumps(environment_spec, indent=2) + "\n\nJudge response:\n\n" + judgement + "\n\n"
                + "Raw run information:\n" + raw_run_information
            )
            if environment_spec is None:
                refinement_prompt = initial_repair_prompt(parse_error) + "\n\nJudge response:\n" + judgement
            generation = call_generator(refinement_prompt)
            print("Validating generation...", flush=True)
            failure_context["stage"] = "CRM refinement validation"
            crm_text, crm, parse_error, accepts_trace = validate_generation(generation)
        if iterations > 1 and accepts_trace:
            ground_env = None
        prepare_generation()
        parse_fails = 0
        while not accepts_trace and parse_fails < 3:
            parse_fails += 1
            if crm is None:
                issue = parse_error or prompt["FAIL_PARSE"]
            else:
                issue = crm_trace_alignment_error(crm, desired_trace)
            print(f"Correcting generation (attempt {parse_fails}/3): {issue}", flush=True)
            correction_prompt = (
                "Validation failure:\n" + str(issue)
                + "\n\nDESIRED_TRACE:\n" + format_compact_trace(desired_trace)
                + "\n\nPrevious CRM:\n" + crm_text
                + "\n\n" + prompt["COMMON_PITFALLS"]
            )
            if environment_spec is None:
                correction_prompt = initial_repair_prompt(issue)
            generation = call_generator(correction_prompt)
            failure_context["stage"] = "CRM correction validation"
            crm_text, crm, parse_error, accepts_trace = validate_generation(generation)
            prepare_generation()
        if not accepts_trace and iterative:
            prepare_generation(build_for_diagnostics=True)
        if not accepts_trace and not iterative:
            issue = parse_error if crm is None else crm_trace_alignment_error(crm, desired_trace)
            stop_generation(
                ("Initial generation could not be parsed" if environment_spec is None
                 else "CRM could not be aligned with DESIRED_TRACE")
                + " after three corrections. " + str(issue),
                completed_iterations=len(iteration_history),
            )

        if not accepts_trace:
            issue = parse_error if crm is None else crm_trace_alignment_error(crm, desired_trace)
            raw_run_information = json.dumps({
                "evaluation_skipped": True,
                "validation_error": str(issue),
                "trace": [],
            }, indent=2)
            iteration_record = {
                "iteration": iterations,
                "environment_spec": environment_spec,
                "grid": ground_env.grid.tolist() if ground_env is not None else [],
                "crm_specification": crm_specification(crm) if crm is not None else None,
                "crm_text": crm_text,
                "desired_trace": list(desired_trace),
                "agent_trace": [],
                "run_information": json.loads(raw_run_information),
                "validation_error": str(issue),
                "trace_correct": False,
                "performance_passed": False,
                "training_episodes_completed": 0,
                "training_evaluations": [],
                "judge_feedback": None,
            }
            iteration_history.append(iteration_record)
            if iterations >= 3:
                finish_iteration(iteration_record, iteration_start)
                stop_generation(
                    ("Initial generation could not be parsed" if environment_spec is None
                     else "CRM could not be aligned with DESIRED_TRACE")
                    + " after three corrections. " + str(issue),
                    completed_iterations=len(iteration_history),
                )
            print("Generation validation failed after three corrections; passing to judge.", flush=True)
            judgement = judge_iteration(iteration_record, generation, raw_run_information, ())
            judge_reflection = iteration_record["judge_feedback"]
            finish_iteration(iteration_record, iteration_start)
            iterations += 1
            continue
        training_evaluations = []
        checkpoint_run = None

        def record_step_metrics(info):
            info["minimum_steps"] = reward_bounds["shortest_valid_steps"]
            info["step_efficiency"] = step_efficiency(info["minimum_steps"], info["steps"]) if not baseline_error else None
            if baseline_error:
                info.update(reward_efficiency=None, minimum_reward=None, maximum_reward=None)
            return info

        def stop_training(q_table, completed_episodes):
            nonlocal checkpoint_run, latest_training_metrics
            checkpoint_run = run_agent(env, q_table, reward_bounds, capture_progression=True)
            record_step_metrics(checkpoint_run)
            latest_training_metrics = {
                "step_efficiency": checkpoint_run.get("step_efficiency"),
                "reward_efficiency": checkpoint_run.get("reward_efficiency"),
            }
            trace_correct = traces_match(desired_trace, checkpoint_run["trace"])
            efficiency = checkpoint_run["step_efficiency"]
            satisfied = _performance_passed(checkpoint_run) and trace_correct
            training_evaluations.append({
                "episodes": completed_episodes,
                "reward_efficiency": checkpoint_run["reward_efficiency"],
                "steps_taken": checkpoint_run["steps"],
                "minimum_steps": checkpoint_run["minimum_steps"],
                "step_efficiency": efficiency,
                "trace_correct": trace_correct,
                "stopped_early": satisfied and completed_episodes < TRAINING_EPISODES,
            })
            print(f"Training checkpoint {completed_episodes}/{TRAINING_EPISODES}: "
                  f"step efficiency={efficiency:.4f}, "
                  f"reward efficiency={checkpoint_run['reward_efficiency']:.4f}, "
                  f"steps={checkpoint_run['steps']}/{checkpoint_run['minimum_steps']} minimum, "
                  f"trace match={trace_correct}"
                  + ("; training complete" if satisfied else "; criteria not met"))
            return satisfied

        training_start = time.perf_counter()
        q_table, episode_returns = traced_call(
            "training", "TOOL", {
                "episodes": TRAINING_EPISODES,
                "learning_rate": TRAINING_LEARNING_RATE,
                "discount_factor": TRAINING_DISCOUNT_FACTOR,
                "initial_epsilon": TRAINING_INITIAL_EPSILON,
                "min_epsilon": TRAINING_MIN_EPSILON,
                "epsilon_decay": TRAINING_EPSILON_DECAY,
                "seed": TRAINING_SEED,
                "reward_shaping": TRAINING_REWARD_SHAPING,
                "evaluation_interval": TRAINING_EVALUATION_INTERVAL,
                "step_efficiency_threshold": TRAINING_STEP_EFFICIENCY_THRESHOLD,
                "reward_efficiency_threshold": TRAINING_REWARD_EFFICIENCY_THRESHOLD,
            },
            {"task_id": task_id},
            lambda: train_agent(
                env,
                episodes=TRAINING_EPISODES,
                learning_rate=TRAINING_LEARNING_RATE,
                discount_factor=TRAINING_DISCOUNT_FACTOR,
                epsilon=TRAINING_INITIAL_EPSILON,
                min_epsilon=TRAINING_MIN_EPSILON,
                epsilon_decay=TRAINING_EPSILON_DECAY,
                seed=TRAINING_SEED,
                reward_shaping=TRAINING_REWARD_SHAPING,
                evaluation_interval=TRAINING_EVALUATION_INTERVAL,
                stop_training=stop_training if not baseline_error else None,
            ),
            lambda result: {
                "episodes": len(result[1]),
                "average_return_last_100": (
                    sum(result[1][-100:]) / len(result[1][-100:]) if result[1] else 0.0
                ),
            },
        )
        training_times.append(time.perf_counter() - training_start)

        run_information = traced_call(
            "evaluation", "TOOL", {"max_steps": env.max_steps},
            {"desired_trace": list(desired_trace)},
            lambda: record_step_metrics(checkpoint_run if checkpoint_run is not None else run_agent(
                env, q_table, reward_bounds, capture_progression=True
            )),
            lambda result: {
                "trace": list(result["trace"]),
                "reward_efficiency": result["reward_efficiency"],
                "minimum_steps": result["minimum_steps"],
                "step_efficiency": result["step_efficiency"],
                "steps": result["steps"],
            },
        )
        latest_training_metrics = {
            "step_efficiency": run_information.get("step_efficiency"),
            "reward_efficiency": run_information.get("reward_efficiency"),
        }
        performance = _performance_passed(run_information) if not baseline_error else traces_match(desired_trace, run_information["trace"])
        agent_trace = run_information["trace"]
        raw_run_information = _serialise_run_information(run_information)

        # Correctness comes from the environment's actual trace, not an LLM's
        # transcription of it.  The judge remains part of the failure loop,
        # where it provides natural-language feedback for CRM refinement.
        trace_correct = traces_match(desired_trace, agent_trace)
        print(f"Evaluation: step efficiency={run_information['step_efficiency']}, "
              f"reward efficiency={run_information['reward_efficiency']}, "
              f"steps={run_information['steps']}/{run_information['minimum_steps']} minimum, "
              f"trace match={trace_correct}, performance passed={performance}", flush=True)
        iteration_record = {
            "iteration": iterations,
            "environment_spec": environment_spec,
            "grid": ground_env.grid.tolist(),
            "crm_specification": crm_specification(crm),
            "desired_trace": list(desired_trace),
            "agent_trace": list(agent_trace),
            "run_information": json.loads(raw_run_information),
            "trace_correct": trace_correct,
            "performance_passed": performance,
            "training_episodes_completed": len(episode_returns),
            "training_evaluations": training_evaluations,
            "judge_feedback": None,
        }
        iteration_history.append(iteration_record)
        if trace_correct and performance:
            status = True
            finish_iteration(iteration_record, iteration_start)
            break

        if not iterative or iterations >= 3:
            finish_iteration(iteration_record, iteration_start)
            break

        judgement = judge_iteration(iteration_record, generation, raw_run_information, agent_trace)
        judge_reflection = iteration_record["judge_feedback"]

        finish_iteration(iteration_record, iteration_start)
        iterations += 1

    # Stop at the task outcome; result preparation and animation are untimed.
    total_system_time = time.perf_counter() - system_start
    failure_context["stage"] = "postprocessing"
    print("Preparing results...", flush=True)
    difficulty = combined_difficulty(
        ground_env.grid.tolist(), environment_spec, crm, desired_trace,
        **({"calculate_route": False} if baseline_error else {})
    )
    specifications = {
        "difficulty": difficulty,
        "environment_spec": environment_spec,
        "crm_specification": crm_specification(crm),
    }
    try:
        animation_path, is_gif = export_grid_animation(
            run_information["grid_progression"], bell_rings=run_information["bell_rings"]
        )
        animation_message = (
            f"Animation: {animation_path}" if is_gif
            else f"PNG frames: {animation_path} (install Pillow to export a GIF)"
        )
    except Exception as error:
        animation_path = None
        animation_message = f"Animation export failed: {error}"

    timing_breakdown = {
        "total_generation_time_seconds": sum(generation_times),
        "total_judge_time_seconds": sum(judge_times),
        "total_training_time_seconds": sum(training_times),
        "total_grid_world_generation_time_seconds": sum(grid_generation_times),
    }
    timing_breakdown["misc_time_seconds"] = max(
        0.0, total_system_time - sum(timing_breakdown.values())
    )

    def average(values):
        return sum(values) / len(values) if values else 0.0

    mlflow_record = {
        "task_id": task_id,
        "task_description": task_description,
        "success_status": status,
        "generation_history": generation_history,
        "generator_model": getattr(generator, "model", type(generator).__name__),
        "generator_backend": generator_backend,
        "judge_backend": "openai+openrouter" if judge_fallback_used and judge_backend == "openai" else judge_backend,
        "judge_model": requested_judge_model,
        "judge_fallback_used": judge_fallback_used,
        "judge_models": list(getattr(openrouter, "resolved_models", [])),
        "iterative": iterative,
        "long": long,
        "training_episodes": TRAINING_EPISODES,
        "training_evaluation_interval": TRAINING_EVALUATION_INTERVAL,
        "training_step_efficiency_threshold": TRAINING_STEP_EFFICIENCY_THRESHOLD,
        "training_reward_efficiency_threshold": TRAINING_REWARD_EFFICIENCY_THRESHOLD,
        "performance_criterion": "trace_only" if baseline_error else "step_and_reward_efficiency",
        "efficiency_baseline_available": not bool(baseline_error),
        "efficiency_baseline_error": baseline_error or "",
        "training_learning_rate": TRAINING_LEARNING_RATE,
        "training_discount_factor": TRAINING_DISCOUNT_FACTOR,
        "training_initial_epsilon": TRAINING_INITIAL_EPSILON,
        "training_min_epsilon": TRAINING_MIN_EPSILON,
        "training_epsilon_decay": TRAINING_EPSILON_DECAY,
        "training_seed": TRAINING_SEED,
        "training_reward_shaping": TRAINING_REWARD_SHAPING,
        "iterations": completed_iterations,
        "generator_call_count": generator_call_count,
        "judge_call_count": judge_call_count,
        "crm_parse_fail_percentage": crm_failure_percentage(),
        "average_training_time": average(training_times),
        "average_grid_world_generation_time": average(grid_generation_times),
        "average_generation_time": average(generation_times),
        "average_judge_time": average(judge_times),
        "average_iteration_time": average(iteration_times),
        "total_system_time": total_system_time,
        "timing_breakdown": timing_breakdown,
        "timing_version": "task-outcome-v3",
        "specifications": specifications,
        "judge_reflection": judge_reflection,
        "iteration_history": iteration_history,
        "overall_difficulty_score": difficulty["difficulty_score"],
        "episode_reward": run_information["episode_reward"],
        "maximum_reward": run_information["maximum_reward"],
        "minimum_reward": run_information["minimum_reward"],
        "steps_taken": run_information["steps"],
        "minimum_steps": run_information["minimum_steps"],
        "step_efficiency": run_information["step_efficiency"],
        "reward_efficiency": run_information["reward_efficiency"],
        "desired_trace": list(desired_trace),
        "final_trace": list(run_information["trace"]),
    }
    mlflow_record["failure_reason"] = failure_reason(mlflow_record)
    try:
        mlflow_run_id = log_experiment_run(mlflow_record, animation_path)
        mlflow_message = f"MLflow run ID: {mlflow_run_id}"
    except RuntimeError as error:
        mlflow_message = f"MLflow logging skipped: {error}"

    print("Success" if status else "Failure")
    if mlflow_record["failure_reason"]:
        print(f"Failure reason: {mlflow_record['failure_reason']}")
    print(f"Iterations: {completed_iterations}")
    print(f"Generator call count: {generator_call_count}")
    print(f"Judge call count: {judge_call_count}")
    print(f"Iterative: {iterative}")
    print(f"Long specifications: {long}")
    print(f"CRM failure percentage (parsing or trace acceptance): {mlflow_record['crm_parse_fail_percentage']:.2f}%")
    print("Specifications:")
    print(json.dumps(specifications, indent=2))
    print(f"Overall Difficulty Score: {difficulty['difficulty_score']}")
    print(f"Episode Reward: {run_information['episode_reward']}")
    print(f"Maximum Reward: {run_information['maximum_reward']}")
    print(f"Steps taken: {run_information['steps']}")
    print(f"Minimum steps: {run_information['minimum_steps']}")
    print(f"Step Efficiency: {run_information['step_efficiency']}")
    print(f"Reward Efficiency: {run_information['reward_efficiency']}")
    print(f"Final Trace: {format_trace(run_information['trace'])}")
    print(animation_message)
    print(mlflow_message)


if __name__ == "__main__":
    main()
