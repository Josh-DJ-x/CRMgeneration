"""Failure descriptions shared by run logging and historical repairs."""


def effective_failure_category(record):
    """Recognize historical environment errors logged with no specific category."""
    reason = record.get("failure_reason")
    if isinstance(reason, str):
        diagnostic = reason.strip().lower()
        if (diagnostic.startswith("judge llm failed: openrouter judge (")
                and ("finish_reason='length'" in diagnostic
                     or "reached the output token budget" in diagnostic)):
            return "judge_output_budget"
    category = record.get("failure_category")
    if isinstance(category, str) and category.strip() not in {"", "other"}:
        return category
    reason = record.get("failure_reason")
    if not isinstance(reason, str):
        return "other"
    reason = reason.strip().lower()
    if reason.startswith("judge: typeerror: 'nonetype' object is not subscriptable"):
        return "judge_provider_error"
    # Match concrete diagnostics, including their old stage/exception prefixes,
    # rather than arbitrary mentions of environment failures in other errors.
    for prefix in ("environment_generation: valueerror: ",
                   "evaluation reward bounds: valueerror: ",
                   "environment generation failed: ",
                   "environment generation stopped: "):
        if reason.startswith(prefix):
            reason = reason[len(prefix):]
            break
    if reason.startswith(("invalid environment specification.",
                          "environment specification is not valid json.")):
        return "invalid_environment"
    if reason.startswith("no ordered route exists for the desired trace."):
        return "no_valid_environment"
    if reason.startswith(("environment generation could not find a valid layout",
                          "environment generation reached its ")):
        return "environment_generation_limit"
    return "other"


def failure_reason(record):
    """Keep explicit errors; otherwise describe recorded evaluation failures."""
    if str(record.get("success_status", "")).lower() in {"true", "1"}:
        return None
    explicit = record.get("failure_reason")
    if isinstance(explicit, str) and explicit.strip():
        return explicit.strip()
    history = record.get("iteration_history") or []
    final = history[-1] if history else {}
    info = record.get("run_information") or final.get("run_information") or {}
    desired = record.get("desired_trace", final.get("desired_trace"))
    actual = record.get("final_trace", info.get("trace", final.get("agent_trace")))
    parts = []
    if info.get("truncated"):
        parts.append(f"Evaluation reached the {info.get('steps', 'maximum')}-step limit without completion.")
        minimum = record.get("minimum_steps", info.get("minimum_steps"))
        try:
            if float(minimum) > float(info.get("steps")):
                parts.append(f"The shortest valid route requires {minimum} steps, exceeding the episode limit; "
                             "more training cannot make this task fit within that limit.")
        except (TypeError, ValueError):
            pass
    if desired is not None and actual is not None and list(desired) != list(actual):
        prefix = 0
        for expected, observed in zip(desired, actual):
            if expected != observed:
                break
            prefix += 1
        expected = desired[prefix] if prefix < len(desired) else "end of trace"
        observed = actual[prefix] if prefix < len(actual) else "end of trace"
        if not actual:
            parts.append(f"No events were collected; the desired trace requires {len(desired)} events.")
        else:
            parts.append(f"Event trace mismatch at event {prefix + 1}: expected {expected}, observed {observed}; "
                         f"matched {prefix}/{len(desired)} required events.")
    criterion = record.get("performance_criterion")
    if criterion in {"step_efficiency", "step_and_reward_efficiency"}:
        efficiency = record.get("step_efficiency", info.get("step_efficiency"))
        threshold = record.get("training_step_efficiency_threshold", 0.95)
        try:
            if float(efficiency) <= float(threshold):
                minimum_steps = record.get("minimum_steps", info.get("minimum_steps"))
                steps = record.get("steps_taken", info.get("steps"))
                parts.append(f"Step efficiency {float(efficiency):.3f} "
                             f"({minimum_steps} minimum / {steps} taken) "
                             f"must be greater than {float(threshold):.3f}.")
        except (TypeError, ValueError):
            pass
        if criterion == "step_and_reward_efficiency":
            efficiency = record.get("reward_efficiency", info.get("reward_efficiency"))
            threshold = record.get("training_reward_efficiency_threshold", 0.94)
            try:
                if float(efficiency) <= float(threshold):
                    parts.append(f"Reward efficiency {float(efficiency):.3f} "
                                 f"must be greater than {float(threshold):.3f}.")
            except (TypeError, ValueError):
                pass
    elif criterion != "trace_only":
        efficiency = record.get("reward_efficiency", info.get("reward_efficiency"))
        try:
            if float(efficiency) < 0.94:
                parts.append(f"Reward efficiency {float(efficiency):.3f} is below the required 0.940.")
        except (TypeError, ValueError):
            pass
    if parts:
        iterations = record.get("iterations", len(history))
        if str(record.get("iterative", "")).lower() in {"false", "0"}:
            parts.append("One-shot evaluation failed.")
        elif iterations:
            parts.append(f"Task remained unsuccessful after {iterations} evaluation iteration(s).")
        return " ".join(parts)
    return "Task failed; no detailed failure diagnostics were recorded."
